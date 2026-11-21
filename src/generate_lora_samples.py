"""Génère des images avec et sans LoRA à partir des légendes de test."""
import argparse
import csv
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline
from PIL import Image, ImageDraw

MODELE = "sd-legacy/stable-diffusion-v1-5"


def lire_prompts(chemin):
    """Liste de (nom de l'image réelle, légende) du jeu de test."""
    with open(chemin, newline="", encoding="utf-8") as f:
        return [(l["file_name"], l["text"]) for l in csv.DictReader(f)]


def generer(pipe, prompt, seed, etapes, guidance):
    graine = torch.Generator("cuda").manual_seed(seed)  # même graine = même point de départ
    return pipe(prompt, num_inference_steps=etapes, guidance_scale=guidance,
                generator=graine).images[0]


def ligne(images, etiquettes, taille=256):
    """Une rangée d'images avec une étiquette au-dessus de chacune."""
    toile = Image.new("RGB", (taille * len(images), taille + 18), "white")
    dessin = ImageDraw.Draw(toile)
    for i, (im, et) in enumerate(zip(images, etiquettes)):
        toile.paste(im.resize((taille, taille)), (i * taille, 18))
        dessin.text((i * taille + 4, 3), et, fill="black")
    return toile


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--test", default="data/raw/lora_test/metadata.csv")
    p.add_argument("--lora", required=True, help="dossier contenant pytorch_lora_weights.safetensors")
    p.add_argument("--out", default="results/lora/samples")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    p.add_argument("--etapes", type=int, default=30)
    p.add_argument("--guidance", type=float, default=7.5)
    a = p.parse_args()

    sortie = Path(a.out)
    sortie.mkdir(parents=True, exist_ok=True)
    dossier_test = Path(a.test).parent
    prompts = lire_prompts(a.test)

    pipe = StableDiffusionPipeline.from_pretrained(
        MODELE, torch_dtype=torch.float16, safety_checker=None,
        requires_safety_checker=False).to("cuda")

    images, journal = {}, []
    for condition in ["base", "lora"]:
        if condition == "lora":
            pipe.load_lora_weights(a.lora)  # on colle le post-it sur le gros livre
        for fichier, prompt in prompts:
            for seed in a.seeds:
                im = generer(pipe, prompt, seed, a.etapes, a.guidance)
                nom = f"{condition}_{Path(fichier).stem}_s{seed}.png"
                im.save(sortie / nom)
                images[(condition, fichier, seed)] = im
                journal.append({"condition": condition, "image_reelle": fichier,
                                "prompt": prompt, "seed": seed, "fichier": nom})
                print(f"{nom} : {prompt}")

    with open(sortie / "samples.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(journal[0]))
        w.writeheader()
        w.writerows(journal)

    lignes = []
    for fichier, prompt in prompts:
        reelle = Image.open(dossier_test / fichier).convert("RGB")
        cases = [reelle] + [images[(c, fichier, s)] for c in ("base", "lora") for s in a.seeds]
        etiquettes = ["reel (test)"] + [f"{c} s{s}" for c in ("base", "lora") for s in a.seeds]
        lignes.append(ligne(cases, etiquettes))
    total = Image.new("RGB", (lignes[0].width, sum(l.height for l in lignes)), "white")
    y = 0
    for l in lignes:
        total.paste(l, (0, y))
        y += l.height
    total.save(sortie / "comparison.png")
    print(f"Écrit : {sortie / 'comparison.png'}")
    print(f"VRAM max (PyTorch) : {torch.cuda.max_memory_allocated() / 1e9:.2f} Go")