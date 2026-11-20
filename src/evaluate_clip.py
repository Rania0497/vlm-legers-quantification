"""CLIPScore (sans références) : la légende colle-t-elle à l'image ?"""
import csv
import statistics
from pathlib import Path

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from src.evaluate_text import DOSSIER, MODELES, PRECISIONS, REFS

IMAGES = Path("data/raw/flickr8k_subset/images")
REPO = "openai/clip-vit-base-patch32"
PREFIXE = "A photo depicts "  # préfixe recommandé pour CLIPScore
POIDS = 2.5                   # facteur de mise à l'échelle de CLIPScore


def lire_legendes_modele(cle, precision):
    """image -> légende brute du modèle."""
    chemin = DOSSIER / f"captions_{cle}_{precision}.csv"
    with open(chemin, newline="", encoding="utf-8") as f:
        return {l["image"]: l["legende"] for l in csv.DictReader(f)}


def lire_legendes_humaines():
    """image -> première légende humaine (niveau de référence)."""
    with open(REFS, newline="", encoding="utf-8") as f:
        return {l["image_filename"]: l["caption"].strip()
                for l in csv.DictReader(f) if l["caption_index"] == "1"}


def clipscores(model, processor, noms, textes, device, lot=32):
    """Une note CLIPScore par paire (image, texte)."""
    notes = []
    for d in range(0, len(noms), lot):
        images = [Image.open(IMAGES / n).convert("RGB") for n in noms[d:d + lot]]
        phrases = [PREFIXE + t for t in textes[d:d + lot]]
        entrees = processor(text=phrases, images=images, return_tensors="pt",
                            padding=True, truncation=True).to(device)
        with torch.no_grad():
            sortie = model(**entrees)
        cos = (sortie.image_embeds * sortie.text_embeds).sum(dim=-1)  # déjà normalisés
        notes += (POIDS * torch.clamp(cos, min=0)).tolist()
    return notes


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Calcul sur : {device}")
    processor = CLIPProcessor.from_pretrained(REPO)
    model = CLIPModel.from_pretrained(REPO).to(device).eval()

    jeux = {("humain", "ref1"): lire_legendes_humaines()}
    for cle in MODELES:
        for prec in PRECISIONS:
            jeux[(cle, prec)] = lire_legendes_modele(cle, prec)

    noms = sorted(jeux[("smolvlm2", "fp16")])  # mêmes 500 images pour tous
    resume, par_image = [], []
    for (cle, prec), legendes in jeux.items():
        notes = clipscores(model, processor, noms, [legendes[n] for n in noms], device)
        resume.append({"modele": cle, "precision": prec, "n_images": len(noms),
                       "clipscore_moyen": round(statistics.mean(notes), 4),
                       "clipscore_median": round(statistics.median(notes), 4)})
        par_image += [{"modele": cle, "precision": prec, "image": n, "clipscore": round(s, 4)}
                      for n, s in zip(noms, notes)]
        print(resume[-1])

    for nom, donnees in [("quality_clip.csv", resume), ("quality_clip_per_image.csv", par_image)]:
        with open(DOSSIER / nom, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(donnees[0]))
            w.writeheader()
            w.writerows(donnees)
    print("Écrit : quality_clip.csv et quality_clip_per_image.csv")
    