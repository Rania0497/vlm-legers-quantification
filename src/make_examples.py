"""Exemples visuels : image + légendes humaines + légendes des 6 configurations."""
import csv
import random
import textwrap
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # pas de fenêtre : on sauvegarde seulement
import matplotlib.pyplot as plt
from PIL import Image

from src.evaluate_text import DOSSIER, MODELES, PRECISIONS, REFS

IMAGES = Path("data/raw/flickr8k_subset/images")
SORTIE = DOSSIER.parent / "examples"
SEED = 42
N_ALEATOIRES = 6
N_DIFFICILES = 3


def lire_legendes_modele(cle, precision):
    """image -> légende brute du modèle."""
    chemin = DOSSIER / f"captions_{cle}_{precision}.csv"
    with open(chemin, newline="", encoding="utf-8") as f:
        return {l["image"]: l["legende"] for l in csv.DictReader(f)}


def dessiner(nom, groupe, humains, legendes, chemin):
    fig, (ax_img, ax_txt) = plt.subplots(
        1, 2, figsize=(12, 6), gridspec_kw={"width_ratios": [1, 1.5]})
    ax_img.imshow(Image.open(IMAGES / nom).convert("RGB"))
    ax_img.axis("off")
    ax_img.set_title(f"{nom} ({groupe})", fontsize=9)

    lignes = ["HUMAINS"]
    lignes += [textwrap.fill("- " + h, 70, subsequent_indent="  ") for h in humains]
    lignes += ["", "MODELES"]
    for (cle, prec), leg in legendes.items():
        lignes.append(textwrap.fill(f"{cle} {prec} : {leg[nom]}", 70, subsequent_indent="  "))
    ax_txt.axis("off")
    ax_txt.text(0, 1, "\n".join(lignes), va="top", fontsize=8, family="monospace")
    fig.tight_layout()
    fig.savefig(chemin, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    humains = defaultdict(list)
    with open(REFS, newline="", encoding="utf-8") as f:
        for l in csv.DictReader(f):
            humains[l["image_filename"]].append(l["caption"].strip())

    legendes = {(c, p): lire_legendes_modele(c, p) for c in MODELES for p in PRECISIONS}
    noms = sorted(legendes[("smolvlm2", "fp16")])

    # CLIPScore moyen de chaque image sur les 6 configurations (sans la référence humaine)
    notes = defaultdict(list)
    with open(DOSSIER / "quality_clip_per_image.csv", newline="", encoding="utf-8") as f:
        for l in csv.DictReader(f):
            if l["modele"] != "humain":
                notes[l["image"]].append(float(l["clipscore"]))
    moyennes = {n: sum(v) / len(v) for n, v in notes.items()}

    aleatoires = random.Random(SEED).sample(noms, N_ALEATOIRES)
    difficiles = [n for n in sorted(moyennes, key=moyennes.get)
                  if n not in aleatoires][:N_DIFFICILES]
    choix = [(n, "aléatoire") for n in aleatoires] + [(n, "difficile pour CLIP") for n in difficiles]

    SORTIE.mkdir(parents=True, exist_ok=True)
    md = ["# Exemples (légendes seulement, images non incluses)", ""]
    for i, (nom, groupe) in enumerate(choix, 1):
        dessiner(nom, groupe, humains[nom], legendes, SORTIE / f"{i:02d}_{Path(nom).stem}.png")
        md += [f"## {i}. {nom} ({groupe}, CLIPScore moyen {moyennes[nom]:.3f})", "",
               "Humains :"] + [f"- {h}" for h in humains[nom]] + ["", "Modèles :"]
        md += [f"- {c} {p} : {legendes[(c, p)][nom]}" for c in MODELES for p in PRECISIONS]
        md.append("")
        print(f"{i:02d} {nom} ({groupe})")
    md.append("Note : 'difficile pour CLIP' = CLIPScore moyen le plus bas sur les 6 configurations. "
              "Ça ne prouve pas que les modèles se trompent, CLIP peut aussi se tromper.")
    (SORTIE / "examples.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Écrit : dossier {SORTIE}")