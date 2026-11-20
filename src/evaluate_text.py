"""Qualité des légendes : BLEU-4 et CIDEr contre les 5 références humaines."""
import csv
import re
from collections import defaultdict
from pathlib import Path

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider

DOSSIER = Path("results/n500")
REFS = Path("data/raw/flickr8k_subset/captions.csv")
MODELES = ["smolvlm2", "qwen25vl"]
PRECISIONS = ["fp16", "8bit", "4bit"]


def normaliser(texte):
    """Minuscules, sans ponctuation, espaces simples."""
    texte = re.sub(r"[^a-z0-9 ]", " ", texte.lower())
    return " ".join(texte.split())


def charger_references():
    """image -> liste des 5 légendes humaines normalisées."""
    refs = defaultdict(list)
    with open(REFS, newline="", encoding="utf-8") as f:
        for ligne in csv.DictReader(f):
            refs[ligne["image_filename"]].append(normaliser(ligne["caption"]))
    return dict(refs)


def charger_legendes(cle, precision):
    """image -> [légende du modèle normalisée] (une liste d'un élément)."""
    chemin = DOSSIER / f"captions_{cle}_{precision}.csv"
    with open(chemin, newline="", encoding="utf-8") as f:
        return {l["image"]: [normaliser(l["legende"])] for l in csv.DictReader(f)}


def evaluer(refs, hyps):
    """Renvoie (bleu4, cider, dict cider par image)."""
    assert set(hyps) <= set(refs), "des images n'ont pas de références"
    images = sorted(hyps)  # même ordre pour toutes les configurations
    gts = {i: refs[i] for i in images}
    res = {i: hyps[i] for i in images}
    bleu, _ = Bleu(4).compute_score(gts, res, verbose=0)
    cider, cider_images = Cider().compute_score(gts, res)
    return bleu[3], cider, dict(zip(images, cider_images))


if __name__ == "__main__":
    refs = charger_references()
    print(f"{len(refs)} images de référence")
    lignes, par_image = [], []
    for cle in MODELES:
        for prec in PRECISIONS:
            hyps = charger_legendes(cle, prec)
            bleu4, cider, cider_img = evaluer(refs, hyps)
            lignes.append({"modele": cle, "precision": prec, "n_images": len(hyps),
                           "bleu4": round(bleu4, 4), "cider": round(cider, 4)})
            par_image += [{"modele": cle, "precision": prec, "image": i, "cider": round(c, 4)}
                          for i, c in cider_img.items()]
            print(lignes[-1])

    for nom, donnees in [("quality_text.csv", lignes), ("quality_cider_per_image.csv", par_image)]:
        with open(DOSSIER / nom, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(donnees[0]))
            w.writeheader()
            w.writerows(donnees)
    print("Écrit : quality_text.csv et quality_cider_per_image.csv")