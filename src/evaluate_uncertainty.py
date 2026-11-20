"""Intervalles d'incertitude (bootstrap apparié) pour CIDEr et CLIPScore."""
import csv
from collections import defaultdict

import numpy as np

from src.evaluate_text import DOSSIER, MODELES, PRECISIONS

N_BOOT = 10_000
SEED = 42  # graine fixe : mêmes intervalles à chaque lancement
METRIQUES = {
    "cider": ("quality_cider_per_image.csv", "cider"),
    "clipscore": ("quality_clip_per_image.csv", "clipscore"),
}
CHAMPS = ["metrique", "modele", "precision", "moyenne", "ic95_bas", "ic95_haut",
          "diff_vs_fp16", "diff_ic95_bas", "diff_ic95_haut", "ecart_reel"]


def charger(fichier, colonne):
    """(modele, precision) -> {image: note}."""
    donnees = defaultdict(dict)
    with open(DOSSIER / fichier, newline="", encoding="utf-8") as f:
        for l in csv.DictReader(f):
            donnees[(l["modele"], l["precision"])][l["image"]] = float(l[colonne])
    return donnees


def ic95(valeurs):
    bas, haut = np.percentile(valeurs, [2.5, 97.5])
    return round(float(bas), 4), round(float(haut), 4)


if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    lignes = []
    for nom, (fichier, colonne) in METRIQUES.items():
        donnees = charger(fichier, colonne)
        images = sorted(donnees[("smolvlm2", "fp16")])  # mêmes 500 images, même ordre
        n = len(images)
        # UN SEUL tirage pour toutes les configurations : c'est ce qui rend
        # la comparaison appariée (les mêmes images sont piochées pour fp16 et 4 bits).
        idx = rng.integers(0, n, size=(N_BOOT, n))
        notes = {k: np.array([v[i] for i in images]) for k, v in donnees.items()}

        for cle in MODELES:
            base = notes[(cle, "fp16")]
            for prec in PRECISIONS:
                x = notes[(cle, prec)]
                bas, haut = ic95(x[idx].mean(axis=1))
                ligne = dict.fromkeys(CHAMPS, "")
                ligne.update(metrique=nom, modele=cle, precision=prec,
                             moyenne=round(float(x.mean()), 4), ic95_bas=bas, ic95_haut=haut)
                if prec != "fp16":
                    d_bas, d_haut = ic95((x[idx] - base[idx]).mean(axis=1))
                    ligne.update(diff_vs_fp16=round(float((x - base).mean()), 4),
                                 diff_ic95_bas=d_bas, diff_ic95_haut=d_haut,
                                 ecart_reel="oui" if d_bas > 0 or d_haut < 0 else "non")
                lignes.append(ligne)
                print({k: v for k, v in ligne.items() if v != ""})

    with open(DOSSIER / "uncertainty.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CHAMPS)
        w.writeheader()
        w.writerows(lignes)
    print("Écrit : uncertainty.csv")