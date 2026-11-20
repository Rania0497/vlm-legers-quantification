"""Longueur des légendes et accord de chaque version avec fp16."""
import csv
import statistics

from src.evaluate_text import (DOSSIER, MODELES, PRECISIONS,
                               charger_legendes, charger_references)


def nb_mots(texte):
    return len(texte.split())


def jaccard(a, b):
    """Part de mots communs entre deux phrases (0 = aucun, 1 = mêmes mots)."""
    ma, mb = set(a.split()), set(b.split())
    return len(ma & mb) / len(ma | mb) if (ma | mb) else 1.0


if __name__ == "__main__":
    refs = charger_references()
    humains = statistics.mean(nb_mots(c) for liste in refs.values() for c in liste)
    print(f"Longueur moyenne des légendes humaines : {humains:.1f} mots")

    lignes = []
    for cle in MODELES:
        base = charger_legendes(cle, "fp16")  # image -> [légende fp16]
        for prec in PRECISIONS:
            hyps = charger_legendes(cle, prec)
            images = sorted(hyps)
            lignes.append({
                "modele": cle,
                "precision": prec,
                "mots_moyens": round(statistics.mean(nb_mots(hyps[i][0]) for i in images), 2),
                "identiques_fp16_pct": round(
                    100 * sum(hyps[i][0] == base[i][0] for i in images) / len(images), 1),
                "mots_communs_fp16": round(
                    statistics.mean(jaccard(hyps[i][0], base[i][0]) for i in images), 3),
            })
            print(lignes[-1])

    with open(DOSSIER / "consistency.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(lignes[0]))
        w.writeheader()
        w.writerows(lignes)
    print("Écrit : consistency.csv")