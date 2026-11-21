"""Nettoie les légendes de la LoRA : retire les débuts creux, garde la légende brute."""
import argparse
import csv
import re
from pathlib import Path

DEBUTS = re.compile(
    r"^(?:the\s+(?:image|photo|picture)\s+(?:is\s+(?:of\s+)?|shows\s+|features\s+)"
    r"|this\s+is\s+(?:a\s+|an\s+)?(?:picture|photo|image)\s+of\s+"
    r"|this\s+is\s+)",
    re.IGNORECASE,
)


def nettoyer(legende):
    propre = DEBUTS.sub("", legende.strip()).strip()
    return propre or legende  # si tout est supprimé, on garde l'original


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dossier", default="data/raw/lora_512")
    p.add_argument("--prefixe", default="zellige, ")
    a = p.parse_args()

    chemin = Path(a.dossier) / "metadata.csv"
    with open(chemin, newline="", encoding="utf-8") as f:
        lignes = list(csv.DictReader(f))

    for l in lignes:
        # on repart de la légende brute du VLM : relancer le script donne le même résultat
        l["text"] = a.prefixe + nettoyer(l["legende_vlm"])
        print(f"{l['file_name']} : {l['text']}")

    with open(chemin, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file_name", "text", "legende_vlm"])
        w.writeheader()
        w.writerows(lignes)
    print(f"{len(lignes)} légendes nettoyées dans {chemin}")