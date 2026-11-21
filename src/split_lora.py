"""Sépare les images de la LoRA en entraînement et test, avec leurs légendes."""
import argparse
import csv
import random
import shutil
from pathlib import Path

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--images", default="data/raw/lora_512")
    p.add_argument("--captions", default="data/lora_captions.csv")
    p.add_argument("--n_test", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sortie", default="data/raw")
    a = p.parse_args()

    with open(a.captions, newline="", encoding="utf-8") as f:
        lignes = list(csv.DictReader(f))
    noms = sorted(l["file_name"] for l in lignes)
    # graine fixe : le même découpage à chaque lancement
    test = set(random.Random(a.seed).sample(noms, a.n_test))

    for jeu, est_dans_jeu in [("lora_train", lambda n: n not in test),
                              ("lora_test", lambda n: n in test)]:
        dossier = Path(a.sortie) / jeu
        dossier.mkdir(parents=True, exist_ok=True)
        retenues = [l for l in lignes if est_dans_jeu(l["file_name"])]
        for l in retenues:
            shutil.copy(Path(a.images) / l["file_name"], dossier / l["file_name"])
        # metadata.csv : le format lu directement par le script d'entraînement
        with open(dossier / "metadata.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["file_name", "text", "legende_vlm"])
            w.writeheader()
            w.writerows(retenues)
        print(f"{jeu} : {len(retenues)} images")