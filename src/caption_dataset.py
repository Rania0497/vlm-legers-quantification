"""Légende les images de la LoRA avec un VLM et écrit metadata.csv."""
import argparse
import csv
from pathlib import Path

from PIL import Image

from src.captioning import charger, legender, vider_gpu

EXTS = {".jpg", ".jpeg", ".png"}

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--images", default="data/raw/lora_512")
    p.add_argument("--model", default="qwen25vl")
    p.add_argument("--precision", default="4bit")
    p.add_argument("--prefixe", default="zellige, ")  # mot repère ajouté devant chaque légende
    a = p.parse_args()

    dossier = Path(a.images)
    fichiers = sorted(f for f in dossier.iterdir() if f.suffix.lower() in EXTS)
    model, processor = charger(a.model, a.precision)

    lignes = []
    for f in fichiers:
        legende = legender(model, processor, Image.open(f).convert("RGB")).rstrip(".")
        lignes.append({"file_name": f.name, "text": a.prefixe + legende, "legende_vlm": legende})
        print(f"{f.name} : {lignes[-1]['text']}")

    # metadata.csv : format lu directement par les scripts d'entraînement LoRA
    with open(dossier / "metadata.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["file_name", "text", "legende_vlm"])
        w.writeheader()
        w.writerows(lignes)

    del model, processor
    vider_gpu()
    print(f"{len(lignes)} légendes écrites dans {dossier / 'metadata.csv'}")