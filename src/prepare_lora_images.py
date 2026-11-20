"""Prépare les images pour la LoRA : recadrage carré centré, puis 512x512."""
import argparse
from pathlib import Path

from PIL import Image, ImageOps

TAILLE = 512
EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def preparer(chemin, destination):
    img = Image.open(chemin)
    img = ImageOps.exif_transpose(img)  # remet à l'endroit les photos de téléphone
    img = img.convert("RGB")
    if min(img.size) < TAILLE:
        print(f"[attention] {chemin.name} est petite {img.size}, elle sera floue")
    img = ImageOps.fit(img, (TAILLE, TAILLE), Image.LANCZOS)  # recadre au centre + redimensionne
    img.save(destination, quality=95)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--entree", default="data/raw/lora_images")
    p.add_argument("--sortie", default="data/raw/lora_512")
    a = p.parse_args()

    sortie = Path(a.sortie)
    sortie.mkdir(parents=True, exist_ok=True)
    fichiers = sorted(f for f in Path(a.entree).iterdir() if f.suffix.lower() in EXTS)
    for i, f in enumerate(fichiers, 1):
        preparer(f, sortie / f"img_{i:03d}.jpg")  # noms neutres
    print(f"{len(fichiers)} images préparées dans {sortie}")
    if not 20 <= len(fichiers) <= 50:
        print("[attention] le guide conseille 20 à 50 images")