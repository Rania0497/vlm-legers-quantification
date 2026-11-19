"""Telecharge un sous-ensemble reproductible de Flickr8k.

Usage (depuis la racine du projet) :
    python data/download_flickr8k.py --n 500 --seed 42
"""
import argparse
import csv
import random
from pathlib import Path

from datasets import Image, load_dataset

# Copie de Flickr8k sur Hugging Face (colonnes : image, image_filename, captions, query)
HF_DATASET = "tsystems/flickr8k"

# Chemins calcules depuis l'emplacement du script : marche sur Windows, Mac et Linux
RACINE = Path(__file__).resolve().parent.parent
DOSSIER_SORTIE = RACINE / "data" / "raw" / "flickr8k_subset"   # ignore par Git
DOSSIER_IMAGES = DOSSIER_SORTIE / "images"
FICHIER_LEGENDES = DOSSIER_SORTIE / "captions.csv"
FICHIER_IDS = RACINE / "data" / "flickr8k_subset_ids.csv"       # suivi par Git


def lire_ids():
    """Relit la liste des images choisies (ligne 1 = en-tete)."""
    lignes = FICHIER_IDS.read_text(encoding="utf-8").splitlines()
    return [ligne.strip() for ligne in lignes[1:] if ligne.strip()]


def ecrire_ids(ids):
    """Sauvegarde la liste des images choisies pour que tout le monde ait les memes."""
    FICHIER_IDS.write_text("image_filename\n" + "\n".join(ids) + "\n", encoding="utf-8")


def main(n, seed):
    DOSSIER_IMAGES.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(HF_DATASET, split="train")
    # decode=False : on garde les octets bruts du JPEG (pas de re-compression)
    ds = ds.cast_column("image", Image(decode=False))
    noms = list(ds["image_filename"])
    index_par_nom = {nom: i for i, nom in enumerate(noms)}

    if FICHIER_IDS.exists():
        # La liste existe deja : on la reutilise, on ne retire pas au hasard une 2e fois
        ids = lire_ids()
        print(f"Liste existante reutilisee : {len(ids)} images.")
    else:
        # Meme graine = meme tirage a chaque fois
        rng = random.Random(seed)
        ids = sorted(rng.sample(noms, n))
        ecrire_ids(ids)
        print(f"Nouvelle liste creee : {len(ids)} images (graine {seed}).")

    manquants = [nom for nom in ids if nom not in index_par_nom]
    if manquants:
        raise SystemExit(
            f"{len(manquants)} images de la liste sont absentes du jeu "
            f"(exemple : {manquants[0]})."
        )

    nb_legendes = 0
    with open(FICHIER_LEGENDES, "w", newline="", encoding="utf-8") as f:
        ecrivain = csv.writer(f)
        ecrivain.writerow(["image_filename", "caption_index", "caption"])
        for nom in ids:
            exemple = ds[index_par_nom[nom]]
            fichier = Path(nom).name          # .name : garde seulement le nom, par securite
            if not Path(fichier).suffix:
                fichier += ".jpg"
            chemin = DOSSIER_IMAGES / fichier
            if not chemin.exists():           # ne retelecharge pas ce qui existe deja
                chemin.write_bytes(exemple["image"]["bytes"])
            for k, legende in enumerate(exemple["captions"], start=1):
                ecrivain.writerow([fichier, k, legende.strip()])
                nb_legendes += 1

    print(f"Termine : {len(ids)} images dans {DOSSIER_IMAGES}")
    print(f"{nb_legendes} legendes de reference dans {FICHIER_LEGENDES}")


if __name__ == "__main__":
    parseur = argparse.ArgumentParser(description="Sous-ensemble Flickr8k")
    parseur.add_argument("--n", type=int, default=500, help="nombre d'images")
    parseur.add_argument("--seed", type=int, default=42, help="graine du hasard")
    args = parseur.parse_args()
    main(args.n, args.seed)