"""Lance les 6 configurations (2 modèles x 3 précisions) sur les mêmes images."""
import argparse
import csv
import statistics
import time
from pathlib import Path

import torch
from PIL import Image

from src.captioning import MODELES, charger, legender, vider_gpu

PRECISIONS = ["fp16", "8bit", "4bit"]


def lister_images(dossier, n):
    """Les n premières images, triées par nom (même sélection à chaque run)."""
    exts = {".jpg", ".jpeg", ".png"}
    return sorted(p for p in Path(dossier).iterdir() if p.suffix.lower() in exts)[:n]


def tester(cle, precision, images, sortie):
    """Une configuration : renvoie un dict de mesures et écrit les légendes."""
    vider_gpu()
    torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    model, processor = charger(cle, precision)
    torch.cuda.synchronize()
    chargement = time.perf_counter() - t0
    vram_poids = torch.cuda.memory_allocated() / 1e9

    # échauffement : non chronométré
    legender(model, processor, Image.open(images[0]).convert("RGB"))

    lignes, durees = [], []
    for chemin in images:
        img = Image.open(chemin).convert("RGB")
        torch.cuda.synchronize()
        t = time.perf_counter()
        legende = legender(model, processor, img)
        torch.cuda.synchronize()
        d = time.perf_counter() - t
        durees.append(d)
        lignes.append({"image": chemin.name, "legende": legende, "secondes": round(d, 3)})

    vram_pic = torch.cuda.max_memory_allocated() / 1e9

    with open(sortie / f"captions_{cle}_{precision}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image", "legende", "secondes"])
        w.writeheader()
        w.writerows(lignes)

    del model, processor  # on supprime NOUS-MÊMES les références
    vider_gpu()
    return {
        "modele": cle, "precision": precision, "statut": "ok", "n_images": len(images),
        "chargement_s": round(chargement, 1),
        "vram_poids_Go": round(vram_poids, 2), "vram_pic_Go": round(vram_pic, 2),
        "temps_moyen_s": round(statistics.mean(durees), 3),
        "temps_median_s": round(statistics.median(durees), 3),
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--images-dir", required=True)
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--out", default="results")
    a = p.parse_args()

    sortie = Path(a.out)
    sortie.mkdir(exist_ok=True)
    images = lister_images(a.images_dir, a.n)
    print(f"{len(images)} images, GPU : {torch.cuda.get_device_name(0)}")

    resume = []
    for cle in MODELES:
        for prec in PRECISIONS:
            if (sortie / f"captions_{cle}_{prec}.csv").exists():
                print(f"[saut] {cle} {prec} déjà fait")
                continue
            print(f"[test] {cle} {prec}")
            try:
                resume.append(tester(cle, prec, images, sortie))
            # MODIFIÉ : on attrape RuntimeError (et pas seulement OOM), car les
            # erreurs de bitsandbytes en sont. Un échec est noté, sans tout arrêter.
            except RuntimeError as e:
                statut = "OOM" if isinstance(e, torch.cuda.OutOfMemoryError) else "erreur"
                resume.append({"modele": cle, "precision": prec,
                               "statut": statut, "detail": str(e)[:80]})
            vider_gpu()  # MODIFIÉ : après le except, une fois l'erreur oubliée
            print(resume[-1])

    if resume:
        champs = list(dict.fromkeys(k for r in resume for k in r))
        with open(sortie / f"summary_n{len(images)}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=champs)
            w.writeheader()
            w.writerows(resume)