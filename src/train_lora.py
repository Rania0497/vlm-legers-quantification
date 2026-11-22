"""Entraîne la LoRA (rang 4 par défaut) avec le script officiel de diffusers.

Étapes : 1) télécharge le script officiel (version figée, empreinte vérifiée),
2) le lance avec accelerate en mesurant la durée et la VRAM (nvidia-smi).
"""
import argparse
import hashlib
import importlib.metadata as meta
import json
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

DIFFUSERS_TAG = "v0.42.1"
URL = (f"https://raw.githubusercontent.com/huggingface/diffusers/{DIFFUSERS_TAG}"
       "/examples/text_to_image/train_text_to_image_lora.py")
SHA256 = "7d67f3ed756a3cb79183153349ebb2d141c999f898da964edb2ea4af5d72d6ab"
SCRIPT = Path("training/train_text_to_image_lora.py")


def version(paquet):
    try:
        return meta.version(paquet)
    except meta.PackageNotFoundError:
        return None


def preparer_script():
    """Télécharge le script officiel (non modifié) et vérifie son empreinte SHA-256."""
    SCRIPT.parent.mkdir(parents=True, exist_ok=True)
    if not SCRIPT.exists():
        urllib.request.urlretrieve(URL, SCRIPT)
    if hashlib.sha256(SCRIPT.read_bytes()).hexdigest() != SHA256:
        raise SystemExit(f"{SCRIPT} ne correspond pas à la version attendue ({DIFFUSERS_TAG}). "
                         "Supprime ce fichier et relance.")
    print(f"Script officiel diffusers {DIFFUSERS_TAG} prêt (empreinte vérifiée)")


class MoniteurVRAM(threading.Thread):
    """Relève la mémoire GPU utilisée (nvidia-smi) toutes les 5 s et garde le maximum."""

    def __init__(self, periode=5):
        super().__init__(daemon=True)
        self.periode, self.max_mio = periode, 0
        self._arret = threading.Event()

    def run(self):
        while not self._arret.is_set():
            try:
                sortie = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=10).stdout.split()
                if sortie:
                    self.max_mio = max(self.max_mio, int(sortie[0]))
            except (OSError, ValueError, subprocess.SubprocessError):
                pass
            self._arret.wait(self.periode)

    def arreter(self):
        self._arret.set()


def commande(a):
    # --mixed_precision est donné deux fois : à accelerate ET au script d'entraînement.
    # Le script ne convertit les poids LoRA en float32 que si on le lui passe lui-même
    # (il lit args.mixed_precision) ; sinon, erreur "Attempting to unscale FP16 gradients".
    return ["accelerate", "launch", "--mixed_precision=fp16", str(SCRIPT),
            "--mixed_precision=fp16",
            f"--pretrained_model_name_or_path={a.modele}", f"--train_data_dir={a.train_dir}",
            "--caption_column=text", "--resolution=512", "--random_flip",
            "--train_batch_size=1", "--gradient_accumulation_steps=4", "--gradient_checkpointing",
            f"--max_train_steps={a.etapes}", f"--learning_rate={a.lr}",
            "--lr_scheduler=cosine", "--lr_warmup_steps=0",
            f"--rank={a.rang}", f"--seed={a.seed}", f"--output_dir={a.out}"]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--modele", default="sd-legacy/stable-diffusion-v1-5")
    p.add_argument("--train_dir", default="data/raw/lora_train")
    p.add_argument("--out", default="lora_out")
    p.add_argument("--etapes", type=int, default=250)
    p.add_argument("--rang", type=int, default=4)
    p.add_argument("--lr", default="1e-04")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true", help="prépare le script et affiche la commande")
    a = p.parse_args()

    preparer_script()
    cmd = commande(a)
    if a.dry_run:
        print(" ".join(cmd))
        raise SystemExit(0)
    if shutil.which("accelerate") is None:
        raise SystemExit("accelerate est introuvable : pip install -r requirements.txt -r requirements-lora.txt")
    if version("torchao"):
        print(f"[attention] torchao {version('torchao')} est installé ; s'il est trop ancien, peft "
              "refuse de démarrer. Dans ce cas : pip uninstall -y torchao")

    moniteur = MoniteurVRAM()
    moniteur.start()
    debut = time.time()
    code = subprocess.run(cmd).returncode
    duree = round(time.time() - debut)
    moniteur.arreter()

    resume = {"code_sortie": code, "duree_totale_s": duree,
              "vram_max_nvidia_smi_mio": moniteur.max_mio,
              "etapes": a.etapes, "rang": a.rang, "seed": a.seed, "modele_base": a.modele,
              "versions": {k: version(k) for k in ["torch", "transformers", "diffusers", "peft", "accelerate"]}}
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "train_run.json").write_text(json.dumps(resume, indent=2), encoding="utf-8")
    print(json.dumps(resume, indent=2))
    sys.exit(code)