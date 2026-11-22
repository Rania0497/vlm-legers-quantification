"""Affiche les versions des bibliothèques utilisées, pour figer les requirements."""
import platform
import sys
from importlib.metadata import PackageNotFoundError, version

PAQUETS = ["torch", "transformers", "accelerate", "bitsandbytes", "pillow",
           "huggingface_hub", "numpy", "pycocoevalcap", "matplotlib", "gradio",
           "diffusers", "peft", "datasets", "ftfy", "tensorboard", "Jinja2", "num2words"]

if __name__ == "__main__":
    print(f"# Python {sys.version.split()[0]} sur {platform.system()}")
    try:
        import torch
        print(f"# GPU CUDA disponible : {torch.cuda.is_available()} | CUDA {torch.version.cuda}")
    except ImportError:
        pass
    for p in PAQUETS:
        try:
            print(f"{p}=={version(p)}")
        except PackageNotFoundError:
            print(f"# {p} : non installé")