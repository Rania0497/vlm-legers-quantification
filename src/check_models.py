"""Vérifie licence et taille des 2 VLMs candidats, sans les télécharger."""
from huggingface_hub import HfApi

MODELES = {
    "SmolVLM2-2.2B": "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
    "Qwen2.5-VL-3B": "Qwen/Qwen2.5-VL-3B-Instruct",
}
OCTETS = {"fp16": 2, "8bit": 1, "4bit": 0.5}  # octets par paramètre

api = HfApi()
for nom, repo in MODELES.items():
    info = api.model_info(repo, expand=["safetensors", "cardData"])
    carte = info.card_data.to_dict() if info.card_data else {}
    licence = carte.get("license_name") or carte.get("license") or "inconnue"
    n = info.safetensors.total if info.safetensors else None

    print(f"\n{nom} ({repo})")
    print(f"  licence : {licence}")
    if n is None:
        print("  paramètres : non disponibles")
        continue
    print(f"  paramètres : {n / 1e9:.2f} milliards")
    for precision, octets in OCTETS.items():
        print(f"  poids en {precision} : {n * octets / 1e9:.1f} Go")