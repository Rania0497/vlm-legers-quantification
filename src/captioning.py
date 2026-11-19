"""Légendage d'images avec un VLM, en fp16, 8 bits ou 4 bits."""
import argparse
import gc

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

MODELES = {
    "smolvlm2": "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
    "qwen25vl": "Qwen/Qwen2.5-VL-3B-Instruct",
}
PROMPT = "Describe this image in one short sentence."


def charger(cle_modele, precision):
    """Charge un modèle et son processor. precision: fp16, 8bit ou 4bit."""
    repo = MODELES[cle_modele]
    args = {"device_map": "auto", "dtype": torch.float16}
    if precision == "8bit":
        args["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    elif precision == "4bit":
        args["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
    elif precision != "fp16":
        raise ValueError(f"précision inconnue : {precision}")
    processor = AutoProcessor.from_pretrained(repo)
    model = AutoModelForImageTextToText.from_pretrained(repo, **args)
    model.eval()
    return model, processor


def legender(model, processor, image, prompt=PROMPT, max_new_tokens=60):
    """Renvoie une légende (str) pour une image PIL."""
    messages = [{"role": "user",
                 "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    texte = processor.apply_chat_template(messages, add_generation_prompt=True)
    entrees = processor(text=texte, images=[image], return_tensors="pt")
    entrees = entrees.to(model.device, dtype=model.dtype)
    with torch.no_grad():
        sortie = model.generate(**entrees, max_new_tokens=max_new_tokens, do_sample=False)
    nouveaux = sortie[:, entrees["input_ids"].shape[1]:]  # on enlève la consigne
    return processor.batch_decode(nouveaux, skip_special_tokens=True)[0].strip()


# MODIFIÉ : l'ancienne liberer(model) est supprimée. `del model` dans une
# fonction n'efface que le nom local, donc la mémoire GPU n'était pas rendue.
# REMPLACÉE par vider_gpu(), à appeler APRÈS un `del model, processor`
# fait par l'appelant lui-même.
def vider_gpu():
    """Rend la VRAM. À appeler après `del model, processor`."""
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=MODELES, required=True)
    p.add_argument("--precision", choices=["fp16", "8bit", "4bit"], default="fp16")
    p.add_argument("--image", required=True)
    a = p.parse_args()
    m, proc = charger(a.model, a.precision)
    img = Image.open(a.image).convert("RGB")
    print(legender(m, proc, img))
    if torch.cuda.is_available():
        print(f"VRAM max : {torch.cuda.max_memory_allocated() / 1e9:.2f} Go")