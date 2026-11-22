"""Démo Gradio : légender une image avec un petit VLM (fp16, 8 bits ou 4 bits)."""
import argparse
import time

import gradio as gr
import torch

from src.captioning import MODELES, charger, legender, vider_gpu

TAILLE_MAX = 512  # comme les images de l'évaluation (Flickr8k, environ 500 px)

# Un seul modèle gardé en mémoire à la fois (limite de 12 Go de VRAM)
_etat = {"cle": None, "modele": None, "processor": None}


def _charger_si_besoin(cle, precision):
    """Remplace le modèle en mémoire seulement si le choix a changé."""
    if _etat["cle"] == (cle, precision):
        return
    if _etat["modele"] is not None:
        _etat.update(cle=None, modele=None, processor=None)  # on lâche l'ancien
        vider_gpu()
    modele, processor = charger(cle, precision)
    _etat.update(cle=(cle, precision), modele=modele, processor=processor)


def decrire(image, cle, precision):
    """Inchangée : renvoie (légende, mesures)."""
    if image is None:
        return "Charge d'abord une image.", ""
    if not torch.cuda.is_available():
        return "Aucun GPU NVIDIA détecté : lance la démo sur Colab (GPU T4).", ""
    try:
        image = image.convert("RGB")
        image.thumbnail((TAILLE_MAX, TAILLE_MAX))  # réduit sans déformer
        _charger_si_besoin(cle, precision)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        legende = legender(_etat["modele"], _etat["processor"], image)
        torch.cuda.synchronize()
        duree = time.perf_counter() - t0
        vram = torch.cuda.max_memory_allocated() / 1e9
        return legende, f"{cle} {precision} | {duree:.2f} s | VRAM max (PyTorch) {vram:.2f} Go"
    except RuntimeError as e:  # inclut les erreurs de mémoire et de bitsandbytes
        return f"Erreur : {str(e)[:200]}", ""


def construire():
    # MODIFIÉ : gr.Blocks au lieu de gr.Interface, pour organiser la page soi-même
    # MODIFIÉ : thème doux avec une couleur d'accent
    with gr.Blocks(theme=gr.themes.Soft(primary_hue="emerald"),
                   title="VLM légers") as demo:
        # AJOUTÉ : titre et sous-titre
        gr.Markdown("# Légendes d'images avec de petits modèles vision-langage\n"
                    "Compare **SmolVLM2** et **Qwen2.5-VL** en **fp16, 8 bits ou 4 bits**. "
                    "Le premier appel charge le modèle (plusieurs dizaines de secondes).")
        # AJOUTÉ : deux colonnes, les réglages à gauche et le résultat à droite
        with gr.Row():
            with gr.Column(scale=1):
                image = gr.Image(type="pil", label="Image", height=320)
                # MODIFIÉ : Radio au lieu de Dropdown, tous les choix sont visibles
                cle = gr.Radio(list(MODELES), value="qwen25vl", label="Modèle")
                precision = gr.Radio(["fp16", "8bit", "4bit"], value="4bit",
                                     label="Précision")
                # AJOUTÉ : un vrai bouton à la place du Submit automatique
                bouton = gr.Button("Décrire l'image", variant="primary")
            with gr.Column(scale=1):
                legende = gr.Textbox(label="Légende", lines=4)
                mesures = gr.Textbox(label="Mesures (temps et VRAM)")
                # AJOUTÉ : petit rappel pour le jury
                gr.Markdown("**Rappel** : 4 bits = moins de mémoire, qualité à vérifier. "
                            "Images réduites à 512 px.")
        # AJOUTÉ : relie le bouton à la fonction decrire
        bouton.click(decrire, inputs=[image, cle, precision],
                     outputs=[legende, mesures])
    return demo


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--share", action="store_true",
                   help="lien public temporaire (utile sur Colab)")
    a = p.parse_args()
    construire().launch(share=a.share)