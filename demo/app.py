"""Démo Gradio : légender une image avec un petit VLM (fp16, 8 bits ou 4 bits)."""
import argparse
import inspect  # AJOUTÉ : pour savoir où Gradio accepte css et theme
import time

import gradio as gr
import torch

from src.captioning import MODELES, charger, legender, vider_gpu

TAILLE_MAX = 512  # comme les images de l'évaluation (Flickr8k, environ 500 px)
VERT = "#00843D"        # AJOUTÉ : vert principal (approximation du vert EMSI, à ajuster)
VERT_FONCE = "#005C2B"  # AJOUTÉ : vert foncé pour le survol et les titres

# MODIFIÉ : le CSS n'est plus mis dans gr.HTML (Gradio retire les balises style par sécurité).
# Il est donné à Gradio via le paramètre css (voir _options_style plus bas).
CSS = f"""
html, body {{ overflow: hidden !important; height: 100vh; }}
.gradio-container {{ max-width: 1150px !important; margin: auto; padding-top: 8px !important;
                     height: 100vh; overflow: hidden !important; }}
#bandeau {{ background: linear-gradient(90deg, {VERT_FONCE}, {VERT});
           color: white; padding: 10px 20px; border-radius: 12px; }}
#bandeau h1 {{ color: white !important; margin: 0; font-size: 22px; }}
#bandeau p {{ color: #E6F4EC !important; margin: 2px 0 0 0; font-size: 13px; }}
#bouton {{ background: {VERT} !important; color: white !important; border: none !important;
          font-weight: 600; }}
#bouton:hover {{ background: {VERT_FONCE} !important; }}
.carte {{ border: 2px solid {VERT} !important; border-radius: 12px !important; padding: 10px; }}
#pied {{ text-align: center; color: {VERT_FONCE}; font-size: 12px; }}
footer {{ display: none !important; }}
"""

# AJOUTÉ : thème vert, pour que les boutons radio et les focus soient verts aussi
THEME = gr.themes.Soft(primary_hue=gr.themes.colors.green)

# AJOUTÉ : bandeau vert en haut de la page (court, pour tenir sans défiler)
BANDEAU = """
<div id="bandeau">
  <h1>EMSI | Légendes d'images avec de petits VLM</h1>
  <p>Projet 7 : SmolVLM2 et Qwen2.5-VL en fp16, 8 bits et 4 bits</p>
</div>
"""


def _options_style(fonction):
    """AJOUTÉ : donne css et theme là où la version de Gradio les accepte.
    Gradio 4 et 5 : dans gr.Blocks(...). Gradio 6 : dans launch(...)."""
    params = inspect.signature(fonction).parameters
    return {k: v for k, v in {"css": CSS, "theme": THEME}.items() if k in params}


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
    # MODIFIÉ : gr.Blocks au lieu de gr.Interface ; le style passe par gr.HTML,
    # ce qui évite les différences entre versions de Gradio (theme/css)
    with gr.Blocks(title="EMSI - VLM légers", **_options_style(gr.Blocks)) as demo:
        gr.HTML(BANDEAU)  # AJOUTÉ : bandeau EMSI
        with gr.Row():
            with gr.Column(scale=1, elem_classes="carte"):  # AJOUTÉ : cadre vert
                gr.Markdown("**1. Image et réglages**")  # MODIFIÉ : titre plus court
                image = gr.Image(type="pil", label="Image", height=250)  # MODIFIÉ : plus petite, pas de défilement
                with gr.Row():  # AJOUTÉ : modèle et précision côte à côte (gain de hauteur)
                    cle = gr.Radio(list(MODELES), value="qwen25vl", label="Modèle")
                    precision = gr.Radio(["fp16", "8bit", "4bit"], value="4bit",
                                         label="Précision")
                bouton = gr.Button("Décrire l'image", elem_id="bouton")  # AJOUTÉ : id pour le CSS
            with gr.Column(scale=1, elem_classes="carte"):  # AJOUTÉ : cadre vert
                gr.Markdown("**2. Résultat**")  # MODIFIÉ : titre plus court
                legende = gr.Textbox(label="Légende", lines=5)
                mesures = gr.Textbox(label="Mesures (temps et VRAM)")
                gr.Markdown("4 bits = moins de mémoire, qualité à vérifier. "
                            "Premier appel lent (chargement du modèle).")
        # AJOUTÉ : pied de page
        gr.HTML('<div id="pied">EMSI | Projet 7 | Images réduites à 512 px</div>')
        # AJOUTÉ : le bouton lance decrire
        bouton.click(decrire, inputs=[image, cle, precision],
                     outputs=[legende, mesures])
    return demo


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--share", action="store_true",
                   help="lien public temporaire (utile sur Colab)")
    a = p.parse_args()
    demo = construire()
    demo.launch(share=a.share, **_options_style(demo.launch))  # MODIFIÉ : css/theme pour Gradio 6
