"""Prépare les notebooks Colab avant de les publier sur GitHub.

Retire : les images affichées (droits des photos), l'état des widgets (fait planter
l'affichage sur GitHub) et remplace les liens publics Gradio (temporaires) par XXXX.
Les textes et les chiffres affichés sont conservés : ce sont les preuves.
"""
import argparse
import json
import re
from pathlib import Path

LIEN_GRADIO = re.compile(r"https://[0-9a-zA-Z-]+\.gradio\.live")


def est_a_retirer(sortie):
    """Sortie contenant une image ou un widget."""
    return any(k.startswith("image/") or "widget" in k for k in sortie.get("data", {}))


def nettoyer(chemin):
    nb = json.loads(chemin.read_text(encoding="utf-8"))
    nb.get("metadata", {}).pop("widgets", None)
    retirees = 0
    for cellule in nb.get("cells", []):
        if cellule.get("cell_type") != "code":
            continue
        gardees = [s for s in cellule.get("outputs", []) if not est_a_retirer(s)]
        retirees += len(cellule.get("outputs", [])) - len(gardees)
        cellule["outputs"] = gardees
    texte = json.dumps(nb, ensure_ascii=False, indent=1)
    texte, n_liens = LIEN_GRADIO.subn("https://XXXX.gradio.live", texte)
    chemin.write_text(texte + "\n", encoding="utf-8")
    return retirees, n_liens


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("notebooks", nargs="*", help="fichiers .ipynb (par défaut : notebooks/*.ipynb)")
    a = p.parse_args()
    fichiers = [Path(f) for f in a.notebooks] or sorted(Path("notebooks").glob("*.ipynb"))
    if not fichiers:
        raise SystemExit("Aucun notebook trouvé dans notebooks/")
    for f in fichiers:
        n_img, n_liens = nettoyer(f)
        print(f"{f.name} : {n_img} sortie(s) image/widget retirée(s), {n_liens} lien(s) Gradio masqué(s), "
              f"{f.stat().st_size / 1e6:.2f} Mo")