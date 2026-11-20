"""Assemble le tableau final (CSV + Markdown) à partir des résultats déjà calculés."""
import csv

from src.evaluate_text import DOSSIER, MODELES, PRECISIONS


def lire(nom):
    """Fichier CSV -> dict {(modele, precision): ligne}."""
    with open(DOSSIER / nom, newline="", encoding="utf-8") as f:
        return {(l["modele"], l["precision"]): l for l in csv.DictReader(f)}


def lire_ic(metrique):
    """Intervalles d'incertitude d'une métrique -> {(modele, precision): ligne}."""
    with open(DOSSIER / "uncertainty.csv", newline="", encoding="utf-8") as f:
        return {(l["modele"], l["precision"]): l
                for l in csv.DictReader(f) if l["metrique"] == metrique}


def avec_ic(l):
    """Formate 'moyenne [bas-haut]'."""
    return f"{float(l['moyenne']):.3f} [{float(l['ic95_bas']):.3f}-{float(l['ic95_haut']):.3f}]"


if __name__ == "__main__":
    perf, texte = lire("summary.csv"), lire("quality_text.csv")
    clip, coh = lire("quality_clip.csv"), lire("consistency.csv")
    cider, cs = lire_ic("cider"), lire_ic("clipscore")

    lignes = []
    for cle in MODELES:
        for prec in PRECISIONS:
            k = (cle, prec)
            lignes.append({
                "Modèle": cle,
                "Précision": prec,
                "VRAM pic (Go)": perf[k]["vram_pic_Go"],
                "Temps/image (s)": perf[k]["temps_moyen_s"],
                "BLEU-4": f"{float(texte[k]['bleu4']):.3f}",
                "CIDEr [IC95]": avec_ic(cider[k]),
                "CLIPScore [IC95]": avec_ic(cs[k]),
                "Mots/légende": coh[k]["mots_moyens"],
                "Identiques à fp16 (%)": coh[k]["identiques_fp16_pct"],
            })

    colonnes = list(lignes[0])
    with open(DOSSIER / "final_table.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=colonnes)
        w.writeheader()
        w.writerows(lignes)

    humain = float(clip[("humain", "ref1")]["clipscore_moyen"])
    md = ["| " + " | ".join(colonnes) + " |",
          "|" + "|".join(["---"] * len(colonnes)) + "|"]
    md += ["| " + " | ".join(str(l[c]) for c in colonnes) + " |" for l in lignes]
    md += ["",
           "Notes : 500 images Flickr8k (graine 42), GPU Tesla T4, une seule exécution.",
           "IC95 = intervalle de confiance bootstrap (10 000 tirages, graine 42).",
           "BLEU-4 est calculé sur tout le corpus, donc sans intervalle.",
            "Seul le décodeur de texte est quantifié : l'encodeur d'images, le connecteur et lm_head restent en fp16.",
           f"CLIPScore d'une légende humaine (référence) : {humain:.3f}."]
    (DOSSIER / "final_table.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print("\nÉcrit : final_table.csv et final_table.md")