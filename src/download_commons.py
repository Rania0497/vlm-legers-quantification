"""Télécharge des images libres depuis Wikimedia Commons et note auteur + licence."""
import argparse
import csv
import html
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://commons.wikimedia.org/w/api.php"
# Wikimedia demande un User-Agent qui permet de contacter l'auteur du script.
USER_AGENT = ("vlm-legers-quantification/1.0 "
              "(https://github.com/Rania0497/vlm-legers-quantification; projet etudiant)")
CHAMPS = ["fichier", "titre", "auteur", "url_page", "licence", "url_licence"]
MIN_COTE = 512  # on ignore les images plus petites que 512 px


def appeler_api(params):
    url = API + "?" + urllib.parse.urlencode(params)
    requete = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(requete, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def nettoyer(texte):
    """Enlève les balises HTML d'un champ de métadonnées."""
    return " ".join(html.unescape(re.sub(r"<[^>]+>", " ", texte or "")).split())


def licence_ok(nom):
    """CC0, domaine public, CC BY, CC BY-SA. Refuse NC (non commercial) et ND."""
    n = nom.lower()
    return n.startswith(("cc0", "cc by", "public domain", "pd")) and not re.search(r"-(nc|nd)\b", n)


def chercher(requete, limite=50):
    params = {"action": "query", "generator": "search", "gsrsearch": requete,
              "gsrnamespace": 6,  # 6 = pages de fichiers
              "gsrlimit": limite, "prop": "imageinfo",
              "iiprop": "url|size|mime|extmetadata", "iiurlwidth": 1024,
              "format": "json", "formatversion": 2}
    return appeler_api(params).get("query", {}).get("pages", [])


def candidat(page):
    """Renvoie les infos utiles si l'image convient, sinon None."""
    infos = page.get("imageinfo")
    if not infos:
        return None
    info = infos[0]
    meta = info.get("extmetadata", {})
    licence = nettoyer(meta.get("LicenseShortName", {}).get("value", ""))
    if info.get("mime") not in ("image/jpeg", "image/png") or not licence_ok(licence):
        return None
    if min(info.get("width", 0), info.get("height", 0)) < MIN_COTE:
        return None
    return {"titre": page.get("title", ""),
            "auteur": nettoyer(meta.get("Artist", {}).get("value", "")),
            "url_page": info.get("descriptionurl", ""),
            "licence": licence,
            "url_licence": meta.get("LicenseUrl", {}).get("value", ""),
            "url_image": info.get("thumburl") or info["url"],  # version 1024 px
            "ext": ".png" if info["mime"] == "image/png" else ".jpg"}


def telecharger(url, destination):
    requete = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(requete, timeout=60) as r, open(destination, "wb") as f:
        f.write(r.read())


def lire_csv(chemin):
    if not chemin.exists():
        return []
    with open(chemin, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def ecrire_csv(chemin, lignes):
    with open(chemin, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CHAMPS)
        w.writeheader()
        w.writerows(lignes)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--query", action="append", default=[], help="mots-clés (répétable)")
    p.add_argument("--n", type=int, default=40, help="nombre d'images à télécharger")
    p.add_argument("--out", default="data/raw/lora_images")
    p.add_argument("--csv", default="data/lora_sources.csv")
    p.add_argument("--sync", action="store_true",
                   help="garde dans le CSV seulement les images encore présentes")
    a = p.parse_args()

    dossier, chemin_csv = Path(a.out), Path(a.csv)
    dossier.mkdir(parents=True, exist_ok=True)
    lignes = lire_csv(chemin_csv)

    if a.sync:
        gardees = [l for l in lignes if (dossier / l["fichier"]).exists()]
        ecrire_csv(chemin_csv, gardees)
        print(f"{len(gardees)} images restantes, CSV mis à jour")
        raise SystemExit

    connus = {l["titre"] for l in lignes}
    for requete in a.query:
        if len(lignes) >= a.n:
            break
        for page in chercher(requete):
            if len(lignes) >= a.n:
                break
            c = candidat(page)
            if c is None or c["titre"] in connus:
                continue
            nom = f"commons_{page['pageid']}{c['ext']}"
            try:
                telecharger(c["url_image"], dossier / nom)
            except OSError as e:
                print(f"[échec] {c['titre']} : {e}")
                continue
            lignes.append({"fichier": nom, **{k: c[k] for k in CHAMPS[1:]}})
            connus.add(c["titre"])
            print(f"[{len(lignes):02d}] {nom}  {c['licence']}  {c['auteur'][:30]}")
            time.sleep(1)  # on ménage le serveur
    ecrire_csv(chemin_csv, lignes)
    print(f"{len(lignes)} images, sources notées dans {chemin_csv}")