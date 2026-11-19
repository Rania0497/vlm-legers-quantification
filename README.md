# Architecture globale du projet
vlm-legers-quantification/
|-- README.md              présentation + comment relancer
|-- .gitignore             liste de ce que Git ignore
|-- requirements.txt       bibliothèques aux versions figées
|-- LICENSE                licence du code
|-- notebooks/             notebooks numérotés
|     01_donnees.ipynb, 02_baseline.ipynb, 03_modele.ipynb,
|     04_evaluation.ipynb, 05_lora.ipynb
|-- src/                   code réutilisable (.py)
|     captioning.py, quantization.py, metrics.py
|-- configs/               paramètres des expériences (.yaml)
|-- data/                  script de téléchargement + fiche des données,
|                          jamais les images elles-mêmes
|-- results/               tableaux, figures, exemples de légendes
|-- demo/                  app.py (Gradio) + courte vidéo
|-- report/                rapport final
|-- docs/                  guide d'utilisation, licences, fiche du modèle, éthique
|-- .venv/                  


# ETAPE 1 : git, environnement et données
1. Sur github.com
Clique sur “New repository”.
Nom : vlm-legers-quantification (ca sera le nom du dossier).
Visibilité : Public si ton prof doit le consulter facilement, sinon Private, puis ajoute-le comme collaborateur.
Coche “Add a README file”.
Dans “.gitignore”, choisis le modèle Python. Il évite d'envoyer par erreur de gros fichiers inutiles.
Clique sur “Create repository”.

2. Sur le terminal du PC
Vérifier que Git est installé
    git --version
Dire à Git qui tu es (une seule fois sur ton PC)
    git config --global user.name "Rania"
    git config --global user.email "ton-email-github@exemple.com"
Se positionner où je veux
    cd /d "E:\z- Projets en cours (Storage Year)"
Copier le dépôt GitHub sur ton PC
    git clone https://github.com/Rania0497/vlm-legers-quantification.git
Entrer dans le dossier du projet
    cd vlm-legers-quantification
Créer la structure du guide
    mkdir notebooks src configs data results demo report docs
    for %d in (notebooks src configs data results demo report docs) do type nul > %d\.gitkeep
    dir /s /b *.gitkeep

3. Créer l’environnement virtuel
Se placer dans le projet et vérifier Python
    cd /d "E:\z- Projets en cours (Storage Year)\vlm-legers-quantification"
    python --version
Créer l’environnement
    python -m venv .venv
L’activer 
    .venv\Scripts\activate

4. Premier commit et envoi sur GitHub
Maintenant on sauvegarde la structure dans Git, puis on l’envoie sur GitHub. Le venv ne sera pas envoyé, car le .gitignore Python l’ignore.
Voir ce que Git a repéré. Tu dois voir les 8 fichiers .gitkeep en “untracked files”, et ne pas voir .venv.
    cd /d "E:\z- Projets en cours (Storage Year)\vlm-legers-quantification"
    git status
Préparer, enregistrer, envoyer. J’écris le message sans accents, car cmd sur Windows déforme parfois les accents dans les messages de commit.
    git add .
    git commit -m "Ajoute la structure de dossiers du projet"
    git push
Vérifier
    git log --oneline
Tu dois voir ton nouveau commit tout en haut, au-dessus du commit initial. Puis ouvre la page du dépôt sur github.com : les 8 dossiers doivent apparaître.

5. Script de téléchargement de Flickr8k (DONNEES)
J’ai pris la copie Hugging Face tsystems/flickr8k recommendé par le prof. Sa fiche indique 8 091 exemples, environ 1,1 Go de téléchargement, et les colonnes image image_filename, captions (liste de légendes) et query. Le script en tire 500 images tirées au hasard avec une graine fixe, pour que tout le monde obtienne les mêmes 500.
Installer les bibliothèques dans le venv
    python -m pip install datasets pillow
Créer le script. Réponds “Oui” pour créer le fichier
    notepad data\download_flickr8k.py
Lancer et vérifier
    python data\download_flickr8k.py --n 500 --seed 42
    python -c "from pathlib import Path; print(len(list(Path('data/raw/flickr8k_subset/images').glob('*'))))"
    git status

6. La fiche des données
La fiche des données est la carte d’identité de ton jeu d’images : d’où il vient, ce qu’il contient, ce qu’on a le droit d’en faire et ses limites. Le guide l’exige dans data/, et le prof peut te poser des questions dessus.
    notepad data\README.md

7. Github
    git add README.md data/download_flickr8k.py data/flickr8k_subset_ids.csv
    git commit -m "Etape1"
    git push
    git status

# Étape 2 : choisir les 2 VLMs
Pour comparer proprement, il faut deux modèles de vision-langage (VLM) différents mais comparables. On les choisit selon 3 critères : ils tiennent dans 12 Go, ils sont ouverts, et ils sont compatibles avec la quantification via transformers.
SmolVLM2-2.2B-Instruct : un modèle léger développé par Hugging Face, distribué sous licence Apache 2.0.
Qwen2.5-VL-3B-Instruct : un modèle développé par Alibaba, appartenant à la famille Qwen.
Ce choix permet de comparer deux familles de modèles différentes, avec des tailles relativement proches, afin d'étudier l'effet de la quantification sur leur consommation mémoire et leurs performances.
1. Un script qui interroge le Hub sans rien télécharger (src/check_models.py)
2. Un fichier de config (configs/models.yaml)
3. Lancer
    pip install huggingface_hub
    python src/check_models.py
4. Github
git add src/check_models.py configs/models.yaml README.md
git status
git commit -m "Etape 2 : choix des 2 VLMs (SmolVLM2-2.2B et Qwen2.5-VL-3B) et script de verification"
git push
git status





