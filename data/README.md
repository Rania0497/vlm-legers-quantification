# Fiche des données

## Identité

- **Nom** : Flickr8k, sous-ensemble de 500 images
- **Tâche** : légender des images (image vers texte)
- **Langue des légendes** : anglais
- **Référence d'origine** : Hodosh, Young et Hockenmaier, "Framing Image Description as a Ranking Task", Journal of Artificial Intelligence Research, 47, 2013.
- **Copie utilisée** : jeu `tsystems/flickr8k` sur Hugging Face (8 091 images, environ 1,1 Go, colonnes `image`, `image_filename`, `captions`, `query`). C'est une copie communautaire, pas la source officielle.

## Sous-ensemble utilisé

- 500 images tirées au hasard avec la graine 42.
- La liste exacte est dans `data/flickr8k_subset_ids.csv` (suivie par Git), donc tout le monde obtient les mêmes images.
- 5 légendes humaines de référence par image, soit 2 500 légendes.

## Comment l'obtenir

Depuis la racine du projet, avec l'environnement virtuel activé :

    python data/download_flickr8k.py --n 500 --seed 42

Le script crée, hors de Git :

- `data/raw/flickr8k_subset/images/` : les 500 images.
- `data/raw/flickr8k_subset/captions.csv` : colonnes `image_filename`, `caption_index`, `caption`.

## Rôle dans le projet

Évaluation uniquement. Les VLMs génèrent une légende par image, et on la compare aux 5 légendes humaines. Aucun entraînement n'est fait sur ces 500 images.

## Licence et droits

- La fiche Hugging Face déclare la licence MIT, mais c'est une déclaration de la personne qui a mis le jeu en ligne, pas des auteurs d'origine.
- Les photos viennent de Flickr, et leurs droits d'origine peuvent différer. À vérifier avant toute réutilisation.
- Les images ne sont pas redistribuées dans ce dépôt. Elles sont retéléchargées par le script.
- Usage prévu : pédagogique et académique uniquement.

## Limites connues

- Légendes en anglais seulement.
- Contenu limité : surtout des personnes et des animaux dans des actions, donc des résultats qui ne se généralisent pas forcément à d'autres types d'images (zellige, architecture).
- Les légendes humaines sont subjectives et inégales : plusieurs formulations correctes existent pour une même image.
- 500 images : de petites différences de score entre modèles peuvent n'être que du bruit statistique.

## Considérations éthiques

- Beaucoup d'images montrent des personnes : on ne cherche pas à les identifier.
- Aucune donnée personnelle n'est ajoutée au dépôt.

## À venir

Pour l'option A (LoRA), un second mini-jeu de 20 à 50 images autorisées d'un domaine précis, avec un fichier CSV listant l'auteur, l'URL et la licence de chaque image.