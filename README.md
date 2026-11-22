# Petits modèles vision-langage : légendes, quantification et LoRA

Projet 7 - IA générative visuelle (EMSI).

Comparaison de deux petits modèles vision-langage en fp16, 8 bits et 4 bits, puis utilisation du meilleur compromis pour légender des images servant à entraîner une LoRA. Contrainte : 12 Go de VRAM maximum.

---

## 1. Notions utilisées

| Notion | Définition |
|---|---|
| VLM | Modèle qui prend une image et produit du texte. |
| Quantification | Stockage des poids sur moins de bits (16, 8 ou 4) pour réduire la mémoire. |
| fp16 / 8bit / 4bit | 2, 1 ou 0,5 octet par paramètre. fp16 sert de référence. |
| VRAM | Mémoire du GPU. |
| LoRA | Petites matrices entraînées en plus d'un modèle gelé. |
| BLEU, CIDEr | Comparaison avec les 5 légendes humaines de Flickr8k. |
| CLIPScore | Accord entre la légende et l'image. |
| Bootstrap | Intervalles de confiance à 95 % (10 000 tirages). |

---

## 2. Les deux modèles

| | SmolVLM2-2.2B-Instruct | Qwen2.5-VL-3B-Instruct |
|---|---|---|
| Créateur | Hugging Face | Alibaba (Qwen) |
| Taille | 2,2 milliards de paramètres | environ 3,75 milliards |
| Licence | Apache 2.0 | qwen-research |

Pourquoi ces deux modèles : ils tiennent dans 12 Go, ils sont ouverts, ils sont compatibles avec la quantification (bitsandbytes) et ils viennent de deux familles différentes.

---

## 3. Résultats principaux (500 images Flickr8k, GPU T4)

| Modèle | Précision | VRAM pic (Go) | Temps/image (s) | CIDEr | CLIPScore |
|---|---|---|---|---|---|
| smolvlm2 | fp16 | 4.96 | 1.42 | 0.856 | 0.840 |
| smolvlm2 | 8bit | 3.36 | 2.57 | 0.917 | 0.833 |
| smolvlm2 | 4bit | 2.62 | 1.50 | 0.995 | 0.822 |
| qwen25vl | fp16 | 7.58 | 0.80 | 0.979 | 0.814 |
| qwen25vl | 8bit | 4.82 | 2.94 | 0.950 | 0.818 |
| qwen25vl | 4bit | 3.64 | 1.12 | 0.962 | 0.816 |

Le tableau complet, avec BLEU-4 et les intervalles, est dans `results/n500/final_table.md`.

**Observations :**
1. **Mémoire** : le 4 bits divise la VRAM par environ 2 (-52 % et -53 %), et non par 4, parce que l'encodeur d'images reste en fp16.
2. **Vitesse** : le 8 bits est **plus lent** que le fp16. Le 4 bits est proche du fp16.
3. **Qualité** : pour Qwen, aucune perte nette en 4 bits. Pour SmolVLM2, CIDEr monte mais CLIPScore baisse : les métriques **se contredisent**, parce que les légendes compressées sont plus courtes.
4. **Choix du légendeur** : Qwen en 4 bits (3,6 Go, sans perte détectable). C'est un compromis mémoire/qualité.

---

## 4. La LoRA (option A)

- 30 photos de zellige (Wikimedia Commons), légendées automatiquement par Qwen 4 bits.
- LoRA sur Stable Diffusion 1.5 : rang 4, 250 étapes, 25 images pour apprendre et 5 pour tester.
- **Résultat : effet visible faible.** Hypothèses (non démontrées) : peu d'images, peu d'étapes, des légendes assez générales.
- But atteint : montrer que **toute la chaîne fonctionne** dans 12 Go.

---

## 5. Les 8 étapes du projet

1. **Dépôt et données** : structure du dépôt, 500 images Flickr8k tirées avec la graine 42.
2. **Choix des 2 VLMs** : `src/check_models.py` lit la licence et la taille sans rien télécharger.
3. **Code de légendage** : `src/captioning.py`, une seule fonction pour 2 modèles et 3 précisions.
4. **Les 6 tests** : `src/benchmark.py` mesure le temps, la VRAM et les légendes.
5. **Mesure de qualité** : BLEU, CIDEr, CLIPScore, accord avec fp16, intervalles bootstrap et exemples visuels.
6. **LoRA minimale** : téléchargement des images, légendes par Qwen, entraînement, comparaison avec et sans LoRA.
7. **Démo Gradio** : `demo/app.py`. On dépose une image, on choisit le modèle et la précision, et on lit la légende, le temps et la VRAM.
8. **Reproductibilité** : versions figées, notebooks nettoyés, test sur un clone vierge.

---

## 6. Organisation du dépôt

```
configs/    réglages (models.yaml)
data/       script de téléchargement + fiche des données (jamais les images)
demo/       app.py (Gradio) + vidéo
notebooks/  01 à 06, les notebooks Colab nettoyés
results/    tableaux et résultats (n500, lora)
src/        tout le code réutilisable
requirements*.txt   versions figées
```

---

## 7. Relancer le projet

Il faut un GPU NVIDIA. Un T4 gratuit sur Colab suffit.

```bash
git clone https://github.com/Rania0497/vlm-legers-quantification.git
cd vlm-legers-quantification
pip install -r requirements.txt -r requirements-lora.txt
pip uninstall -y torchao
python data/download_flickr8k.py
python -m src.benchmark --images-dir data/raw/flickr8k_subset/images --n 3
python -m demo.app --share
```

Pour rejouer tout le pipeline : `notebooks/06_test_clone_vierge.ipynb`.
Pour l'évaluation sans GPU : `requirements-eval.txt`, puis `pip install --no-deps pycocoevalcap==1.2`.

---

## 8. Limites

- Une seule exécution, sur un seul GPU T4 partagé.
- 500 images : les petites différences peuvent n'être que du bruit.
- Les métriques automatiques sont imparfaites et se contredisent parfois.
- Seul le décodeur de texte est quantifié.
- Les images de la LoRA ne sont pas redistribuées, à cause des licences.
