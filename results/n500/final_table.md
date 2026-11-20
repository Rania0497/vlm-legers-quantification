| Modèle | Précision | VRAM pic (Go) | Temps/image (s) | BLEU-4 | CIDEr [IC95] | CLIPScore [IC95] | Mots/légende | Identiques à fp16 (%) |
|---|---|---|---|---|---|---|---|---|
| smolvlm2 | fp16 | 4.96 | 1.422 | 0.271 | 0.856 [0.801-0.915] | 0.840 [0.833-0.846] | 14.43 | 100.0 |
| smolvlm2 | 8bit | 3.36 | 2.573 | 0.285 | 0.917 [0.861-0.977] | 0.833 [0.826-0.839] | 13.14 | 42.6 |
| smolvlm2 | 4bit | 2.62 | 1.499 | 0.313 | 0.995 [0.935-1.058] | 0.822 [0.815-0.829] | 11.38 | 11.4 |
| qwen25vl | fp16 | 7.58 | 0.804 | 0.280 | 0.979 [0.919-1.046] | 0.814 [0.807-0.821] | 11.28 | 100.0 |
| qwen25vl | 8bit | 4.82 | 2.937 | 0.272 | 0.950 [0.891-1.015] | 0.818 [0.811-0.824] | 11.67 | 66.6 |
| qwen25vl | 4bit | 3.64 | 1.122 | 0.270 | 0.962 [0.900-1.028] | 0.816 [0.809-0.823] | 11.73 | 38.4 |

Notes : 500 images Flickr8k (graine 42), GPU Tesla T4, une seule exécution.
IC95 = intervalle de confiance bootstrap (10 000 tirages, graine 42).
BLEU-4 est calculé sur tout le corpus, donc sans intervalle.
Seul le décodeur de texte est quantifié : l'encodeur d'images, le connecteur et lm_head restent en fp16.
CLIPScore d'une légende humaine (référence) : 0.806.
