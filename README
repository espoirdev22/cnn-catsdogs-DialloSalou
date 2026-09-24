# CNN from scratch vs Transfert Learning — Cats vs Dogs

Comparaison de deux approches de classification binaire d'images : un CNN
entraîné entièrement à partir de zéro et un ResNet18 pré-entraîné sur
ImageNet en feature extraction.

**Résultat principal : 97,8 % d'accuracy en transfert learning contre
80,6 % from scratch, avec 50 000 fois moins de paramètres entraînés.**

---

## Table des matières

- [CNN from scratch vs Transfert Learning — Cats vs Dogs](#cnn-from-scratch-vs-transfert-learning--cats-vs-dogs)
  - [Table des matières](#table-des-matières)
  - [Objectif](#objectif)
  - [Environnement](#environnement)
  - [Organisation des données](#organisation-des-données)
  - [Structure du dépôt](#structure-du-dépôt)
  - [Méthodologie](#méthodologie)
    - [Modèle A : CNN from scratch](#modèle-a--cnn-from-scratch)
    - [Modèle B : transfert learning (ResNet18)](#modèle-b--transfert-learning-resnet18)
  - [Entraînement](#entraînement)
  - [Évaluation](#évaluation)
  - [Résultats](#résultats)
    - [Recherche du learning rate — Modèle A](#recherche-du-learning-rate--modèle-a)
    - [Recherche du learning rate — Modèle B](#recherche-du-learning-rate--modèle-b)
    - [Courbes comparées (validation)](#courbes-comparées-validation)
    - [Résultats finaux sur le jeu de test](#résultats-finaux-sur-le-jeu-de-test)
    - [Matrices de confusion](#matrices-de-confusion)
    - [Erreurs typiques](#erreurs-typiques)
  - [Analyse](#analyse)
  - [Limites et pistes d'amélioration](#limites-et-pistes-damélioration)
  - [Auteur](#auteur)

---

## Objectif

Ce projet compare deux stratégies pour classer des images de chats et de
chiens :

- **Expérience A** : un CNN convolutif à trois blocs, entraîné entièrement
  depuis une initialisation aléatoire.
- **Expérience B** : un ResNet18 pré-entraîné sur ImageNet, dont le
  backbone est gelé et seule la couche de classification est réentraînée.

L'objectif est de mesurer l'impact du transfert d'apprentissage sur trois
dimensions : la performance finale, la vitesse de convergence et le coût
d'entraînement.

---

## Environnement

Développé et exécuté sur **Google Colab**, GPU **Tesla T4**, Python 3.13.

```bash
pip install -r requirements.txt
```

| Bibliothèque | Version |
|---|---|
| torch | 2.11.0 |
| torchvision | 0.26.0 |
| scikit-learn | 1.6.1 |
| numpy | 2.1.3 |
| matplotlib | 3.10.0 |

Les versions de `torch` et `torchvision` correspondent à celles de Colab.
Pour une installation locale avec GPU, voir [pytorch.org](https://pytorch.org)
selon la version de CUDA disponible.

L'entraînement sur CPU est possible mais déconseillé : environ 2 minutes par
epoch sur T4 contre plusieurs heures sur processeur.

**Reproductibilité.** Les graines de `random`, `numpy` et `torch` (CPU et GPU)
sont fixées à 42. cuDNN est configuré en mode déterministe
(`deterministic = True`, `benchmark = False`), au prix d'un léger
ralentissement des convolutions.

---

## Organisation des données

Jeu de données : [Cats vs Dogs (Udacity)](https://s3.amazonaws.com/content.udacity-data.com/nd089/Cat_Dog_data.zip)

Le jeu de données n'est pas inclus dans ce dépôt en raison de sa taille.
Téléchargez l'archive avec le lien ci-dessus et décompressez-la à la racine
du projet pour obtenir la structure suivante :

```
Cat_Dog_data/
├─ train/
│  ├─ cat/    11 275 images
│  └─ dog/    11 250 images
└─ test/
   ├─ cat/     1 250 images
   └─ dog/     1 250 images
```

**Découpage.** Le dossier `train` est divisé en 80 % entraînement
(18 020 images) et 20 % validation (4 505 images) via `random_split` avec
un générateur à graine fixe.

Le dossier `test` (2 500 images) est réservé à l'évaluation finale. Il
n'intervient ni dans l'entraînement, ni dans le choix des hyperparamètres,
ni dans la sélection du checkpoint.

**Prétraitement.**

| Jeu | Transformations |
|---|---|
| Entraînement | RandomResizedCrop(224), RandomHorizontalFlip, RandomRotation(15), ToTensor, Normalize |
| Validation / Test | Resize(256), CenterCrop(224), ToTensor, Normalize |

Normalisation avec les statistiques ImageNet
(`mean = [0.485, 0.456, 0.406]`, `std = [0.229, 0.224, 0.225]`),
indispensables pour le modèle pré-entraîné et appliquées aux deux modèles
pour garantir des conditions identiques.

Le dossier `train` est chargé deux fois, avec et sans augmentation, puis
découpé avec le même générateur. Cela garantit que la validation n'hérite
pas des transformations aléatoires du train, tout en portant sur exactement
les mêmes images.

**Note d'exécution.** Les données sont copiées de Google Drive vers le
disque local de Colab avant l'entraînement. La lecture de 22 500 fichiers
depuis Drive était le goulot d'étranglement, le GPU restant sous-utilisé.

---

## Structure du dépôt

```
cnn-catsdogs-DialloSalou/
├─ notebook.ipynb          # notebook complet, de la préparation à l'analyse
├─ figures/
│  ├─ comparaison.png      # courbes comparées des deux modèles
│  ├─ confusion.png        # matrices de confusion sur le test
│  └─ erreurs.png          # exemples d'images mal classées
├─ requirements.txt
├─ .gitignore
└─ README.md
```

Les checkpoints (`*.pth`) et les données ne sont pas versionnés.

---

## Méthodologie

### Modèle A : CNN from scratch

```
Entrée  [3, 224, 224]
Bloc 1  Conv2d(3→32)   + BatchNorm + ReLU + MaxPool   →  [32, 112, 112]
Bloc 2  Conv2d(32→64)  + BatchNorm + ReLU + MaxPool   →  [64, 56, 56]
Bloc 3  Conv2d(64→128) + BatchNorm + ReLU + MaxPool   →  [128, 28, 28]
Flatten                                               →  100 352
Linear(100352 → 512) + ReLU + Dropout(0.5)
Linear(512 → 2)
```

Toutes les convolutions utilisent `kernel_size=3`, `stride=1`, `padding=1`.
La réduction spatiale est confiée aux MaxPool.

**Total : 51 475 458 paramètres**, tous entraînés.

**BatchNorm** : placée après chaque convolution, avant la ReLU. Elle
recentre les sorties de la convolution, ce qui stabilise l'apprentissage et
autorise un learning rate plus élevé.

**Dropout** : placé uniquement dans le classifieur, après la première
couche Linear. Plus de 99 % des paramètres du réseau se trouvent dans
`fc1`, le risque de surapprentissage y est donc concentré. Les blocs
convolutifs sont déjà régularisés par le partage de poids et la BatchNorm.

### Modèle B : transfert learning (ResNet18)

```
ResNet18 pré-entraîné sur ImageNet
├─ backbone (conv1 → layer4)     gelé, requires_grad = False
└─ fc : Linear(512 → 1000)       remplacé par Linear(512 → 2)
```

**1 026 paramètres entraînables sur 11 177 538**, soit 0,009 % du réseau.

**Stratégie : feature extraction.** Le backbone est entièrement gelé. Ce
choix s'explique par la taille modérée du jeu de données (18 020 images) et
sa proximité avec le domaine d'ImageNet, qui contient déjà des races de
chats et de chiens. L'alternative, le fine-tuning, n'a pas été testée.

---

## Entraînement

Le notebook s'exécute de haut en bas. Les sections 4 et 5 contiennent les
quatre expériences.

**Hyperparamètres communs**

| Paramètre | Valeur |
|---|---|
| Epochs | 15 |
| Batch size | 32 |
| Fonction de perte | CrossEntropyLoss |
| Seed | 42 |

**Configurations testées**

| Modèle | Optimiseur | Learning rate | Momentum |
|---|---|---|---|
| A | Adam | 0.001 puis 0.0001 | — |
| A | SGD | 0.01 puis 0.001 | 0.9 |
| B | Adam | 0.001 | — |
| B | SGD | 0.01 | 0.9 |

**Checkpoints.** Deux fichiers par expérience :

- `*_last.pth` : état complet (poids, optimiseur, epoch, historique,
  meilleur score), écrit à chaque epoch pour permettre la reprise après
  interruption.
- `*_best.pth` : meilleur epoch sur la validation, utilisé pour le test
  final.

L'écriture est atomique (fichier temporaire puis renommage) afin d'éviter
les fichiers corrompus en cas de coupure. Les checkpoints sont écrits sur
le disque local puis copiés vers Drive toutes les 3 epochs, l'écriture
directe de fichiers de 590 Mo sur Drive à chaque epoch provoquant des
corruptions.

Au lancement, `load_checkpoint` détecte automatiquement un checkpoint
existant et reprend à l'epoch suivante.

---

## Évaluation

Le test final (section 7 du notebook) recharge chaque modèle depuis son
checkpoint dans une architecture vide, puis l'évalue sur les 2 500 images
de test.

```python
model_a_test = ScratchCNN().to(device)
ckpt = torch.load("checkpoints/scratch_adam_lr1e4_best.pth", map_location=device)
model_a_test.load_state_dict(ckpt["model_state"])
evaluate(model_a_test, testloader, criterion, device)
```

```python
model_b_test = models.resnet18(weights=None)
model_b_test.fc = nn.Linear(model_b_test.fc.in_features, 2)
model_b_test = model_b_test.to(device)
ckpt = torch.load("checkpoints/resnet_adam_best.pth", map_location=device)
model_b_test.load_state_dict(ckpt["model_state"])
evaluate(model_b_test, testloader, criterion, device)
```

Précision et recall sont calculés en moyenne pondérée sur les deux classes
(`average='weighted'`). Les classes étant équilibrées, cette moyenne
reflète la performance globale sans favoriser une classe.

---

## Résultats

### Recherche du learning rate — Modèle A

| Optimiseur | Learning rate | Accuracy finale | Val loss | Observation |
|---|---:|---:|---:|---|
| Adam | 0.001 | 0.609 | 0.652 | Apprentissage bloqué |
| Adam | 0.0001 | **0.823** | **0.410** | Meilleur résultat |
| SGD (momentum 0.9) | 0.01 | 0.502 | 0.693 | Prédit une seule classe |
| SGD (momentum 0.9) | 0.001 | 0.782 | 0.500 | Convergence lente mais régulière |

Les deux optimiseurs ont d'abord échoué à leur valeur usuelle. Adam a
divergé dès la première epoch (train_loss à 1.41) avant de se figer à
60 %. SGD s'est effondré sur une prédiction constante, l'accuracy de
0.502 correspondant exactement à la proportion de la classe majoritaire.

Cette sensibilité s'explique par l'architecture : avec 99 % des paramètres
concentrés dans `fc1`, un pas trop grand y provoque des mises à jour
massives qui déstabilisent le réseau entier.

### Recherche du learning rate — Modèle B

| Optimiseur | Learning rate | Meilleure accuracy | Val loss | Observation |
|---|---:|---:|---:|---|
| Adam | 0.001 | **0.981** | **0.057** | Stable dès l'epoch 3 |
| SGD (momentum 0.9) | 0.01 | 0.978 | 0.099 | Fortes oscillations |

Les deux optimiseurs atteignent un niveau comparable, mais SGD oscille
fortement : sa val loss passe de 0.067 à l'epoch 5 à 0.234 à l'epoch 6.
Un learning rate de 0.01 est trop élevé pour une couche de 1 026
paramètres. Le comportement est inversé par rapport au modèle A, où SGD
était le plus régulier : le choix de l'optimiseur dépend de l'architecture.

### Courbes comparées (validation)

![Comparaison des deux modèles](figures/comparaison.png)

### Résultats finaux sur le jeu de test

| Modèle | Accuracy | Précision | Recall | Loss | Paramètres entraînés | Checkpoint |
|---|---:|---:|---:|---:|---:|---:|
| A — CNN from scratch (Adam, lr 1e-4) | 80,6 % | 80,6 % | 80,6 % | 0,427 | 51 475 458 | 590 Mo |
| B — ResNet18 gelé (Adam, lr 1e-3) | **97,8 %** | **97,8 %** | **97,8 %** | **0,057** | **1 026** | **43 Mo** |

Les deux modèles conservent sur le test un score proche de leur meilleure
validation (82,3 % pour A, 98,1 % pour B), ce qui indique une
généralisation correcte et l'absence de surapprentissage sur la validation.

### Matrices de confusion

![Matrices de confusion](figures/confusion.png)

| Modèle | Erreurs totales | Chats → chiens | Chiens → chats |
|---|---:|---:|---:|
| A | 486 / 2 500 | 274 | 212 |
| B | 56 / 2 500 | 12 | 44 |

Le modèle B commet presque 9 fois moins d'erreurs. Les deux modèles ne
penchent pas du même côté : A se trompe un peu plus sur les chats, B
nettement plus sur les chiens.

### Erreurs typiques

![Exemples d'erreurs](figures/erreurs.png)

Les images mal classées présentent souvent des difficultés visibles :
visage caché derrière des barreaux ou un grillage, tête coupée par le
recadrage central, image floue, ou présence des deux animaux sur la même
photo avec une seule étiquette.

La confiance du modèle confirme ces difficultés : **98 % en moyenne sur les
bonnes réponses contre 73 % sur les erreurs**. Le modèle hésite davantage
quand il se trompe, même si certaines erreurs restent faites avec
assurance.

---

## Analyse

**Performance.** Le transfert learning obtient 97,8 % contre 80,6 %, soit
17,2 points de pourcentage d'écart, ou 9 fois moins d'erreurs en valeur
absolue. Cet écart n'est pas marginal : il sépare un modèle utilisable en
production d'un modèle qui se trompe une fois sur cinq.

**Convergence.** Le modèle B atteint 95,6 % dès la première epoch et
plafonne à partir de la troisième. Sa première epoch dépasse déjà le
résultat du modèle A après quinze. À l'inverse, le modèle A progressait
encore à l'epoch 15, ses pertes descendaient toujours : il n'avait pas
convergé sur ce budget.

**Coût.** Le contraste le plus frappant concerne les paramètres. Le modèle
B entraîne 1 026 valeurs contre 51 475 458, soit 50 000 fois moins, et
produit un checkpoint quatorze fois plus léger. Un réseau plus performant
n'est donc pas un réseau plus gros, mais un réseau dont les
représentations ont déjà été apprises sur un corpus bien plus vaste
(1,2 million d'images contre 18 020).

**Conclusion.** Sur un jeu de données de taille modérée et proche du
domaine d'ImageNet, réutiliser un backbone pré-entraîné est nettement plus
efficace que de tout apprendre depuis zéro, à la fois en performance, en
temps de calcul et en ressources. La recherche du learning rate s'est
également révélée déterminante : aucun des deux modèles n'apprenait
correctement avec les valeurs testées en premier.

---

## Limites et pistes d'amélioration

**Architecture du modèle A.** Plus de 99 % de ses paramètres se trouvent
dans `fc1`, conséquence du Flatten appliqué sur une sortie de
128 × 28 × 28. Un `AdaptiveAvgPool2d(1)` avant le classifieur réduirait le
modèle à moins de 200 000 paramètres, allégerait les checkpoints et
limiterait le risque de surapprentissage.

**Budget d'entraînement.** Le modèle A n'avait pas convergé à l'epoch 15.
Un entraînement plus long améliorerait probablement son score, sans pour
autant combler l'écart avec le transfert learning.

**Optimisation de SGD.** Sur le modèle B, le learning rate de 0.01 provoque
de fortes oscillations. Une valeur plus faible ou un scheduler
(`StepLR`, `CosineAnnealingLR`) stabiliserait l'apprentissage.

**Fine-tuning non testé.** Seul le feature extraction a été évalué.
Dégeler `layer4` de ResNet18 avec un learning rate très faible
(de l'ordre de 1e-5) pourrait encore améliorer les résultats.

**Qualité du jeu de données.** Certaines images de test contiennent les
deux animaux avec une seule étiquette. Ces cas sont comptés comme des
erreurs alors que la prédiction n'est pas nécessairement fausse.

**Journalisation.** Le suivi des métriques repose sur un dictionnaire
sauvegardé dans les checkpoints. Un outil dédié (TensorBoard, W&B)
faciliterait la comparaison d'un plus grand nombre d'expériences.

---

## Auteur

Diallo Salou — Master 1 Intelligence Artificielle, Dakar Institute of Technology