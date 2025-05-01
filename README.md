
# 📘 Instructions pour l'entraînement et l'évaluation des modèles BART

## ✅ 1. Création de l’environnement virtuel

Utilisez `pyenv` pour créer un environnement Python **3.12.3** :

```bash
pyenv install 3.12.3
pyenv virtualenv 3.12.3 bart_env
pyenv activate bart_env
```

---

## 📦 2. Installation des dépendances

Installez les dépendances nécessaires via `requirements.txt` :

```bash
pip install -r requirements.txt
```

---

## 🚀 3. Modèle BART Base

### 🔧 Entraînement (10 époques)

```bash
python train_bart_base.py
```

### 📊 Évaluation (ROUGE scores)

```bash
python test_bart_base.py
```

📂 Les graphiques d'entrainement et de performance ROUGE seront automatiquement sauvegardés dans le dossier :

```text
plots_bart_Base/
```

---

## 🚀 4. Modèle BART Large

### 🔧 Entraînement (3 époques)

```bash
python train_bart_large.py
```

### 📊 Évaluation (ROUGE scores)

```bash
python test_bart_large.py
```

📂 Les graphiques d'entrainement et de performance ROUGE seront automatiquement sauvegardés dans le dossier :

```text
plots_bart_Large/
```
