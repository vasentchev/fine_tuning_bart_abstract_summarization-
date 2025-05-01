
# 📘 Instructions for Training and Evaluating BART Models

## 📥 Downloading the Training Dataset

```text
Download the [corpus.csv](https://www.kaggle.com/datasets/manueldesiretaira/dataset-for-text-summarization/data) file from Kaggle and place it at the root of the project directory.  
You can then proceed with training and evaluation.
```


## ✅ 1. Creating the Virtual Environment

Use `pyenv` to create a Python **3.12.3** virtual environment:

```bash
pyenv install 3.12.3
pyenv virtualenv 3.12.3 bart_env
pyenv activate bart_env
```

---

## 📦 2. Installing Dependencies

Install the required dependencies using `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

## 🚀 3. BART Base Model

### 🔧 Training (10 epochs)

```bash
python train_bart_base.py
```

### 📊 Evaluation (ROUGE scores)

```bash
python test_bart_base.py
```

📂 Training and ROUGE performance plots will be automatically saved in the folder:

```text
plots_bart_Base/
```

---

## 🚀 4. BART Large Model

### 🔧 Training (3 epochs)

```bash
python train_bart_large.py
```

### 📊 Evaluation (ROUGE scores)

```bash
python test_bart_large.py
```

📂 Training and ROUGE performance plots will be automatically saved in the folder:

```text
plots_bart_Large/
```

---

