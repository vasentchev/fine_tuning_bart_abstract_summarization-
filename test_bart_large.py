import pandas as pd
from transformers import BartForConditionalGeneration, BartTokenizer
from rouge_score import rouge_scorer
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


# 1. Charger votre modèle fine-tuné et le tokenizer
model_path = "./save_fine_tuning/rep_model_fine_tuning_BART_large"
tokenizer_path = "./save_fine_tuning/rep_tokenizer_fine_tuning_BART_large"

tokenizer = BartTokenizer.from_pretrained(tokenizer_path)
model = BartForConditionalGeneration.from_pretrained(model_path)

# 2. Paramètres (ajustez selon votre cas)
MAX_INPUT_LENGTH = 1024  # en tokens
MAX_SUMMARY_LENGTH = 154  # correspond à votre longueur cible

# 3. Initialiser le scorer ROUGE
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL', 'rougeLsum'], use_stemmer=True)

def generate_summary(text):
    """Génère un résumé avec votre modèle fine-tuné"""
    inputs = tokenizer(
        text, 
        max_length=MAX_INPUT_LENGTH, 
        truncation=True, 
        return_tensors="pt"
    )
    
    summary_ids = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        num_beams=4,
        max_length=MAX_SUMMARY_LENGTH,
        early_stopping=True
    )
    
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def evaluate_rouge(dataset_path):
    """Évaluation complète avec calcul des scores ROUGE"""
    df = pd.read_csv(dataset_path)
    
    rouge_scores = {
        'rouge1': [],
        'rouge2': [],
        'rougeL': [],
        'rougeLsum': []
    }
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        reference = row['summary']
        generated = generate_summary(row['text'])
        
        scores = scorer.score(reference, generated)
        
        for key in rouge_scores.keys():
            rouge_scores[key].append(scores[key].fmeasure)
    
    # Calcul des moyennes
    return {k: np.mean(v) for k, v in rouge_scores.items()}

# 4. Exécution
if __name__ == "__main__":
    dataset_path = "df_test.csv"  # Remplacez par votre fichier
    
    print("Début de l'évaluation...")
    results = evaluate_rouge(dataset_path)
    
    print("\nRésultats ROUGE pour votre modèle fine-tuné:")
    print(f"ROUGE-1: {results['rouge1']:.4f}")
    print(f"ROUGE-2: {results['rouge2']:.4f}")
    print(f"ROUGE-L: {results['rougeL']:.4f}")
    print(f"ROUGE-Lsum: {results['rougeLsum']:.4f}")


    # sauvegarde du plot en formt png
    labels = list(results.keys())
    scores = list(results.values())

    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, scores, color='skyblue')
    plt.ylim(0, 1)
    plt.title("Scores ROUGE du modèle fine-tuné")
    plt.xlabel("Type de score ROUGE")
    plt.ylabel("F-mesure moyenne")

    # Annoter les barres avec les valeurs
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f"{yval:.3f}", ha='center', va='bottom')

    # Sauvegarde du graphique
    plt.tight_layout()
    plt.savefig("plots_bart_Large/scores_rouge_bart_large.png")  # <-- fichier PNG sauvegardé ici
    plt.close()
