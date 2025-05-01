from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
import pandas as pd
from transformers import AutoTokenizer, BartForConditionalGeneration
import nltk
import evaluate
import numpy as np
from transformers import AutoModelForSeq2SeqLM
from transformers import DataCollatorForSeq2Seq
import transformers
import matplotlib.pyplot as plt



df = pd.read_csv('corpus_test.csv',usecols=["text","summary"])
df = df.drop(82659) # on supprime cette ligne car text incomplet


# data split train_df 90% et test_df 10%
train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)

#  Convertir les DataFrames en objets Dataset
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# création du datasetDict
dataset = DatasetDict({
    "train": train_dataset,
    "test": test_dataset
})


# on charge le tokenizer de BART Large
tokenizer = AutoTokenizer.from_pretrained('facebook/bart-base')


# fonction qui prend un batch en entrée sous forme de dico
def preprocess_function(examples):

    # On tokenize les textes sources (ce qu’on veut résumer).
    model_inputs = tokenizer(
        examples['text'], max_length=1024, #  limite la longueur d’entrée à 1024 tokens (la taille max que BART accepte).
          truncation=True #  coupe les textes trop longs pour qu’ils tiennent dans 1024 tokens.
    )

    #  on tokenize les résumés
    labels = tokenizer(
        text_target=examples['summary'], max_length=128, truncation=True
    )
    # On ajoute les labels (tokenisés) aux model_inputs
    model_inputs['labels'] = labels['input_ids']
    return model_inputs


# On transforme en Token , tout le dataset, pour les futurs batch, et on supprime le nom des colonnes, car inutile pour le fine tuning
tokenized_df = dataset.map(preprocess_function, batched=True, remove_columns=["text", "summary"])


# on charge les métriques rouge Score, en local (car problème de pip install)
nltk.data.path.append("./punkt")
metric = evaluate.load('rouge')


# fonction, qui va évaluer chaque batch, avec les métriques Rouge Score
def compute_metrics(eval_preds):

    # Décompacte les prédictions des résumés et les labels (texte de référence) du tuple donné
    preds, labels = eval_preds

    # Remplace les valeurs -100 (utilisées pour ignorer certaines positions lors de l'entraînement)
    # par l'identifiant du token de padding, afin de pouvoir les décoder correctement
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    # Décode les séquences de tokens des prédictions en texte lisible
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Décode les séquences de tokens des labels en texte lisible
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Formate chaque prédiction et chaque label pour que chaque phrase soit sur une ligne distincte.
    # Cela améliore la précision du calcul ROUGE, qui est sensible à la structure des phrases.
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

    # Calcule les scores ROUGE (precision, recall, F1) entre les prédictions et les labels.
    # L'option use_stemmer=True permet de faire le calcul avec une racinisation des mots
    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

    # Retourne les résultats sous forme de dictionnaire
    return result


# Chargement du modèle BART-large pour la génération de texte (résumé)
model = AutoModelForSeq2SeqLM.from_pretrained('facebook/bart-base')


# Création du collator de données pour préparer le batch
# préparer les données pour chaque batch, en s'assurant que toutes les entrées sont correctement remplies, notamment les séquences de tokens qui nécessitent un padding.
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)


# Définir les arguments de l'entraînement
training_args = transformers.Seq2SeqTrainingArguments(
    output_dir='./bart_finetuning_results',
    eval_strategy='epoch',               # Evaluation à chaque epoch
    save_strategy='epoch',               # Sauvegarde à chaque epoch (doit MATCHER eval_strategy)
    logging_strategy='epoch',            # Logging à chaque epoch
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    weight_decay=0.01,
    save_total_limit=3,                  # Garde seulement 3 checkpoints
    load_best_model_at_end=True,         # Charge le meilleur modèle à la fin
    num_train_epochs=10,
    fp16=True,
    predict_with_generate=True,
    metric_for_best_model='eval_rougeL', # Métrique pour choisir le "meilleur" modèle
    greater_is_better=True               # Important si vous utilisez ROUGE (plus haut = mieux)
)


trainer = transformers.Seq2SeqTrainer(
    model=model,                                # Ton modèle BART
    args=training_args,                         # Les arguments d'entraînement
    train_dataset=tokenized_df['train'],        # Ton dataset d'entraînement tokenisé
    eval_dataset=tokenized_df['test'],          # Ton dataset de test tokenisé
    tokenizer=tokenizer,                        # Le tokenizer que tu utilises
    data_collator=data_collator,                # La fonction de préparation des données pour l'entraînement
    compute_metrics=compute_metrics             # La fonction pour évaluer les résultats avec ROUGE
)

trainer.train() # lancement de l'entrainement

trainer.save_model("./rep_model_fine_tuning_BART_base")  # sauvegarde du modèle
tokenizer.save_pretrained("./rep_tokenizer_fine_tuning_BART_base")  # Sauvegarde du tokenizer

# Accéder aux logs après l'entraînement
log_history = trainer.state.log_history
print(log_history)


# fonction pour enregistrer les plots de la cross validation et du scoring pour les métriques
def plot_training_metrics(log_history):
    # Extraire les données
    epochs = []
    train_losses = []
    eval_losses = []
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    rougeLsum_scores = []
    
    for log in log_history:
        if 'epoch' in log:
            epoch = log['epoch']
            if epoch not in epochs and isinstance(epoch, float):
                epochs.append(epoch)
            
            if 'loss' in log:  # Train loss
                train_losses.append((epoch, log['loss']))
            if 'eval_loss' in log:  # Eval loss
                eval_losses.append((epoch, log['eval_loss']))
            if 'eval_rouge1' in log:  # ROUGE scores
                rouge1_scores.append((epoch, log['eval_rouge1']))
                rouge2_scores.append((epoch, log['eval_rouge2']))
                rougeL_scores.append((epoch, log['eval_rougeL']))
                rougeLsum_scores.append((epoch, log['eval_rougeLsum']))
    
    # Trier par époque
    train_losses = sorted(train_losses, key=lambda x: x[0])
    eval_losses = sorted(eval_losses, key=lambda x: x[0])
    rouge1_scores = sorted(rouge1_scores, key=lambda x: x[0])
    rouge2_scores = sorted(rouge2_scores, key=lambda x: x[0])
    rougeL_scores = sorted(rougeL_scores, key=lambda x: x[0])
    rougeLsum_scores = sorted(rougeLsum_scores, key=lambda x: x[0])
    
    # Créer les graphiques
    plt.figure(figsize=(20, 5))
    
    # Graphique 1: Train et Validation Loss
    plt.subplot(1, 2, 1)
    if train_losses:
        plt.plot([x[0] for x in train_losses], [x[1] for x in train_losses], label='Train Loss', marker='o')
    if eval_losses:
        plt.plot([x[0] for x in eval_losses], [x[1] for x in eval_losses], label='Validation Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig("plots_bart_Base/loss_train_val.png")
    
    # Graphique 2: Scores ROUGE
    plt.subplot(1, 2, 2)
    if rouge1_scores:
        plt.plot([x[0] for x in rouge1_scores], [x[1] for x in rouge1_scores], label='ROUGE-1', marker='o')
    if rouge2_scores:
        plt.plot([x[0] for x in rouge2_scores], [x[1] for x in rouge2_scores], label='ROUGE-2', marker='o')
    if rougeL_scores:
        plt.plot([x[0] for x in rougeL_scores], [x[1] for x in rougeL_scores], label='ROUGE-L', marker='o')
    if rougeLsum_scores:
        plt.plot([x[0] for x in rougeLsum_scores], [x[1] for x in rougeLsum_scores], label='ROUGE-Lsum', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('ROUGE Score')
    plt.title('ROUGE Scores')
    plt.legend()
    plt.grid(True)
    plt.savefig("plots_bart_Base/val_rouge_scores.png")

    
    # Graphique 3: Comparaison entre Validation Loss et Scores ROUGE
    plt.figure(figsize=(7, 5))
    if eval_losses:
        plt.plot([x[0] for x in eval_losses], [x[1] for x in eval_losses], label='Validation Loss', marker='o', linestyle='--')
    if rouge1_scores:
        plt.plot([x[0] for x in rouge1_scores], [x[1] for x in rouge1_scores], label='ROUGE-1 Score', marker='x')
    if rouge2_scores:
        plt.plot([x[0] for x in rouge2_scores], [x[1] for x in rouge2_scores], label='ROUGE-2 Score', marker='x')
    if rougeL_scores:
        plt.plot([x[0] for x in rougeL_scores], [x[1] for x in rougeL_scores], label='ROUGE-L Score', marker='x')
    if rougeLsum_scores:
        plt.plot([x[0] for x in rougeLsum_scores], [x[1] for x in rougeLsum_scores], label='ROUGE-Lsum Score', marker='x')
    
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Comparison of Validation Loss and ROUGE Scores')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("plots_bart_Base/Comparison_val_score.png")
    plt.show()

plot_training_metrics(log_history)



