# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import DatasetDict, Dataset
import transformers 

print(transformers.__version__)

# %%
model_checkpoint = "t5-small"

# %%
from evaluate import load
metric = load("rouge")

# %%
df = pd.read_csv('https://raw.githubusercontent.com/KAILASHVenkat/Paraphrasing_model/main/filtered_data.csv')
df.shape

# %%
max_length_input_text = df['input_text'].str.len().max()
max_length_target_text = df['target_text'].str.len().max()
print(max_length_input_text)
print(max_length_target_text)

# %%
train_data, temp_data = train_test_split(df, test_size=0.2, random_state=42)

# Step 2: Split the temp data into validation and test sets (50% validation, 50% test)
validation_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

train_data.reset_index(drop=True, inplace=True)
validation_data.reset_index(drop=True, inplace=True)
test_data.reset_index(drop=True, inplace=True)

# Step 3: Save the datasets under the variable name raw_datasets
raw_datasets = DatasetDict({
    'train': Dataset.from_pandas(train_data[['input_text', 'target_text']]),
    'validation': Dataset.from_pandas(validation_data[['input_text', 'target_text']]),
    'test': Dataset.from_pandas(test_data[['input_text', 'target_text']])
})

# %%
raw_datasets

# %%
import datasets
import random
import pandas as pd
from IPython.display import display, HTML

def show_random_elements(dataset, num_examples=5):
    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset)-1)
        while pick in picks:
            pick = random.randint(0, len(dataset)-1)
        picks.append(pick)
    
    df = pd.DataFrame(dataset[picks])
    for column, typ in dataset.features.items():
        if isinstance(typ, datasets.ClassLabel):
            df[column] = df[column].transform(lambda i: typ.names[i])
    display(HTML(df.to_html()))

# %%
show_random_elements(raw_datasets["train"])

# %%
metric

# %%
from transformers import AutoTokenizer
    
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# %%
if model_checkpoint in ["t5-small", "t5-base", "t5-larg", "t5-3b", "t5-11b"]:
    prefix = "paraphrase: "
else:
    prefix = ""

# %%
max_input_length = 650
max_target_length = 580

def preprocess_function(examples):
    inputs = [doc for doc in examples["input_text"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # Setup the tokenizer for targets
    labels = tokenizer(text_target=examples["target_text"], max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# %%
preprocess_function(raw_datasets['train'][:2])

# %%
tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)

# %%
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer

model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

# %%
batch_size = 8
model_name = model_checkpoint.split("/")[-1]
args = Seq2SeqTrainingArguments(
    f"{model_name}-finetuned-T5",
    evaluation_strategy = "epoch",
    learning_rate=7e-3,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=5,
    predict_with_generate=True,
    fp16=False,
    push_to_hub=False,
)

# %%
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# %%
import nltk
import numpy as np
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu, corpus_bleu

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Rouge expects a newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

    # Calculate BLEU score
    smoothing = SmoothingFunction().method1
    bleu_score = corpus_bleu([[ref.split()] for ref in decoded_labels], [pred.split() for pred in decoded_preds], smoothing_function=smoothing)
    
    # Calculate Exact Sentence-level Recall (Exact SR) and Exact F1 (Exact FE)
    exact_sr = sum([1 for label, pred in zip(decoded_labels, decoded_preds) if label == pred]) / len(decoded_labels)
    exact_fe = 2 * (exact_sr * bleu_score) / (exact_sr + bleu_score) if (exact_sr + bleu_score) > 0 else 0.0

    # ROUGE scores (existing code)
    rouge_output = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True, use_aggregator=True)
    rouge_scores = {key: value * 100 for key, value in rouge_output.items()}

    # Add mean generated length
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    gen_len = np.mean(prediction_lens)

    result = {
        "gen_len": gen_len,
        "bleu": bleu_score * 100,
        "exact_sr": exact_sr * 100,
        "exact_fe": exact_fe * 100,
        **rouge_scores,
    }

    return {k: round(v, 4) for k, v in result.items()}

# %%
trainer = Seq2SeqTrainer(
model,
args,
train_dataset=tokenized_datasets["train"],
eval_dataset=tokenized_datasets["validation"],
data_collator=data_collator,
tokenizer=tokenizer,
compute_metrics=compute_metrics
)

# %%
trainer.train()

# %%
import torch

# %%
import torch

# Assuming tokenized_datasets["test"] contains your test dataset
sample_input = tokenized_datasets["test"][5]

# Tokenize the input
tokenized_input = tokenizer(sample_input["input_text"], return_tensors="pt", max_length=max_input_length, truncation=True)

# Move input tensors to the GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenized_input = {key: value.to(device) for key, value in tokenized_input.items()}

# Generate output
with torch.no_grad():
    generated_output = model.generate(
        **tokenized_input,
        max_length=400,  # Set the desired maximum length
        num_beams=4,     # You can adjust the number of beams for diverse outputs
    )

# Postprocess the Output
decoded_output = tokenizer.batch_decode(generated_output, skip_special_tokens=True)[0]

# Print input text and generated output
print("Input Text:")
print(sample_input["input_text"])
print("\nGenerated Output:")
print(decoded_output)

import pickle

Pkl_Filename = "Pickle_T5_Model.pkl"  
with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(model, file)


Pkl_Filename = "Pickle_T5_Tokenizer.pkl"  
with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(tokenizer, file)