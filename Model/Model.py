from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import Dataset
import torch
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Kullanılan cihaz: {device}")

# 1. Veri Yükleme ve Hazırlama
with open('Model/temizlenmis_tarifler.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

data_dict = {
    "input_text": [],
    "target_text": []
}

for item in data:
    description = item['description']
    malzemeler = ", ".join(item['materials'])
    
    input_text = f"tarif: {description}"
    target_text = malzemeler
    
    data_dict["input_text"].append(input_text)
    data_dict["target_text"].append(target_text)

dataset = Dataset.from_dict(data_dict)

# Model ve tokenizer yükleniyor
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

model.to(device)

# Tokenizasyon
def tokenize(example):
    model_inputs = tokenizer(example["input_text"], max_length=128, padding="max_length", truncation=True)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(example["target_text"], max_length=128, padding="max_length", truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = dataset.map(tokenize)

# Eğitim argümanları
training_args = Seq2SeqTrainingArguments(
    output_dir="./Model/saved_model/results",
    per_device_train_batch_size=24,
    num_train_epochs=400,
    learning_rate=5e-5,
    logging_steps=1,
    save_total_limit=1,
    predict_with_generate=True
)

# Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer
)

# Modeli eğit
trainer.train()
model.save_pretrained("./Model/saved_model")
tokenizer.save_pretrained("./Model/saved_model")
