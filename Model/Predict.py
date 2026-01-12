from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = T5ForConditionalGeneration.from_pretrained("./Model/saved_model")
tokenizer = T5Tokenizer.from_pretrained("./Model/saved_model")

def malzeme_cikar(tarif_metni):
    input_ids = tokenizer(tarif_metni, return_tensors="pt", padding="max_length", truncation=True, max_length=128).input_ids

    model.eval()
    with torch.no_grad():
        generated_ids = model.generate(input_ids=input_ids, max_length=50)
        output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return output
