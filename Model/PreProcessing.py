import pandas as pd
import re
import json
import spacy

nlp = spacy.load("xx_ent_wiki_sm")
nlp.add_pipe("sentencizer")  # cümle sınırlarını belirlemek için bileşeni ekliyoruz

# Metin temizleme fonksiyonu
def temizle_metin(metin):
    if pd.isna(metin):
        return ""
    metin = re.sub(r'\s+', ' ', metin)  # fazla boşluklar
    metin = re.sub(r'[^a-zA-Z0-9ğüşıöçĞÜŞİÖÇ\s.,]', '', metin)  # özel karakter temizliği
    return metin.strip().lower()

# Malzemeleri ayırma ve temizleme
def tokenize_malzemeler(metin):
    return [temizle_metin(m) for m in metin.split(',') if m.strip() != ""]

# Açıklamayı cümlelere ayırma
def tokenize_tarif(metin):
    metin = temizle_metin(metin)
    doc = nlp(metin)
    return " ".join([sent.text.strip() for sent in doc.sents])

# CSV dosyasını oku
df = pd.read_csv("tarifler.csv", encoding="utf-8-sig")

# Sütunları temizle ve işleyelim
df['foodname'] = df['foodname'].apply(temizle_metin)
df['foodcategory'] = df['foodcategory'].apply(temizle_metin)
df['city'] = df['city'].apply(temizle_metin)
df['materials'] = df['materials'].apply(tokenize_malzemeler)
df['description'] = df['description'].apply(tokenize_tarif)

# Temiz veriyi JSON olarak dışa aktar
temizlenmis = df.to_dict(orient="records")

with open("temizlenmis_tarifler.json", "w", encoding="utf-8") as f:
    json.dump(temizlenmis, f, ensure_ascii=False, indent=2)

print("✅ Temizleme tamamlandı. 'temizlenmis_tarifler.json' oluşturuldu.")
