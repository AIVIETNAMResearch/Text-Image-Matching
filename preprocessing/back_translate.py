from transformers import MarianMTModel, MarianTokenizer
import json
import torch
from tqdm import tqdm

class BackTranslateAug:
    def __init__(self, first_model_name='Helsinki-NLP/opus-mt-en-vi', second_model_name='Helsinki-NLP/opus-mt-vi-en') -> None:
        self.first_model_tkn = MarianTokenizer.from_pretrained(first_model_name)
        self.first_model  = MarianMTModel.from_pretrained(first_model_name)

        self.second_model_tkn = MarianTokenizer.from_pretrained(second_model_name)
        self.second_model = MarianMTModel.from_pretrained(second_model_name)
    
    def format_batch_texts(self,language_code, batch_texts):
        formatted_batch = [">>{}<< {}".format(language_code, text) for text in batch_texts]

        return formatted_batch
    
    def translate(self, batch_texts, model, tokenizer, language="fr"):
        # Prepare the text data into appropriate format for the model
        formated_batch_texts = self.format_batch_texts(language, batch_texts)
        model = model.to(torch.device('cuda')).eval()
        # Generate translation using model
        tokenized_text = tokenizer(formated_batch_texts, return_tensors="pt", padding=True, max_length=512).to(torch.device('cuda'))
        translated = model.generate(**tokenized_text)

        # Convert the generated tokens indices back into text
        translated_texts = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
        
        return translated_texts

    def __call__(self, original_texts, language="vi"):
        translated_text = self.translate(original_texts, self.first_model, self.first_model_tkn)
        back_translate_text = self.translate(translated_text, self.second_model, self.second_model_tkn)
        return back_translate_text

def generate_aug():
    with open("./data/CUHK-PEDES/reid_raw.json", "r") as f:
        data = json.load(f)
    batch_size = 128
    for i in tqdm(range(0, len(data), batch_size), total=len(data)//batch_size):
        batch = data[i:i+batch_size]
        original_texts = [x["captions"][0] for x in batch]
        back_translate_text = BackTranslateAug()(original_texts)
        for j, x in enumerate(batch):
            data[i+j]["aug_cap_1"] = back_translate_text[j]
    
    for i in tqdm(range(0, len(data), batch_size), total=len(data)//batch_size):
        batch = data[i:i+batch_size]
        original_texts = [x["captions"][1] for x in batch]
        back_translate_text = BackTranslateAug()(original_texts)
        for j, x in enumerate(batch):
            data[i+j]["aug_cap_2"] = back_translate_text[j]

    with open("./data/CUHK-PEDES/reid_raw_back_translate.json", "w") as f:
        json.dump(data, f)


if __name__ == "__main__":
    generate_aug()