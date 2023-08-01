import spacy
import json
import numpy as np
from tqdm import tqdm
human_descriptions = ["man", "woman", "women", "pedestrian"]


def extract_noun_chunks(doc):

    chunk_indices = []
    for chunk in doc.noun_chunks:
        valid = True
        for token in chunk:
            if token.pos_ == "PRON" or token.text.lower() in human_descriptions:
                valid = False
        chunk_indices.append(list(range(chunk.start, chunk.end))) if valid else None
        
    return chunk_indices

def main():
    with open("./data/CUHK-PEDES/reid_raw.json", "r") as f:
        data = json.load(f)
    nlp = spacy.load('en_core_web_sm')

    chunk_nums = []
    tmp_data = []
    zero_chunk = 0

    for idx, sample in tqdm(enumerate(data)):
        docs = nlp.pipe(sample["captions"])
        chunks = []
        for doc in docs:
            chunk_indicies = extract_noun_chunks(doc)
            chunks.append(list(chunk_indicies))
            chunk_nums.append(len(chunk_indicies))
        

        tmp_data.append(sample)
        tmp_data[idx]["noun_chunks"] = list(chunks)

    print(np.mean(chunk_nums), " MEAN")
    print(np.max(chunk_nums), " MAX")
    print(np.min(chunk_nums), " MIN")

    with open("./data/CUHK-PEDES/reid_noun_chunks.json", "w") as f:
        json.dump(tmp_data, f)

if __name__ == "__main__":
    main()