import json
import os
import random
from PIL import Image
import cv2
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from utils.simple_tokenizer import SimpleTokenizer
from preprocessing.text_aug import eda
import random 
import numpy as np


def default_loader(path):
    return Image.open(path).convert('RGB')

def cv2_loader(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

# clip tokenize function
def tokenize(caption: str, tokenizer, text_length=77, truncate=True) -> torch.LongTensor:
    sot_token = tokenizer.encoder["<|startoftext|>"]
    eot_token = tokenizer.encoder["<|endoftext|>"]
    tokens = [sot_token] + tokenizer.encode(caption) + [eot_token]

    result = torch.zeros(text_length, dtype=torch.long)
    if len(tokens) > text_length:
        if truncate:
            tokens = tokens[:text_length]
            tokens[-1] = eot_token
        else:
            raise RuntimeError(
                f"Input {caption} is too long for context length {text_length}"
            )
    result[:len(tokens)] = torch.tensor(tokens)
    return result


class TextImageDataset(Dataset):
    def __init__(self, data_cfg, json_path, transform=None, 
                 split="train", mode='train') -> None:
        super().__init__()
        self.name = data_cfg.NAME
        self.data_cfg = data_cfg.clone()
        self.root_dir = os.path.join(data_cfg.DATA_DIR, self.name)
        self.json_path = os.path.join(self.root_dir, json_path)
        self.transform = transform
        self.mode = mode
        if mode not in ['train', 'text', 'image']:
            raise ValueError('Invalid mode. Must be one of [train, text, image]')

        with open(self.json_path, 'r') as f:
            self.data = json.load(f)

        self.split = split
        max_len = self.data_cfg.TEXT_MAX_LEN if self.data_cfg.TEXT_MAX_LEN else 77
        truncate = self.data_cfg.TEXT_TRUNCATE if self.data_cfg.TEXT_TRUNCATE else True
        self.tokenizer = SimpleTokenizer(bpe_path=self.data_cfg.BPE_PATH)

        tmp_data = []
        captions, caption_pids, images, image_ids = [], [], [], []
        
        for sample in self.data:
            if sample['split'] == self.split:
                for cap in sample['captions']:
                    cap = tokenize(cap, self.tokenizer, max_len, truncate)
                    captions.append(cap)
                    caption_pids.append(sample['id'] - 1)
                    tmp_data.append((sample['id'] - 1, sample['file_path' if data_cfg.NAME == "CUHK-PEDES" else "img_path"], cap))
                images.append(sample['file_path' if data_cfg.NAME == "CUHK-PEDES" else "img_path"])
                image_ids.append(sample['id'] - 1) 
        self.captions = captions
        self.caption_pids = caption_pids
        self.images = images
        self.image_ids = image_ids
        self.data = tmp_data

    def __len__(self):
        if self.mode == 'train':
            return len(self.data)
        elif self.mode == 'text':
            return len(self.captions)
        elif self.mode == 'image':
            return len(self.images)
        else:
            raise ValueError('Invalid mode. Must be one of [train, text, image]')
    
    def __getitem__(self, index):
        if self.mode == 'train':
            (pid, img_path, caption) = self.data[index]     
            # load image
            image = default_loader(os.path.join(self.root_dir, 'imgs', img_path))
            if self.transform is not None:
                image = self.transform(image)

            return {
                'pid': pid,
                'image': image,
                'caption': caption
            }
        elif self.mode == 'text':
            caption = self.captions[index]
            caption_pid = self.caption_pids[index]
            return caption_pid, caption
        elif self.mode == 'image':
            image = default_loader(os.path.join(self.root_dir, 'imgs', self.images[index]))
            if self.transform is not None:
                image = self.transform(image)
            return self.image_ids[index], image



class TextImageDatasetMLM(Dataset):
    def __init__(self, data_cfg, json_path, transform=None, aug_transform = None,
                 text_augment=True, split="train", mode='train', multiview=True) -> None:
        super().__init__()
        self.name = data_cfg.NAME
        self.data_cfg = data_cfg.clone()
        self.root_dir = os.path.join(data_cfg.DATA_DIR, self.name)
        self.json_path = os.path.join(self.root_dir, json_path)
        self.transform = transform
        self.aug_transform = aug_transform
        self.mode = mode
        self.text_augment = text_augment
        self.multiview = multiview
        if mode not in ['train', 'text', 'image']:
            raise ValueError('Invalid mode. Must be one of [train, text, image]')

        with open(self.json_path, 'r') as f:
            self.data = json.load(f)

        self.split = split
        self.max_len = self.data_cfg.TEXT_MAX_LEN if self.data_cfg.TEXT_MAX_LEN else 77
        self.truncate = self.data_cfg.TEXT_TRUNCATE if self.data_cfg.TEXT_TRUNCATE else True
        self.tokenizer = SimpleTokenizer(bpe_path=self.data_cfg.BPE_PATH)

        tmp_data = []
        if data_cfg.SAMPLE != -1:
            num_samples = data_cfg.SAMPLE
        else:
            num_samples = len(self.data)
            
        for sample in self.data[:num_samples]:
            if sample['split'] == self.split:
                for i, cap in enumerate(sample['captions'][:2]):
                    tmp_data.append((sample['id'] - 1 if data_cfg.NAME == "CUHK-PEDES" else sample['id'], 
                                     sample['file_path' if data_cfg.NAME == "CUHK-PEDES" else "img_path"], 
                                     cap, 
                                     sample['noun_chunks'][i] if data_cfg.CHUNKED_MLM else None, 
                                     sample[f'aug_cap_{i+1}'] if data_cfg.BACK_TRANSLATE else None))
                
        self.data = tmp_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.mode == 'train':
            pid, img_path, caption = self.data[index][:3]   

            # load image
            ori_image = default_loader(os.path.join(self.root_dir, 'imgs', img_path))
            #aug_image = cv2.imread(os.path.join(self.root_dir, 'imgs', img_path))
            aug_image = default_loader(os.path.join(self.root_dir, 'imgs', img_path))
            
            if self.transform is not None:
                image = self.transform(ori_image)

            if self.aug_transform is not None:
                aug_image = self.aug_transform(aug_image)

                #aug_image = self.aug_transform(image=aug_image)
                #aug_image = aug_image['image']
            
            if self.text_augment:
                if self.data_cfg.BACK_TRANSLATE:
                    aug_caption = self.data[index][4]
                else:
                    aug_captions = eda(caption, num_aug=8)
                    aug_captions.append(self.data[index][4])
                    aug_caption = aug_captions[random.randint(0, 8)]
                    #aug_caption = eda(caption, alpha_sr = 0, alpha_ri=0, alpha_rs=0, p_rd=0, p_bt=1., num_aug=1)[0]
                    #back_translation_aug = naw.BackTranslationAug(
                    #    from_model_name='facebook/wmt19-en-de', 
                    #    to_model_name='facebook/wmt19-de-en'
                    #)
                    #aug_caption = back_translation_aug.augment(caption)                
                aug_caption = tokenize(aug_caption, self.tokenizer, self.max_len, self.truncate)

            caption = tokenize(caption, self.tokenizer, self.max_len, self.truncate)
            noun_chunks = None
            if self.data_cfg.CHUNKED_MLM:
                noun_chunks = self.data[index][3]
            mlm_tokens, mlm_labels = self._build_masked_tokens_and_labels(caption.cpu().numpy().copy(), noun_chunks)
            if not self.multiview:
                return {
                    'pid': pid,
                    'image': image,
                    'caption': caption,
                    'mlm': [mlm_tokens, mlm_labels]
                }
            return {
                'pid': pid,
                'images': [image, aug_image],
                'captions': [caption, aug_caption],
                'mlm': [mlm_tokens, mlm_labels]
            }
        elif self.mode == 'text':
            caption = self.captions[index]
            caption_pid = self.caption_pids[index]
            return caption_pid, caption
        elif self.mode == 'image':
            image = default_loader(os.path.join(self.root_dir, 'imgs', self.images[index]))
            if self.transform is not None:
                image = self.transform(image)
            return self.image_ids[index], image
        
    def _build_masked_tokens_and_labels(self, tokens, noun_chunks=None):
        """
        Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
        :param tokens: list of int, tokenized sentence.
        :return: (list of int, list of int), masked tokens and related labels for MLM prediction
        """
        mask = self.tokenizer.encoder["<|mask|>"]
        token_range = list(range(1, len(self.tokenizer.encoder)-3)) # 1 ~ 49405
        
        if noun_chunks is None: 

            labels = []
            for i, token in enumerate(tokens):
                if 0 < token < 49405:
                    prob = random.random()
                    # mask token with 15% probability
                    if prob < 0.15:
                        prob /= 0.15

                        # 80% randomly change token to mask token
                        if prob < 0.8:
                            tokens[i] = mask

                        # 10% randomly change token to random token
                        elif prob < 0.9:
                            tokens[i] = random.choice(token_range)

                        # -> rest 10% randomly keep current token

                        # append current token to output (we will predict these later)
                        labels.append(token)
                    else:
                        # no masking token (will be ignored by loss function later)
                        labels.append(0)
                else:
                    labels.append(0)
            
            if all(l == 0 for l in labels):
                # at least mask 1
                labels[1] = tokens[1]
                tokens[1] = mask

            return torch.tensor(tokens), torch.tensor(labels)

        else:
            labels = [0 for _ in range(len(tokens))]
            for chunk in noun_chunks:
                for i in chunk:
                    if i >= 77:
                        continue
                    token = tokens[i]
                    if 0 < token < 49405:
                        prob = random.random()
                        # mask noun chunk with 50% probability
                        if prob < 0.50:
                            prob /= 0.50

                            # 80% randomly change token to mask token
                            if prob < 0.8:
                                tokens[i] = mask

                            # 10% randomly change token to random token
                            elif prob < 0.9:
                                tokens[i] = random.choice(token_range)

                            # -> rest 10% randomly keep current token

                            # append current token to output (we will predict these later)
                            labels[i] = token
            
            if all(l == 0 for l in labels):
                # at least mask 1
                labels[1] = tokens[1]
                tokens[1] = mask
            return torch.tensor(tokens), torch.tensor(labels)