import torch
from torch.utils.data import Dataset
import h5py


class ClassificationCollator():

    def __init__(self, tokenizer, max_length, with_text=False):
        '''
            tokenizer : hugging face tokeninzer
            max_length : limit sequence length of texts
            with_text : return with original text which means not passing through tokenizer

        '''
        self.tokenizer = tokenizer      # transformers.PreTrainedTokenizer
        self.max_length = max_length
        self.with_text = with_text

    def __call__(self, bs):
        texts = [b['text'] for b in bs]    # bs = from ClassificationDataset
        labels = [b['label'] for b in bs]
        imgs = [b['img'] for b in bs]

        encoding = self.tokenizer(   # __call__
            texts,
            padding=True,
            truncation=True,            
            return_tensors="pt",        # return as pytorch
            max_length=self.max_length  # limit to maximum length
        )
        ### [bs, seq, 1]   # the reason of last shape size 1 is got one data from (ClassificationDataset __getitem__)

        return_value = {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'labels': torch.tensor(labels, dtype=torch.long),
            'imgs': torch.cat(imgs, 0),
        }
        
        # original text
        if self.with_text:
            return_value['text'] = texts

        return return_value


class ClassificationDataset(Dataset):
    '''
        text, img, label
        if img None -> return img:None
    '''
    def __init__(self, texts, labels, img, img_path):
        '''

            texts, labels, img : [ train_sz, 1 ]
            

            texts : [bs, seq, hs]
            labels : [bs, 1]
            img : [bs, 2048]  this tensor from encoded from ResNet50
        '''
        self.texts = texts
        self.labels = labels
        self.imgs = img
        self.img_h5_path = img_path

    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]

        with h5py.File(self.img_h5_path, 'r') as img_feats:
            img_feat = img_feats['img_feat'][self.imgs[item]]

        img_feat = torch.FloatTensor(img_feat).reshape(1,-1)

        return {
            'text':text,
            'img': img_feat,
            'label':label
        }


if __name__ == "__main__":
    from transformers import BertModel, BertTokenizerFast
    import random
    import pandas as pd
    from torch.utils.data import DataLoader
    
    a = [torch.randn([1,3]), torch.randn([1,3])]
    print(torch.cat(a,0).shape)