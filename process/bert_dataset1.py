import torch
from torch.utils.data import Dataset
import h5py


class ClassificationCollator():

    def __init__(self, tokenizer, max_length, with_text=False):
        '''
            tokenizer : hugging face tokeninzer
            max_length : ㅉㅏ르ㄹ거

        '''
        self.tokenizer = tokenizer      # transformers.PreTrainedTokenizer
        self.max_length = max_length
        self.with_text = with_text

    def __call__(self, samples):
        texts = [s['text'] for s in samples]    # samples = from textDlassificationDataset
        labels = [s['label'] for s in samples]
        imgs = [s['img'] for s in samples]
        pids = [s['pid'] for s in samples]

        # 이 부분을 만들어 줘야겠네
        encoding = self.tokenizer(   # __call__
            texts,
            padding=True,
            truncation=True,      # maximum length로 잘라.
            return_tensors="pt",  # pytorch 타입으로 리턴
            max_length=self.max_length
        )
        ### [bs, seq, 1]   # 왜 1이냐면, next iter해서 하나의 idx만 가지고 온거야.

        return_value = {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'labels': torch.tensor(labels, dtype=torch.long),
            'imgs': torch.cat(imgs, 0),
            'pid': pids
        }
        
        if self.with_text:
            return_value['text'] = texts

        return return_value


class ClassificationDataset(Dataset):
    '''
        text, img, label
        if img None -> return img:None
    '''
    def __init__(self, texts, labels, img, img_path, pid):
        '''
            texts : [bs, seq, hs]
            labels : [bs, 1]
        '''
        self.texts = texts
        self.labels = labels
        self.imgs = img
        self.img_h5_path = img_path
        self.pid = pid

    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]

        with h5py.File(self.img_h5_path, 'r') as img_feats:
            img_feat = img_feats['img_feat'][self.imgs[item]]

        img_feat = torch.FloatTensor(img_feat).reshape(1,-1)

        pid = self.pid[item]

        return {
            'text':text,
            'img': img_feat,
            'label':label,
            'pid':pid,
        }


if __name__ == "__main__":
    from transformers import BertModel, BertTokenizerFast
    import random
    import pandas as pd
    from torch.utils.data import DataLoader
    
    a = [torch.randn([1,3]), torch.randn([1,3])]
    print(torch.cat(a,0).shape)