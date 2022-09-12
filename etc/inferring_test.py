# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
import random
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler


from transformers import BertModel, BertTokenizerFast
from transformers import BertForSequenceClassification, AlbertForSequenceClassification
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

import torch_optimizer as custom_optim

# from process.bert_trainer import BertTrainer as Trainer
from process.bert_dataset import ClassificationDataset, ClassificationCollator
from process.utils import read_text

from model.multimodel import MultiModalClassifier
from trainer import BertTrainer as Trainer

def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument('--model_fn', default='./saved/text.pth')
    p.add_argument('--train_fn', default='./data/train_df1')
    p.add_argument('--pretrained_model_name', type=str, default='beomi/kcbert-base')
    p.add_argument('--use_albert', action='store_true', default=False)
    p.add_argument('--gpu_id', type=int, default=-1)
    p.add_argument('--verbose', type=int, default=2)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--n_epochs', type=int, default=50)                
    p.add_argument('--lr', type=float, default=5e-5)
    p.add_argument('--warmup_ratio', type=float, default=.2)       
    p.add_argument('--adam_epsilon', type=float, default=1e-8)
    p.add_argument('--use_radam', action='store_true')                 
    p.add_argument('--valid_ratio', type=float, default=.2)
    p.add_argument('--max_length', type=int, default=100)
    p.add_argument("--num_b", type=int, default=57)
    p.add_argument("--num_m", type=int, default=552)
    p.add_argument("--num_s", type=int, default=3190)
    p.add_argument("--num_d", type=int, default=404)
    p.add_argument("--modality", type=str, default='text')

    config = p.parse_args()

    return config

config = define_argparser()


def get_loaders(fn, tokenizer, img = None, valid_ratio=.2):
    '''
        fn : train_df path
        tokenizer : bertTokenizer
        img : img path
        
    '''
    # Get list of labels and list of texts.
    train_df=pd.read_csv(fn)
    
    texts=train_df['product']
    bcateid = train_df['bcateid']
    mcateid = train_df['mcateid']
    scateid = train_df['scateid']
    dcateid = train_df['dcateid']
    labels = list(zip(bcateid, mcateid, scateid, dcateid))
    # img정보 더 붙여야함.
    if img == None:
        imgs = [img]*len(texts)
    else:
        pass ########################## 여기 채워야함.


    shuffled = list(zip(texts, labels, imgs))  # 묶은다음 셔플링.
    random.shuffle(shuffled)
    texts = [e[0] for e in shuffled]
    labels = [e[1] for e in shuffled]
    imgs = [e[2] for e in shuffled]
    idx = int(len(texts) * (1 - valid_ratio))

    # Get dataloaders using given tokenizer as collate_fn.
    train_loader = DataLoader(
        ClassificationDataset(texts[:idx], labels[:idx], imgs[:idx]),
        batch_size= config.batch_size,      ########################################
        shuffle=True,
        collate_fn=ClassificationCollator(tokenizer, config.max_length), ########################
    )

    valid_loader = DataLoader(
        ClassificationDataset(texts[idx:], labels[idx:], imgs[idx:]),
        batch_size=config.batch_size,       ##########################################
        collate_fn=ClassificationCollator(tokenizer, config.max_length), #######################
    )

    return train_loader, valid_loader

def get_optimizer(model, config):
    if config.use_radam:
        optimizer = custom_optim.RAdam(model.parameters(), lr=config.lr)
    else:
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']    # 애들은 no decay한데.
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': 0.01
            },
            {
                'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]

        optimizer = optim.AdamW(
            optimizer_grouped_parameters,
            lr=config.lr,
            eps=config.adam_epsilon
        )

    return optimizer


load = torch.load("./saved/text/text.02.-0.10-0.02-.0.01-0.00-.pth")
import easydict

savedConfig = easydict.EasyDict(vars(load['config']))
print(savedConfig)

tokenizer = BertTokenizerFast.from_pretrained(config.pretrained_model_name)    # 적당히 전처리 된 text를 넣으면됨.
# Get dataloaders using tokenizer from untokenized corpus.
train_loader, valid_loader = get_loaders(
    config.train_fn,
    tokenizer,
    valid_ratio=config.valid_ratio
)

mini = next(iter(train_loader))
print(mini.keys())
text = mini['input_ids'].to('cuda')
attention_mask = mini['attention_mask'].to('cuda')

model = MultiModalClassifier(config=savedConfig).cuda()
model.load_state_dict(load['model'])

optimizer = get_optimizer(model, savedConfig)
optimizer.load_state_dict(load['opt'])

with autocast():
    y = model(text=text, attention_mask=attention_mask)
    print(y)