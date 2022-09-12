import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
import random
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch_optimizer as custom_optim
from torchtext import data, datasets

# from process.bert_trainer import BertTrainer as Trainer
from process.bert_dataset import ClassificationDataset, ClassificationCollator
from process.utils import read_text

from model.multimodel import MultiModalClassifier
from trainer import BertTrainer as Trainer


def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument('--model_fn', default='./saved/text/text.pth')
    p.add_argument('--train_fn', default='./data/train_df_negOne')    # 학습에 사용될 파일이름. // train_df1은 label 1씩 뺀것.
    # Recommended model list:
    # - kykim/bert-kor-base         # bs : 80
    # - kykim/albert-kor-base       # bs : 80
    # - beomi/kcbert-base           # bs : 80
    # - beomi/kcbert-large          # bs : 30
    p.add_argument('--pretrained_model_name', type=str, default='beomi/kcbert-base')  # 다운받을 모델명(인터넷기준)
    p.add_argument('--use_albert', action='store_true', default=False)
    
    p.add_argument('--gpu_id', type=int, default=0)
    p.add_argument('--verbose', type=int, default=2)

    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--n_epochs', type=int, default=6)                # base기준 2번만 돌려도 괜춘한 성능을 보임.

    p.add_argument('--lr', type=float, default=5e-5) # 5e-5
    p.add_argument('--warmup_ratio', type=float, default=.2)         # transformer가 학습이 까다로워.. 웜업함. // 위에 두줄은 안건들여도됨.
    p.add_argument('--adam_epsilon', type=float, default=1e-8)
    # If you want to use RAdam, I recommend to use LR=1e-4.
    # Also, you can set warmup_ratio=0.
    p.add_argument('--use_radam', default = True)                   # radam을 쓸때는 warup_ratio를 0으로 해야함. 그리고 추천한 lr = 1e-4이다.
    p.add_argument('--valid_ratio', type=float, default=.2)

    p.add_argument('--max_length', type=int, default=30)
    p.add_argument("--num_b", type=int, default=57)
    p.add_argument("--num_m", type=int, default=552)
    p.add_argument("--num_s", type=int, default=3190)
    p.add_argument("--num_d", type=int, default=404)
    p.add_argument("--modality", type=str, default='text')



    config = p.parse_args()

    return config


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

    TEXT = data.Field(
        sequential=True,
        use_vocab = True,
        batch_first = True,
        include_lengths = True,
        fix_length = config.max_length,
    )

    LABELS = data.Field(
        sequential=False,
        use_vocab = False,
        batch_first = True,
        is_target = True
    )

    IMGS = data.Field(
        sequential = False,
        use_vocab = False,
        batch_first = True
    )

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


def main(config):

    os.environ['PYTHONHASHSEED'] = str(42)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    
    # tokenizer = BertTokenizerFast.from_pretrained(config.pretrained_model_name)    # 적당히 전처리 된 text를 넣으면됨.
    # Get dataloaders using tokenizer from untokenized corpus.
    train_loader, valid_loader = get_loaders(
        config.train_fn,
        tokenizer,
        valid_ratio=config.valid_ratio
    )
    mini = next(iter(train_loader))

    print(mini['input_ids'].shape)

    # print(
    #     '|train| =', len(train_loader) * config.batch_size,
    #     '|valid| =', len(valid_loader) * config.batch_size,
    # )

    # n_total_iterations = len(train_loader) * config.n_epochs
    # n_warmup_steps = int(n_total_iterations * config.warmup_ratio)
    # print(
    #     '#total_iters =', n_total_iterations,
    #     '#warmup_iters =', n_warmup_steps,
    # )

    # #############################################################
    # model = MultiModalClassifier(config=config).cuda()
    # ############## # trainer에 4개 아웃풋 나오는것도 바꿔야함.

    # # model_loader = AlbertForSequenceClassification
    # # model = model_loader.from_pretrained(
    # #     config.pretrained_model_name,
    # #     num_labels=config.num_b             # 맨끝에잇는 <cls>token자리에 - layer를 하나 덧붙여줘.
    # # )
    # print(model)
    # '''
    #  (pooler): Linear(in_features=768, out_features=768, bias=True)
    # (pooler_activation): Tanh()
    #     )
    #     (dropout): Dropout(p=0.1, inplace=False)
    #     (classifier): Linear(in_features=768, out_features=57, bias=True)
    #     )

    # '''
    # ##############################################################


    # optimizer = get_optimizer(model, config)

    # # By default, model returns a hidden representation before softmax func.
    # # Thus, we need to use CrossEntropyLoss, which combines LogSoftmax and NLLLoss.
    # crit = nn.CrossEntropyLoss(ignore_index=-2).cuda()
    # scheduler = get_linear_schedule_with_warmup(
    #     optimizer,
    #     n_warmup_steps,
    #     n_total_iterations
    # )

    # if config.gpu_id >= 0:
    #     model.cuda()
    #     crit.cuda()


    # # Start train.
    # trainer = Trainer(config)
    # model = trainer.train(
    #     model,
    #     crit,
    #     optimizer,
    #     scheduler,
    #     train_loader,
    #     valid_loader,
    # )

    # torch.save({
    #     'rnn': None,
    #     'cnn': None,
    #     'bert': model.state_dict(),
    #     'config': config,
    #     'vocab': None,
    #     # 'classes': index_to_label,
    #     'tokenizer': tokenizer,
    # }, config.model_fn)

if __name__ == '__main__':
    config = define_argparser()

    main(config)

    # name = 'beomi/kcbert-base'
    # tokenizer = BertTokenizerFast.from_pretrained(name)    # 적당히 전처리 된 text를 넣으면됨.

    # train_loader, valid_loader = get_loaders("./data/train_df", tokenizer)
    # minibatch = next(iter(train_loader))
    # input_ids, attention_mask, labels = minibatch['input_ids'], minibatch['attention_mask'], minibatch['labels']

    # print(labels)
    # print(labels[:,0])
    # from easydict import EasyDict as edict
    # config = edict({'pretrained_model_name': 'beomi/kcbert-base'})
    # m = MultiModalClassifier(config=config, modality='text')
    # y = m(input_ids, attention_mask=attention_mask)
    # print(y)
    # model_loader = BertForSequenceClassification
    # model = model_loader.from_pretrained(name, num_labels = 300)
    
    # print(input_ids)
    # print(attention_mask)
    # print(labels)
    # y_hat = model(input_ids, attention_mask= attention_mask).logits
    # print(y_hat.shape)

