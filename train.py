import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
import random
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from transformers import BertModel, BertTokenizerFast
# from transformers import BertForSequenceClassification, AlbertForSequenceClassification
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

import torch_optimizer as custom_optim

# from process.bert_trainer import BertTrainer as Trainer
from process.bert_dataset import ClassificationDataset, ClassificationCollator
# from process.utils import read_text

from model.multimodel import MultiModalClassifier, BertClassifier
from trainer import BertTrainer as Trainer


def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument('--train_fn', default='./data/train_df_negOne')
    p.add_argument('--pretrained_model_name', type=str, default='kykim/bert-kor-base')
    p.add_argument('--use_albert', action='store_true', default=False)
    p.add_argument('--gpu_id', type=int, default=0)
    p.add_argument('--verbose', type=int, default=2)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--adam_epsilon', type=float, default=1e-8)
    p.add_argument('--use_radam', default = True)                   
    p.add_argument('--valid_ratio', type=float, default=0.000001)
    # sequence token length
    p.add_argument('--max_length', type=int, default=30)
    p.add_argument("--num_b", type=int, default=57)    # 대분류 카테고리수
    p.add_argument("--num_m", type=int, default=552)   # 중분류 카테고리수
    p.add_argument("--num_s", type=int, default=3190)  # 소분류 카테고리수
    p.add_argument("--num_d", type=int, default=404)   # 세분류 카테고리수
    p.add_argument("--img_path", type=str, default='./data/train_img_feat.h5')              # img data filepath
    p.add_argument('--n_epochs', type=int, default=6)                                       # base기준 2번만 돌려도 준수한 성능
    p.add_argument("--logging_fn", default = './saved/multi/full/multi_cross_6block.log')   # log save filepath
    p.add_argument('--model_fn', default='./saved/multi/full/multi_cross_6block.pth')       # model save filepath
    p.add_argument("--modality", type=str, default='both')                                  # modality : text, img, both
    p.add_argument("--scheduler", default=True)                                             # warm up ratio
    # Training Warm Up ratio 
    p.add_argument('--warmup_ratio', type=float, default=.2)
    p.add_argument('--lr', type=float, default=5e-5)
    ## if use multi-modal
    p.add_argument("--load_model_path", default='./saved/text/text.06.-0.93-0.88-.0.68-0.09-.pth') # pre-trained model사용하려면 모델 불러워야함.
    p.add_argument("--n_block", default = 6)
    p.add_argument("--multiModal_type", default = 'cross')                                  # simple(concatenate), else cross attention
    p.add_argument("--num_head", default = 1)

    config = p.parse_args()

    return config


def get_loaders(filepath, tokenizer, config, valid_ratio=.2 ):
    '''
        filepath : train_df path(str)
        tokenizer : bertTokenizer
        
        return : train_Dataloader, valid_Dataloader
    '''
    # Get list of labels and list of texts.
    train_df = pd.read_csv(filepath)
    train_df['img_idx'] = train_df.index

    texts = train_df['product']
    bcateid = train_df['bcateid']
    mcateid = train_df['mcateid']
    scateid = train_df['scateid']
    dcateid = train_df['dcateid']
    labels = list(zip(bcateid, mcateid, scateid, dcateid))
    imgs = train_df['img_idx']

    shuffled = list(zip(texts, labels, imgs))
    random.shuffle(shuffled)
    texts = [i[0] for i in shuffled]
    labels = [i[1] for i in shuffled]
    imgs = [i[2] for i in shuffled]
    idx = int(len(texts) * (1 - valid_ratio))


    # set DataLoader
    train_loader = DataLoader(
        ClassificationDataset(texts[:idx], labels[:idx], imgs[:idx], config.img_path),
        batch_size= config.batch_size,
        shuffle=True,
        collate_fn=ClassificationCollator(tokenizer, config.max_length),
    )

    valid_loader = DataLoader(
        ClassificationDataset(texts[idx:], labels[idx:], imgs[idx:], config.img_path),
        batch_size=config.batch_size,       ##########################################
        collate_fn=ClassificationCollator(tokenizer, config.max_length), #######################
    )

    return train_loader, valid_loader



def get_optimizer(model, config):
    '''
        set optimizer and return optimizer
    '''

    if config.use_radam:
        # if using RAdam
        optimizer = custom_optim.RAdam(model.parameters(), lr=config.lr)

    else:
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']    # no decay on bias and layerNorm
        optimizer_grouped_parameters = [
            # weight decay except no_decay
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
    print('modality : ', config.modality)
    print(config.pretrained_model_name)

    # Get pretrained tokenizer.
    tokenizer = BertTokenizerFast.from_pretrained(config.pretrained_model_name)

    # get dataloaders for both train and valid
    train_loader, valid_loader = get_loaders(
        config.train_fn,
        tokenizer,
        valid_ratio = config.valid_ratio,
        config = config
    )

    # print logs : To check data shapes, and iterations
    mini = next(iter(train_loader))
    print(mini['input_ids'].shape, mini['attention_mask'].shape, mini['labels'].shape, mini['imgs'].shape)
    print(
        '|train| =', len(train_loader) * config.batch_size,
        '|valid| =', len(valid_loader) * config.batch_size,
    )
    n_total_iterations = len(train_loader) * config.n_epochs
    n_warmup_steps = int(n_total_iterations * config.warmup_ratio)
    if config.scheduler:
        print(
            '#total_iters =', n_total_iterations,
            '#warmup_iters =', n_warmup_steps,
        )

    
    # if trainig multi-modality using fine-tuned pretrained model
    if config.load_model_path and config.modality == 'both':
        model = MultiModalClassifier(config=config)                 # load frame of model
        package = torch.load(config.load_model_path)['model']       # load saved weight of model
        print(torch.load(config.load_model_path)['config'])         # check model saved path
        model.load_state_dict(package, strict=False)                # overide weights to model
        print(model)
    else:
        # if not using multimodality or first time of fine-tunning
        model = MultiModalClassifier(config=config).cuda()

    # check model frame
    print(model)

    # set optimizer
    optimizer = get_optimizer(model, config)

    # set loss fn
    crit = nn.CrossEntropyLoss(ignore_index=-2).cuda()
    
    # set warmup scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        n_warmup_steps,
        n_total_iterations
    )

    # put model and lossFn on gpu
    if config.gpu_id >= 0:
        model.cuda()
        crit.cuda()

    # Start train.
    trainer = Trainer(config)
    model = trainer.train(
        model,
        crit,
        optimizer,
        scheduler,
        train_loader,
        valid_loader,
    )

    # save model
    torch.save({
        'rnn': None,
        'cnn': None,
        'bert': model.state_dict(),
        'config': config,
        'vocab': None,
        'tokenizer': tokenizer,
    }, config.model_fn)

if __name__ == '__main__':
    config = define_argparser()
    main(config)