##############################################################################################################################
##################################################    run on terminal    #####################################################
##############################################################################################################################
# python inference.py --load_model_path_both ./saved/multi/multi_cross.06.-0.94-0.90-.0.90-0.98-.pth --predict_fn ./predict/dev/multi/cross_predict_6th.csv --logging_fn ./predict/dev/multi/cross_predict_6th.log --multiModal_type cross --config_fn ./predict/dev/multi/cross_predict_6th --n_block 1 --num_head 1
# python inference.py --load_model_path_both ./saved/multi/multi_cross.06.-0.94-0.90-.0.90-0.98-.pth --predict_fn ./predict/dev/multi/cross_predict_6th.csv --logging_fn ./predict/dev/multi/cross_predict_6th.log --multiModal_type cross --config_fn ./predict/dev/multi/cross_predict_6th --n_block 1 --num_head 1
# python inference.py --load_model_path_both ./saved/multi/multi_cross.06.-0.94-0.90-.0.90-0.98-.pth --predict_fn ./predict/dev/multi/cross_predict_6th.csv --logging_fn ./predict/dev/multi/cross_predict_6th.log --multiModal_type cross --config_fn ./predict/dev/multi/cross_predict_6th --n_block 1 --num_head 1
# python inference.py --validation_mode True --modality img --predict_fn ./predict/valid/img/img_predict.csv --logging_fn ./predict/valid/img/img_predict.log --multiModal_type img --config_fn ./predict/valid/img/img_config && 
# python inference.py --validation_mode True --modality text --multiModal_type text --predict_fn ./predict/valid/text/text_predict.csv --logging_fn ./predict/valid/text/text_predict.log --config_fn ./predict/valid/text/text_config && python inference.py --validation_mode True --modality both --multiModal_type simple --predict_fn ./predict/valid/multi/multi_simple_predict.csv --logging_fn ./predict/valid/multi/multi_simple_predict.log --config_fn ./predict/valid/multi/multi_config && python inference.py --validation_mode True --modality both --multiModal_type cross --n_block 2 --num_head 1 --load_model_path_both ./saved/multi/multi_cross.06.-0.94-0.90-.0.90-0.98-.pth --predict_fn ./predict/valid/multi/multi_cross_2block_1head_predict.csv --logging_fn ./predict/valid/multi/multi_cross_2block_1head.log --config_fn ./predict/valid/multi/multi_cross_2block_1head_config && python inference.py --validation_mode True --modality both --multiModal_type cross --n_block 6 --num_head 1 --load_model_path_both ./saved/multi/multi_cross_6block.06.-0.94-0.90-.0.90-0.98-.pth --predict_fn ./predict/valid/multi/multi_cross_6block_1head_predict.csv --logging_fn ./predict/valid/multi/multi_cross_6block_1head.log --config_fn ./predict/valid/multi/multi_cross_6block_1head_config && 
# python inference.py --validation_mode True --modality both --multiModal_type cross --n_block 12 --num_head 1 --load_model_path_both ./saved/multi/multi_cross_12block.06.-0.94-0.90-.0.91-0.98-.pth --predict_fn ./predict/valid/multi/cross_12block_1head_predict.csv --logging_fn ./predict/valid/multi/cross_12block_1head.log --config_fn ./predict/valid/multi/cross_12block_1head_config && python inference.py --validation_mode True --modality both --multiModal_type cross --n_block 6 --num_head 2 --load_model_path_both ./saved/multi/multi_cross_6block_2head.06.-0.94-0.90-.0.91-0.98-.pth --predict_fn ./predict/valid/multi/cross_6block_2head_predict.csv --logging_fn ./predict/valid/multi/cross_6block_2head.log --config_fn ./predict/valid/multi/cross_6block_2head_config && python inference.py --validation_mode True --modality both --multiModal_type cross --n_block 6 --num_head 4 --load_model_path_both ./saved/multi/multi_cross_6block_4head.06.-0.94-0.90-.0.90-0.98-.pth --predict_fn ./predict/valid/multi/cross_6block_4head_predict.csv --logging_fn ./predict/valid/multi/cross_6block_4head.log --config_fn ./predict/valid/multi/cross_6block_4head_config && python inference.py --validation_mode True --modality both --multiModal_type cross --n_block 6 --num_head 6 --load_model_path_both ./saved/multi/multi_cross_6block_6head.06.-0.94-0.90-.0.91-0.97-.pth --predict_fn ./predict/valid/multi/cross_6block_6head_predict.csv --logging_fn ./predict/valid/multi/cross_6block_6head.log --config_fn ./predict/valid/multi/cross_6block_6head_config 
# python inference.py --predict_fn ./predict/dev/multi/simple_6th_predict --logging_fn ./predict/dev/multi/multi_simple_6th_predict.log --n_block 1 --multiModal_type simple --config_fn ./predict/dev/multi/dev_multi_simple_config --load_model_path_both ./saved/multi/multi_simple_concat.06.-0.93-0.89-.0.90-0.97-.pth && 
# python inference.py --predict_fn ./predict/dev/multi/cross_2block_6th_predict.csv --logging_fn ./predict/dev/multi/multi_cross_2block_6th_predict.log --n_block 2 --multiModal_type cross --config_fn ./predict/dev/multi/dev_multi_cross_2block_config --load_model_path_both ./saved/multi/multi_cross.06.-0.94-0.90-.0.90-0.98-.pth && 
# python inference.py --predict_fn ./predict/dev/multi/cross_6block_6th_predict.csv --logging_fn ./predict/dev/multi/multi_cross_6block_6th_predict.log --n_block 6 --multiModal_type cross --config_fn ./predict/dev/multi/dev_multi_cross_6block_config --load_model_path_both ./saved/multi/multi_cross_6block.06.-0.94-0.90-.0.90-0.98-.pth && 
# python inference.py --predict_fn ./predict/dev/multi/cross_12block_6th_predict.csv --logging_fn ./predict/dev/multi/multi_cross_12block_6th_predict.log --n_block 12 --multiModal_type cross --config_fn ./predict/dev/multi/dev_multi_cross_12block_config --load_model_path_both ./saved/multi/multi_cross_12block.06.-0.94-0.90-.0.91-0.98-.pth && 
# python inference.py --predict_fn ./predict/dev/multi/cross_6block_2head_6th_predict.csv --logging_fn ./predict/dev/multi/multi_cross_6block_2head_6th_predict.log --n_block 6 --num_head 2 --multiModal_type cross --config_fn ./predict/dev/multi/dev_multi_cross_6block_2head_config --load_model_path_both ./saved/multi/multi_cross_6block_2head.06.-0.94-0.90-.0.91-0.98-.pth && 
# python inference.py --predict_fn ./predict/dev/multi/cross_6block_4head_6th_predict.csv --logging_fn ./predict/dev/multi/multi_cross_6block_4head_6th_predict.log --n_block 6 --num_head 4 --multiModal_type cross --config_fn ./predict/dev/multi/dev_multi_cross_6block_4head_config --load_model_path_both ./saved/multi/multi_cross_6block_4head.06.-0.94-0.90-.0.90-0.98-.pth && 
# python inference.py --predict_fn ./predict/dev/multi/cross_6block_6head_6th_predict.csv --logging_fn ./predict/dev/multi/multi_cross_6block_6head_6th_predict.log --n_block 6 --num_head 6 --multiModal_type cross --config_fn ./predict/dev/multi/dev_multi_cross_6block_6head_config --load_model_path_both ./saved/multi/multi_cross_6block_6head.06.-0.94-0.90-.0.91-0.97-.pth &&
# python inference.py --predict_fn ./predict/dev/text/text_predict.csv --logging_fn ./predict/dev/text/text_predict.log --modality text --config_fn ./predict/dev/text/text_config &&
# python inference.py --predict_fn ./predict/dev/img/img_predict.csv --logging_fn ./predict/dev/img/img_predict.log --modality img --config_fn ./predict/dev/img/img_config && 
# python inference.py --predict_fn ./predict/dev/multi/cross_6block_6th_predict.csv --logging_fn ./predict/dev/multi/multi_cross_6block_6th_predict.log --n_block 6 --num_head 1 --multiModal_type cross --config_fn ./predict/dev/multi/dev_multi_cross_6block_6head_config --load_model_path_both ./saved/multi/multi_cross_6block.pth

from logging import raiseExceptions
from multiprocessing.sharedctypes import Value
import os
from selectors import EpollSelector
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
from transformers import BertForSequenceClassification, AlbertForSequenceClassification
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

import torch_optimizer as custom_optim

# from process.bert_trainer import BertTrainer as Trainer
from process.bert_dataset1 import ClassificationDataset, ClassificationCollator
from process.utils import read_text

from model.multimodel import MultiModalClassifier, BertClassifier
from trainer_valid import ValidationForBert


def define_argparser():
    p = argparse.ArgumentParser()
    p.add_argument('--train_fn', default='./data/train_df_negOne')    
    p.add_argument('--pretrained_model_name', type=str, default='kykim/bert-kor-base')
    p.add_argument('--use_albert', action='store_true', default=False)
    p.add_argument('--gpu_id', type=int, default=0)
    p.add_argument('--verbose', type=int, default=2)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--warmup_ratio', type=float, default=.2)        
    p.add_argument('--adam_epsilon', type=float, default=1e-8)
    p.add_argument('--use_radam', default = True) 
    p.add_argument('--valid_ratio', type=float, default=.2)
    p.add_argument('--max_length', type=int, default=30)
    p.add_argument("--num_b", type=int, default=57)
    p.add_argument("--num_m", type=int, default=552)
    p.add_argument("--num_s", type=int, default=3190)
    p.add_argument("--num_d", type=int, default=404)
    p.add_argument("--scheduler", default=True)
    p.add_argument('--n_epochs', type=int, default=1)  
    p.add_argument("--load_model_path_text", default='./saved/text/text.06.-0.93-0.88-.0.68-0.09-.pth')                         
    p.add_argument("--load_model_path_img", default = './saved/img/img_ffn.09.-0.77-0.68-.0.66-0.88-.pth')
    ########     validating       ############
    p.add_argument("--test_fn", default='./data/test_df')                              # test_df path
    p.add_argument("--dev_fn", default='./data/dev_df')                                # dev_df path
    p.add_argument("--test_img_path", type=str, default='./data/test_img_feat.h5')     # test img path
    p.add_argument("--img_path", type=str, default='./data/train_img_feat.h5')         # train img path
    
    ##############################################################################################################################
    #########################################################    변경할 것들    #####################################################
    ##############################################################################################################################
    p.add_argument('--validation_mode', type=bool, default=True)                       # True : validation   False : test
    p.add_argument('--dev_mode', type=bool, default=False)                             # True : validation   False: test
    # dev가 true이면    dev_df임.

    #########    저장할 모델의 위치    ##############
    p.add_argument("--modality", type=str, default='text')                             # type of modality. modality : text, img, both
    p.add_argument("--predict_fn", default = './predict/new/text/text_predict.csv')    # locate predict filepath
    p.add_argument("--logging_fn", default = './predict/new/text/text_predict.log')    # locate log filepath
    p.add_argument("--config_fn", type=str, default='./predict/new/text/text_config')  # locate config filepath
    p.add_argument("--load_model_path_both", default = './saved/multi/full/multi_cross_6block.06.-0.93-0.89-.0.90-0.97-.pth')  # it works when modality:both
    p.add_argument("--n_block", default = 6, type=int)
    p.add_argument("--multiModal_type", default = 'cross')
    p.add_argument("--num_head", default = 1, type=int)

    config = p.parse_args()

    return config


def get_loaders(fn, tokenizer, config, valid_ratio=.2 ):
    train_df=pd.read_csv(fn)
    train_df['img_idx'] = train_df.index
    texts=train_df['product']
    bcateid = train_df['bcateid']
    mcateid = train_df['mcateid']
    scateid = train_df['scateid']
    dcateid = train_df['dcateid']
    labels = list(zip(bcateid, mcateid, scateid, dcateid))
    imgs = train_df['img_idx']
    pids = train_df['pid']

    shuffled = list(zip(texts, labels, imgs, pids))
    random.shuffle(shuffled)
    texts = [e[0] for e in shuffled]
    labels = [e[1] for e in shuffled]
    imgs = [e[2] for e in shuffled]
    pids = [e[3] for e in shuffled]
    idx = int(len(texts) * (1 - valid_ratio))

    train_loader = DataLoader(
        ClassificationDataset(texts[:idx], labels[:idx], imgs[:idx], config.img_path, pids[:idx]),
        batch_size= config.batch_size,      ########################################
        shuffle=True,
        collate_fn=ClassificationCollator(tokenizer, config.max_length), ########################
    )

    train_df.iloc[idx:].to_csv("validation.csv", index=False)
    valid_loader = DataLoader(
        ClassificationDataset(texts[idx:], labels[idx:], imgs[idx:], config.img_path, pids[idx:]),
        batch_size=config.batch_size,       ##########################################
        collate_fn=ClassificationCollator(tokenizer, config.max_length), #######################
    )

    return train_loader, valid_loader


def get_loaders_test(fn, tokenizer, config):

    train_df=pd.read_csv(fn)
    print(train_df.head())
    train_df['img_idx'] = train_df.index

    texts=train_df['product']
    bcateid = train_df['bcateid']
    mcateid = train_df['mcateid']
    scateid = train_df['scateid']
    dcateid = train_df['dcateid']
    labels = list(zip(bcateid, mcateid, scateid, dcateid))
    imgs = train_df['img_idx']
    pids = train_df['pid']

    train_loader = DataLoader(
        ClassificationDataset(texts, labels, imgs, config.img_path, pid = pids),
        batch_size= config.batch_size,      ########################################
        collate_fn=ClassificationCollator(tokenizer, config.max_length), ########################
    )

    return train_loader




def get_optimizer(model, config):
    if config.use_radam:
        optimizer = custom_optim.RAdam(model.parameters(), lr=config.lr)
    else:
        no_decay = ['bias', 'LayerNorm.weight']
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
    print('modality : ', config.modality)
    print(config.pretrained_model_name)
    tokenizer = BertTokenizerFast.from_pretrained(config.pretrained_model_name)    

    if config.validation_mode:
        _, valid_loader = get_loaders(
            config.train_fn,
            tokenizer,
            valid_ratio=config.valid_ratio,
            config = config
        )
        mini = next(iter(valid_loader))

    else:
        test_fn = config.dev_fn if config.dev_mode else config.test_fn

        valid_loader = get_loaders_test(
            test_fn,
            tokenizer,
            config=config
        )
        mini = next(iter(valid_loader))


    print(mini['input_ids'].shape, mini['attention_mask'].shape, mini['labels'].shape, mini['imgs'].shape)

    print(
        '|valid| =', len(valid_loader) * config.batch_size,
    )

    n_total_iterations = len(valid_loader) * config.n_epochs
    n_warmup_steps = int(n_total_iterations * config.warmup_ratio)
    if config.scheduler:
        print(
            '#total_iters =', n_total_iterations,
            '#warmup_iters =', n_warmup_steps,
        )

    


    print(config.modality)
    if config.modality=='text':
        model = MultiModalClassifier(config=config)
        package = torch.load(config.load_model_path_text)['model']
        model.load_state_dict(package)

    elif config.modality=='img':
        model = MultiModalClassifier(config=config)
        package = torch.load(config.load_model_path_img)['model']
        model.load_state_dict(package)

    elif config.modality == 'both':
        model = MultiModalClassifier(config=config)
        print(model)
        print(torch.load(config.load_model_path_both).keys())
        package = torch.load(config.load_model_path_both)['model']
        model.load_state_dict(package, strict=False) # overide parital weights by setting strict as False

    else:
        raise ValueError("check config.modality")

    print(model)


    crit = nn.CrossEntropyLoss(ignore_index=-2).cuda()

    if config.gpu_id >= 0:
        model.cuda()
        crit.cuda()

    if config.validation_mode: 
        print('--------------------train_fn-----------------------')
        print(config.train_fn)
    elif config.validation_mode == False and config.dev_mode: 
        print('--------------------dev_fn-----------------------')
        print(config.dev_fn)
    else:
        print('--------------------test_fn-----------------------')
        print(config.test_fn)


    # Start train.
    trainer = ValidationForBert(config)
    model = trainer.valid(
        model,
        valid_loader,
        crit,
    )

    torch.save({
        'config': config,
    }, config.config_fn)


if __name__ == '__main__':
    config = define_argparser()

    main(config)



