import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch import optim
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler

from process.utils import get_grad_norm, get_parameter_norm
from ignite.engine import Events

VERBOSE_SILENT = 0
VERBOSE_EPOCH_WISE = 1
VERBOSE_BATCH_WISE = 2

from model.trainer import Trainer, MyEngine
import logging


class ValidationForBert():

    def __init__(self, config):
        self.config = config

    def valid(self, model, dataloader, crit):
        model.eval()
        device = next(model.parameters()).device
        print(f'model is on {device}')


        with torch.no_grad():
            b = []
            m = []
            s = []
            d = []
            pids = []

            for mini_batch in dataloader:
                x, y, img = mini_batch['input_ids'], mini_batch['labels'], mini_batch['imgs']
                mask = mini_batch['attention_mask']
                mask = mask.to(device)              # gpu로 옮겨             
                pid = mini_batch['pid']   

                if self.config.modality == "text":
                    x, y, img = x.to(device), y.to(device), None
                    x = x[:, :self.config.max_length]        # [bs, seq, 1]  or [bs, seq]
                    b_hat, m_hat, s_hat, d_hat = model(text= x, attention_mask=mask)   # logit : softmax넣기 직전값...

                elif self.config.modality == 'img':
                    x, y, img = None, y.to(device), img.to(device)
                    b_hat, m_hat, s_hat, d_hat = model(img = img)   # logit : softmax넣기 직전값...

                elif self.config.modality == 'both':
                    x, y, img = x.to(device), y.to(device), img.to(device)
                    x = x[:, :self.config.max_length]        # [bs, seq, 1]  or [bs, seq]
                    b_hat, m_hat, s_hat, d_hat = model(text=x, attention_mask=mask, img = img)   # logit : softmax넣기 직전값...


                with autocast():
                    if self.config.validation_mode:
                        # Take feed-forward
                        b_loss = crit(b_hat, y[:,0].long())
                        m_loss = crit(m_hat, y[:,1].long())
                        s_loss = crit(s_hat, y[:,2].long())
                        d_loss = crit(d_hat, y[:,3].long())
                        loss = (b_loss + 1.2*m_loss + 1.3*s_loss + 1.4*d_loss)/(1+1.2+1.3+1.4)
                    else:
                        loss = torch.tensor([0])

                if isinstance(y, torch.LongTensor) or isinstance(y, torch.cuda.LongTensor):
                    accuracy1 = (torch.argmax(b_hat, dim=-1) == y[:,0]).sum() / (y[:,0]>=-1).sum().item()
                    accuracy2 = (torch.argmax(m_hat, dim=-1) == y[:,1]).sum() / (y[:,1]>=-1).sum().item()
                    accuracy3 = (torch.argmax(s_hat, dim=-1) == y[:,2]).sum() / ((y[:,2]>=-1).sum().item()+1e-06)
                    accuracy4 = (torch.argmax(d_hat, dim=-1) == y[:,3]).sum() / ((y[:,3]>=-1).sum().item()+1e-06)
                else:
                    accuracy1,accuracy2, accuracy3, accuracy4 = 0,0,0,0

                result = {'loss' : loss.tolist(), 
                          'accuracy_b':accuracy1.tolist(), 
                          'accuracy_m':accuracy2.tolist(),
                          'accuracy_s':accuracy3.tolist(), 
                          'accuracy_d':accuracy4.tolist()}

                
                self.record_log(result, self.config)

                b += torch.argmax(b_hat, dim=-1).tolist()
                m += torch.argmax(m_hat, dim=-1).tolist()
                s += torch.argmax(s_hat, dim=-1).tolist()
                d += torch.argmax(d_hat, dim=-1).tolist()
                pids += pid

            print(f'pid : {len(pids)}, b : {len(b)}, m : {len(m)}, s:{len(s)}, d:{len(d)}')
            tmp = pd.DataFrame({'pid':pids, 'b':b,'m':m,'s':s,'d':d})
            tmp.to_csv(self.config.predict_fn, index = False)



    
    def record_log(self, result, config):
        # result : dict type
        logging.basicConfig(filename=config.logging_fn, level=logging.INFO)

        avg_train_loss = result['loss']
        acc_b = result['accuracy_b']
        acc_m = result['accuracy_m']
        acc_s = result['accuracy_s']
        acc_d = result['accuracy_d']       

        logging.info({"loss":avg_train_loss,
                    "acc_b":acc_b,
                    "acc_m":acc_m,
                    "acc_s":acc_s,
                    "acc_d":acc_d})



