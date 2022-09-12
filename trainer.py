import numpy as np
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



class EngineForBert(MyEngine):

    def __init__(self, func, model, crit, optimizer, scheduler, config):
        self.scheduler = scheduler
        self.scaler = GradScaler()
        self.config = config

        super().__init__(func, model, crit, optimizer, config)

    @staticmethod
    def train(engine, mini_batch):
        '''
        feed forward -> loss -> back propa -> gradient descent
        '''

        engine.model.train()
        engine.optimizer.zero_grad()

        x, y, img = mini_batch['input_ids'], mini_batch['labels'], mini_batch['imgs']
        mask = mini_batch['attention_mask']
        mask = mask.to(engine.device)

        # Train with only text
        if engine.config.modality == "text":
            x, y, img = x.to(engine.device), y.to(engine.device), None
            x = x[:, :engine.config.max_length]                                      # [bs, seq, 1]  or [bs, seq]
            b_hat, m_hat, s_hat, d_hat = engine.model(text= x, attention_mask=mask)  # logit : softmax넣기 직전값

        # Train with only img
        elif engine.config.modality == 'img':
            x, y, img = None, y.to(engine.device), img.to(engine.device)
            b_hat, m_hat, s_hat, d_hat = engine.model(img = img)

        # Train with both modality
        elif engine.config.modality == 'both':
            x, y, img = x.to(engine.device), y.to(engine.device), img.to(engine.device)
            x = x[:, :engine.config.max_length]                                                  # [bs, seq, 1]  or [bs, seq]
            b_hat, m_hat, s_hat, d_hat = engine.model(text=x, attention_mask=mask, img = img)

        #################### autocast #################
        # autocast won't work with unkown reason
        # with autocast():
        # Take feed-forward

        ################### BerClassifier ###################
        # b_hat = engine.model(x, attention_mask=mask).logits
        # m_hat = torch.randn(b_hat.shape).to('cuda')
        # s_hat = torch.randn(b_hat.shape).to('cuda')
        # d_hat = torch.randn(b_hat.shape).to('cuda')
        # y_hat : [n, |c|]
        ####################################################################

        # calculate loss
        b_loss = engine.crit(b_hat, y[:,0].long())
        m_loss = engine.crit(m_hat, y[:,1].long())
        s_loss = engine.crit(s_hat, y[:,2].long())
        d_loss = engine.crit(d_hat, y[:,3].long())

        loss = (b_loss + 1.2*m_loss + 1.3*s_loss + 1.4*d_loss)

        # autocast won't work
        # engine.scaler.scale(loss).backward()  # loss.backward()
        
        # calculate gradient
        loss.backward()


        if isinstance(y, torch.LongTensor) or isinstance(y, torch.cuda.LongTensor):
            accuracy1 = (torch.argmax(b_hat, dim=-1) == y[:,0]).sum() / (y[:,0]>=0).sum().item()
            accuracy2 = (torch.argmax(m_hat, dim=-1) == y[:,1]).sum() / (y[:,1]>=0).sum().item()
            accuracy3 = (torch.argmax(s_hat, dim=-1) == y[:,2]).sum() / ((y[:,2]>=0).sum().item()+1e-06)
            accuracy4 = (torch.argmax(d_hat, dim=-1) == y[:,3]).sum() / ((y[:,3]>=0).sum().item()+1e-06)

        else:
            accuracy1,accuracy2, accuracy3, accuracy4 = 0,0,0,0

        p_norm = float(get_parameter_norm(engine.model.parameters()))
        g_norm = float(get_grad_norm(engine.model.parameters()))

        # Take a step of gradient descent.
        engine.optimizer.step()
        if engine.config.scheduler:
            engine.scheduler.step()

        return {
            'loss': float(loss),
            'accuracy_b': float(accuracy1),
            'accuracy_m': float(accuracy2),
            'accuracy_s': float(accuracy3),
            'accuracy_d': float(accuracy4),
            '|param|': p_norm,
            '|g_param|': g_norm,
        }

    @staticmethod
    def validate(engine, mini_batch):
        engine.model.eval()

        with torch.no_grad():
            x, y, img = mini_batch['input_ids'], mini_batch['labels'], mini_batch['imgs']
            mask = mini_batch['attention_mask']
            mask = mask.to(engine.device)
            if engine.config.modality == "text":
                x, y, img = x.to(engine.device), y.to(engine.device), None
                x = x[:, :engine.config.max_length]                                      # [bs, seq, 1]  or [bs, seq]
                b_hat, m_hat, s_hat, d_hat = engine.model(text= x, attention_mask=mask)  

            elif engine.config.modality == 'img':
                x, y, img = None, y.to(engine.device), img.to(engine.device)
                b_hat, m_hat, s_hat, d_hat = engine.model(img = img)   

            elif engine.config.modality == 'both':
                x, y, img = x.to(engine.device), y.to(engine.device), img.to(engine.device)
                x = x[:, :engine.config.max_length]                                      # [bs, seq, 1]  or [bs, seq]
                b_hat, m_hat, s_hat, d_hat = engine.model(text=x, attention_mask=mask, img = img)   


            with autocast():
                # Take feed-forward
                b_loss = engine.crit(b_hat, y[:,0].long())
                m_loss = engine.crit(m_hat, y[:,1].long())
                s_loss = engine.crit(s_hat, y[:,2].long())
                d_loss = engine.crit(d_hat, y[:,3].long())
                loss = b_loss
                        # + 1.2*m_loss + 1.3*s_loss + 1.4*d_loss)/(1+1.2+1.3+1.4)

            if isinstance(y, torch.LongTensor) or isinstance(y, torch.cuda.LongTensor):
                accuracy1 = (torch.argmax(b_hat, dim=-1) == y[:,0]).sum() / (y[:,0]>=0).sum().item()
                accuracy2 =(torch.argmax(m_hat, dim=-1) == y[:,1]).sum() / (y[:,1]>=0).sum().item()
                accuracy3 =(torch.argmax(s_hat, dim=-1) == y[:,2]).sum() / ((y[:,2]>=0).sum().item()+1e-06)
                accuracy4 =(torch.argmax(d_hat, dim=-1) == y[:,3]).sum() / ((y[:,3]>=0).sum().item()+1e-06)
            else:
                accuracy1,accuracy2, accuracy3, accuracy4 = 0,0,0,0


        return {
            'loss': float(loss),
            'accuracy_b': float(accuracy1),
            'accuracy_m': float(accuracy2),
            'accuracy_s': float(accuracy3),
            'accuracy_d': float(accuracy4),
        }

    @staticmethod
    def record_log(engine, train_engine, config):
        '''
            for recording logs
        '''
        logging.basicConfig(filename=config.logging_fn, level=logging.INFO)

        avg_train_loss = train_engine.state.metrics['loss']
        acc_b = train_engine.state.metrics['accuracy_b']
        acc_m = train_engine.state.metrics['accuracy_m']
        acc_s = train_engine.state.metrics['accuracy_s']
        acc_d = train_engine.state.metrics['accuracy_d']       

        logging.info({"loss":avg_train_loss,
                    "acc_b":acc_b,
                    "acc_m":acc_m,
                    "acc_s":acc_s,
                    "acc_d":acc_d})

    @staticmethod
    def save_model(engine, train_engine, config):
        avg_train_loss = train_engine.state.metrics['loss']
        acc_b = train_engine.state.metrics['accuracy_b']
        acc_m = train_engine.state.metrics['accuracy_m']
        acc_s = train_engine.state.metrics['accuracy_s']
        acc_d = train_engine.state.metrics['accuracy_d']


        model_fn = config.model_fn.split(".")
        model_fn = model_fn[:-1] + [f'{train_engine.state.epoch:02d}',f'-{acc_b:.2f}-{acc_m:.2f}-',f'{acc_s:.2f}-{acc_d:.2f}-'] + [model_fn[-1]]
        model_fn = ".".join(model_fn)

        torch.save(
            {
                'model':engine.model.state_dict(),
                'opt':train_engine.optimizer.state_dict(),
                'config':config,
            }, model_fn
        )

class BertTrainer(Trainer):

    def __init__(self, config):
        self.config = config

    def train(
        self,
        model, crit, optimizer, scheduler,
        train_loader, valid_loader,
    ):
        train_engine = EngineForBert(
            EngineForBert.train,                            # 매 이터레이션 마다 train 을 호출
            model, crit, optimizer, scheduler, self.config
        )
        validation_engine = EngineForBert(
            EngineForBert.validate,                          # valid 호출
            model, crit, optimizer, scheduler, self.config
        )

        EngineForBert.attach(          # traner.py
            train_engine,
            validation_engine,
            verbose=self.config.verbose
        )

        def run_validation(engine, validation_engine, valid_loader):
            validation_engine.run(valid_loader, max_epochs=1)

        train_engine.add_event_handler(
            Events.EPOCH_COMPLETED,
            EngineForBert.save_model,
            train_engine,
            self.config,
        )

        train_engine.add_event_handler(
            Events.EPOCH_COMPLETED(every=3),   # epoch 이 끝날때마다.
            run_validation,                    # validation function을 실행
            validation_engine, valid_loader,   # 첫번째 arguments제외한 나머지 argument
        )
        validation_engine.add_event_handler(
            Events.EPOCH_COMPLETED,   # event
            EngineForBert.check_best, # function
        )

        train_engine.add_event_handler(
            Events.ITERATION_COMPLETED,
            EngineForBert.record_log,
            train_engine,
            self.config
        )

        train_engine.run(
            train_loader,
            max_epochs=self.config.n_epochs,
        )

        model.load_state_dict(validation_engine.best_model)

        return model


