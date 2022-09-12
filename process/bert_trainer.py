# import torch
# import torch.nn.utils as torch_utils

# from ignite.engine import Events

# from process.utils import get_grad_norm, get_parameter_norm

# VERBOSE_SILENT = 0
# VERBOSE_EPOCH_WISE = 1
# VERBOSE_BATCH_WISE = 2

# from trainer import Trainer, MyEngine


# class EngineForBert(MyEngine):

#     def __init__(self, func, model, crit, optimizer, scheduler, config):
#         self.scheduler = scheduler

#         super().__init__(func, model, crit, optimizer, config)

#     @staticmethod
#     def train(engine, mini_batch):
#         '''
#         feed forward -> loss -> back propa -> gradient descent
#         '''
#         # You have to reset the gradients of all model parameters
#         # before to take another step in gradient descent.
#         engine.model.train() # Because we assign model as class variable, we can easily access to it.
#         engine.optimizer.zero_grad()

#         x, y = mini_batch['input_ids'], mini_batch['labels']
#         x, y = x.to(engine.device), y.to(engine.device)
#         mask = mini_batch['attention_mask']
#         mask = mask.to(engine.device)              # gpu로 옮겨

#         x = x[:, :engine.config.max_length]        # [bs, seq, 1]  or [bs, seq]

#         # Take feed-forward
#         y_hat = engine.model(x, attention_mask=mask).logits   # logit : softmax넣기 직전값...
#         # y_hat : [n, |c|]

#         loss = engine.crit(y_hat, y)  # neglog, or crossEntropyLoss -> -sigma(y*softmax(y_hat))
#         loss.backward()

#         # Calculate accuracy only if 'y' is LongTensor,
#         # which means that 'y' is one-hot representation.
#         if isinstance(y, torch.LongTensor) or isinstance(y, torch.cuda.LongTensor):
#             accuracy = (torch.argmax(y_hat, dim=-1) == y).sum() / float(y.size(0))
#         else:
#             accuracy = 0

#         p_norm = float(get_parameter_norm(engine.model.parameters()))
#         g_norm = float(get_grad_norm(engine.model.parameters()))

#         # Take a step of gradient descent.
#         engine.optimizer.step()   # gradient descent
#         engine.scheduler.step()

#         return {
#             'loss': float(loss),
#             'accuracy': float(accuracy),
#             '|param|': p_norm,
#             '|g_param|': g_norm,
#         }

#     @staticmethod
#     def validate(engine, mini_batch):
#         engine.model.eval()

#         with torch.no_grad():
#             x, y = mini_batch['input_ids'], mini_batch['labels']
#             x, y = x.to(engine.device), y.to(engine.device)
#             mask = mini_batch['attention_mask']
#             mask = mask.to(engine.device)

#             x = x[:, :engine.config.max_length]

#             # Take feed-forward
#             y_hat = engine.model(x, attention_mask=mask).logits

#             loss = engine.crit(y_hat, y)

#             if isinstance(y, torch.LongTensor) or isinstance(y, torch.cuda.LongTensor):
#                 accuracy = (torch.argmax(y_hat, dim=-1) == y).sum() / float(y.size(0))
#             else:
#                 accuracy = 0

#         return {
#             'loss': float(loss),
#             'accuracy': float(accuracy),
#         }


# class BertTrainer(Trainer):

#     def __init__(self, config):
#         self.config = config

#     def train(
#         self,
#         model, crit, optimizer, scheduler,
#         train_loader, valid_loader,
#     ):
#         train_engine = EngineForBert(
#             EngineForBert.train,                            # 매 이터레이션 마다 train 을 부를거야
#             model, crit, optimizer, scheduler, self.config
#         )
#         validation_engine = EngineForBert(
#             EngineForBert.validate,                          # valid도 불러와
#             model, crit, optimizer, scheduler, self.config
#         )

#         EngineForBert.attach(          # traner.py
#             train_engine,
#             validation_engine,
#             verbose=self.config.verbose
#         )

#         def run_validation(engine, validation_engine, valid_loader):
#             validation_engine.run(valid_loader, max_epochs=1)

#         train_engine.add_event_handler(
#             Events.EPOCH_COMPLETED, # epoch 이 끝날때마다.
#             run_validation,         # function
#             validation_engine, valid_loader, # 첫번째 arguments제외한 나머지 argument
#         )
#         validation_engine.add_event_handler(
#             Events.EPOCH_COMPLETED, # event
#             EngineForBert.check_best, # function
#         )

#         train_engine.run(
#             train_loader,
#             max_epochs=self.config.n_epochs,
#         )

#         model.load_state_dict(validation_engine.best_model)

#         return model
