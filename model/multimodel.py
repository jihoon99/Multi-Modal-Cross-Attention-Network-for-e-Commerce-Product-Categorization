import torch
import torch.nn as nn

from transformers import BertModel, BertTokenizerFast
from transformers import BertForSequenceClassification, AlbertForSequenceClassification
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

from easydict import EasyDict as edict


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, Q, K, V, dk = 128):        # dk 128 means 4heads when Q,K,V's shape as 512
        # K=V. It does not depend on number of modalities
        # but for example if K has n modalities, shape of K will be [bs, n, 128] 

        w = torch.bmm(Q,K.transpose(1,2))        # Q : [bs, 1, 128]  // K : [bs, 2, 128] // => w : [bs, 1, 2] 
        w = self.softmax(w/(dk**.5))
        c = torch.bmm(w, V)                      # C : [bs, 1, 128]
        return c


class MultiHead(nn.Module):
    def __init__(self, hidden_size, n_splits=4):  # 512 / 4 = 128
        '''
            hidden_size : last tensor shape
            n_splits : number of heads

            when 'query' is text hidden_size will be 768 since it shape as [bs, seq, 768]

        '''
        super().__init__()
        
        self.hidden_size = hidden_size
        self.n_splits = n_splits

        # projection
        self.Q_linear = nn.Linear(hidden_size, hidden_size, bias=False)
        self.K_linear = nn.Linear(hidden_size, hidden_size, bias=False)
        self.V_linear = nn.Linear(hidden_size, hidden_size, bias=False)

        self.attn = Attention()

        self.linear = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, Q, K, V, mask=None):
        QWs = self.Q_linear(Q).split(self.hidden_size//self.n_splits, dim=-1)    # [bs, seq, 768] -> ([bs, seq, 192], [bs,seq,192] , ..)
        KWs = self.K_linear(K).split(self.hidden_size//self.n_splits, dim=-1)
        VWs = self.V_linear(V).split(self.hidden_size//self.n_splits, dim=-1)

        QWs = torch.cat(QWs, dim=0)  # ([bs, seq, 192])*4 -> [4bs, seq, 192]
        KWs = torch.cat(KWs, dim=0)
        VWs = torch.cat(VWs, dim=0)
        
        c = self.attn(
            QWs, KWs, VWs,
            mask=mask,
            dk=self.hidden_size//self.n_splits,
        )

        c = c.split(Q.size(0), dim=0)         # [4bs, seq, 192] => ([bs, seq, 192])*4
        c = self.linear(torch.cat(c, dim=-1)) # [bs, seq, 768]
        return c


class CrossModality(nn.Module):
    '''

        Should not change shape in here since it can't be architected as deeply as possible

    '''
    def __init__(self, hidden_size, n_splits, dropout_p=.1):
        super().__init__()
        
        # self.masked_attn = MultiHead(hidden_size, n_splits)
        # self.masked_attn_norm = nn.LayerNorm(hidden_size)
        # self.masked_attn_dropout = nn.Dropout(dropout_p)

        self.attn = MultiHead(hidden_size, n_splits)
        self.attn_norm = nn.LayerNorm(hidden_size)
        self.attn_dropout = nn.Dropout(dropout_p)

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size*2),
            nn.LeakyReLU(),
            nn.Linear(hidden_size*2, hidden_size)
        )

        self.fc_norm = nn.LayerNorm(hidden_size)
        self.fc_dropout = nn.Dropout(dropout_p)

    def forward(self, x, key_and_value, mask=None):
        '''
            x : [bs, 2, 768]
            key_and_value : [bs, 2, 768]
        '''
        normed_key_and_value = self.attn_norm(key_and_value)     # first layer norm which is not the same as original paper of transformer
        z = x + self.attn_dropout(
            self.attn_dropout(
                self.attn(
                    Q = self.attn_norm(x),
                    K = normed_key_and_value,
                    V = normed_key_and_value,
                    mask = mask
        )))

        z = z + self.fc_dropout(self.fc(self.fc_norm(z)))
        return z, key_and_value, mask


class MySequential(nn.Sequential):
    '''
    nn.Sequential만 상속 받아서 forward만 갈아 엎음.
    *x같은 경우 x,key_and_value,mask,prev,future_mask같은 tuple이 들어갈거야.
    '''

    def forward(self, *x):
        for module in self._modules.values():
            x = module(*x)

        return x





class BertClassifier(nn.Module):
    '''
        BertEncoder without head part

        last_hidden_state : [bs, sq, hs] of last encoder
        pooler_output : [bs, hs] of first token of last encoder

        https://huggingface.co/transformers/v1.2.0/_modules/pytorch_transformers/modeling_bert.html#BertModel
    '''
    
    def __init__(self, config):
        super().__init__()
        # Set model
        model_loader = BertModel 
        self.model = model_loader.from_pretrained(config.pretrained_model_name)
        # make last tensor between [-1,1]
        self.tanh = nn.Tanh()
        self.config = config

    def forward(self, x, attention_mask):
        
        y_hat = self.tanh(self.model(x, attention_mask=attention_mask).pooler_output)
        return y_hat




class MultiModalClassifier(nn.Module):
    
    def __init__(self, config, backbone = None):
        super().__init__()
        print('model : ', config.modality)
        
        self.config = config

        # training with only text
        if config.modality == 'text':
            self.BertModel = BertClassifier(config)

            def add_cls(target_size, dropout=0.1):
                '''
                    target_size : number of targets
                    add head : the classifier 
                '''
                return nn.Sequential(
                        nn.LayerNorm(768),
                        nn.Dropout(dropout),
                        nn.Linear(768, 
                        target_size)
                        )

            self.b = add_cls(config.num_b)
            self.m = add_cls(config.num_m)
            self.s = add_cls(config.num_s)
            self.d = add_cls(config.num_d)

        # trainig with only image
        # shape of image data are [bs, 2048] since it passed Pre-trained ResNet50
        elif config.modality == 'img':
            def add_cls(target_size, dropout=0.1):

                return nn.Sequential(
                                    nn.LayerNorm(2048),
                                    nn.Dropout(dropout),
                                    nn.Linear(2048, 1024),
                                    nn.GELU(),
                                    nn.LayerNorm(1024),
                                    nn.Dropout(dropout),
                                    nn.Linear(1024, 512),
                                    nn.GELU(),
                                    nn.LayerNorm(512),
                                    nn.Linear(512, target_size))

            self.b = add_cls(config.num_b)
            self.m = add_cls(config.num_m)
            self.s = add_cls(config.num_s)
            self.d = add_cls(config.num_d)


        # taining with multi-modality
        elif config.modality == 'both':
            def add_cls(target_size, a=1, dropout=0.1):
                '''
                    a : number of modality
                '''
                return nn.Sequential(
                        nn.LayerNorm(768*a),
                        nn.Dropout(dropout),
                        nn.Linear(768*a, target_size))

            # text encoder
            self.BertModel = BertClassifier(config)

            # transform img dimension to 786 which is same shape as text
            self.img_ = nn.Linear(2048, 768)

            # set classifier
            self.bb = add_cls(config.num_b,2)
            self.mm = add_cls(config.num_m,2)
            self.ss = add_cls(config.num_s,2)
            self.dd = add_cls(config.num_d,2)

            # cross modal text to image referring
            self.text_img = MySequential(
                *[CrossModality(768, config.num_head)
                for _ in range(config.n_block)])

            # cross modal image to text referring
            self.img_text = MySequential(
                *[CrossModality(768, config.num_head) 
                for _ in range(config.n_block)])

            self.common_dense = nn.Linear(768*2, 768)
            self.final_dense = nn.Linear(768*2, 768)


    def forward(self, text=None, attention_mask=None, img=None):
        # Training if with only text
        if self.config.modality == 'text':
            y_hat = self.BertModel(text, attention_mask)  # [bs, 768]
            b_hat = self.b(y_hat)
            m_hat = self.m(y_hat)
            s_hat = self.s(y_hat)
            d_hat = self.d(y_hat)
            return b_hat, m_hat, s_hat, d_hat

        # Training if with only img
        elif self.config.modality == 'img':
            b_hat = self.b(img)
            m_hat = self.m(img)
            s_hat = self.s(img)
            d_hat = self.d(img)
            return b_hat, m_hat, s_hat, d_hat


        # Training with multi modality
        elif self.config.modality == 'both':
            # pre-trained part is not going to be trained since the purpose of research
            with torch.no_grad():
                y_hat = self.BertModel(text, attention_mask)  # passed linear transform

            if self.config.multiModal_type == 'simple':
                img_ = self.img_(img)
                y_hat = torch.cat([y_hat,img_], axis = -1)

                b_hat = self.bb(y_hat)
                m_hat = self.mm(y_hat)
                s_hat = self.ss(y_hat)
                d_hat = self.dd(y_hat)        

            else:
                text_encoder = y_hat.unsqueeze(1)
                img_encoder = self.img_(img).unsqueeze(1)

                common_data = torch.cat([text_encoder, img_encoder], axis=1)      # bs, 2, dim

                ############## 0502 #########################

                text_base,_,_ = self.text_img(text_encoder, common_data)
                img_base,_,_ = self.img_text(img_encoder, common_data)
                text_base = text_base.squeeze(1)    # bs, 768
                img_base = img_base.squeeze(1)      # bs, 768

                common_data = self.common_dense(torch.cat([text_base, img_base], axis=-1))
                y_hat = torch.cat([common_data, text_encoder.squeeze(1)], axis = -1)
                # y_hat = self.final_dense(final_data)

                b_hat = self.bb(y_hat)
                m_hat = self.mm(y_hat)
                s_hat = self.ss(y_hat)
                d_hat = self.dd(y_hat)

            return b_hat, m_hat, s_hat, d_hat


if __name__ == "__main__":

    text_img = MySequential(
        *[CrossModality(512, 4) 
        for _ in range(2)])

    config = edict({'modality': 'both'})
    # print(config.pretrained_model_name)
    text_encoder = torch.randn([128,512])
    img_encoder = torch.randn([128,512])
    text_encoder = text_encoder.unsqueeze(1)
    img_encoder = img_encoder.unsqueeze(1)

    common_data = torch.cat([text_encoder, img_encoder], axis = 1)
    text_base,_,_ = text_img(text_encoder, common_data)


    # model_loader = BertModel
    # model = model_loader.from_pretrained(config.pretrained_model_name)
    # m = MultiModalClassifier(modality = 'text', config=config)


    # x = m(text = x)
    # print(x)