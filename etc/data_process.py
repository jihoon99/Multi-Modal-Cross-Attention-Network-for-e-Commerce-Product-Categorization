from importlib.resources import path
import os
import json
from webbrowser import get
import h5py
import numpy as np
import pandas as pd

RAW_DATA_DIR = '../dataset'
PROCESSED_DATA_DIR = './data/new'
VOCAB_DIR = os.path.join(PROCESSED_DATA_DIR, 'vocab')

train_file_list = [
    'train.chunk.01',
    'train.chunk.02',
    'train.chunk.03',
    'train.chunk.04',
    'train.chunk.05',
    'train.chunk.06',
    'train.chunk.07',
    'train.chunk.08',
    'train.chunk.09',
]


dev_file_list = [
    'dev.chunk.01'
]

test_file_list = [
    'test.chunk.01',
    'test.chunk.02'
]

train_path_list = [os.path.join(RAW_DATA_DIR, fn) for fn in train_file_list]
dev_path_list = [os.path.join(RAW_DATA_DIR, fn) for fn in dev_file_list]
test_path_list = [os.path.join(RAW_DATA_DIR, fn) for fn in test_file_list]

os.makedirs(PROCESSED_DATA_DIR, exist_ok = True)
os.makedirs(VOCAB_DIR, exist_ok=True)


def get_column_data(path_list, div, col):
    col_data = []
    for path in path_list:
        h = h5py.File(path, 'r')
        col_data.append(h[div][col][:])
        h.close()
    return np.concatenate(col_data)

def get_dataframe(path_list, div):
    pids = get_column_data(path_list, div ,col = 'pid')
    products = get_column_data(path_list, div, col='product')
    brands = get_column_data(path_list, div, col = 'brand')
    makers = get_column_data(path_list, div, col = 'maker')
    models = get_column_data(path_list, div, col = 'model')
    prices = get_column_data(path_list, div, col='price')
    updttms = get_column_data(path_list, div, col='updttm')
    bcates = get_column_data(path_list, div, col='bcates')
    mcates = get_column_data(path_list, div, col='mcates')
    scates = get_column_data(path_list, div, col='scates')
    dcates = get_column_data(path_list, div, col='dcates')
    
    df = pd.DataFrame({'pid':pids, "product":products, 'brand':brands, 'maker':makers,
                        'model':models, 'price':prices, 'updttm':updttms, 'bcateid':bcates, 'mcateid':mcates,
                        'scateid':scates, 'dcateid':dcates})

    # df['pid'] = df['pid'].map(lambda x: x.decode('utf-8'))
    # df['product'] = df['product'].map(lambda x: x.decode('utf-8'))
    # df['brand'] = df['brand'].map(lambda x: x.decode('utf-8'))
    # df['maker'] = df['maker'].map(lambda x: x.decode('utf-8'))
    # df['model'] = df['model'].map(lambda x: x.decode('utf-8'))
    # df['updttm'] = df['updttm'].map(lambda x: x.decode('utf-8'))

    return df

train_df = get_dataframe(train_path_list, 'train')
dev_df = get_dataframe(dev_path_list, 'dev')
test_df = get_dataframe(test_path_list, 'test')

print("-"*20, 'checking shapes',"-"*20)
print(train_df.shape)
print(dev_df.shape)
print(test_df.shape)


import json

cate_json = json.load(open(os.path.join(RAW_DATA_DIR,'cate1.json')))

bid2nm = dict([(cid, name) for name, cid in cate_json['b'].items()])
mid2nm = dict([(cid, name) for name, cid in cate_json['m'].items()])
sid2nm = dict([(cid, name) for name, cid in cate_json['s'].items()])
did2nm = dict([(cid, name) for name, cid in cate_json['d'].items()])

train_df['bcatenm'] = train_df['bcateid'].map(bid2nm)
train_df['mcatenm'] = train_df['mcateid'].map(bid2nm)
train_df['scatenm'] = train_df['scateid'].map(bid2nm)
train_df['dcatenm'] = train_df['dcateid'].map(bid2nm)


train_df = train_df[['pid','product','bcateid','mcateid','scateid','dcateid']]
dev_df = dev_df[['pid','product','bcateid','mcateid','scateid','dcateid']]
test_df = test_df[['pid','product','bcateid','mcateid','scateid','dcateid']]

train_df.to_csv("train_df", index=False)
dev_df.to_csv("dev_df", index=False)
test_df.to_csv("test_df", index=False)

