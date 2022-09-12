# Multi-Modal-Cross-Attention-Network-for-e-Commerce-Product-Categorization

---

## ABSTRACT

---

As the spread of smart phones and the development of ICT, e-commerce market is rapidly growing. In the case of Naver shopping, a domestic online shopping platform, 20 million products are registered per day, and 5,000 product categories are selective. Retail vendor have difficulties in registering products by classifying them into an appropriate category among 5,000 categories. Categories have an important role for accurate search and correct product exposure. However, in most online shopping platforms, it is difficult to ensure accuracy by classifying categories based on retail vendor’s judgment. Accordingly, number of studies have been conducted to suggest machine learning and deep learning models for more accurate and objective product classification. However, most studies have been conducted on a single modality such as text, and studies on multi-modality have a limitation which interaction between modalities are not reflected. In addition, online product data is difficult to classify because the quality of the data is poor and categories are extremely asymmetric. Therefore, maximize the use of intermodality information is important. In this study, we propose a Cross Attention Block that reflects the interaction between independent modalities.

### Purpose of Model

This model get product name(text) and image as input to classify Large, Medium, Small Sub category which has 57, 552, 3190, 404 categories respectively.
<br>
Preview : Structure of the model <br>
<img src = 'png/1.png' height='300' width='500'> <br>
How to refer between two different modalities. <br>
<img src = 'png/2.png' height='300' width='500'> <br>

## Getting Trained

train.py holds all the training process.
<br>
To train text only modal
<br>

`python train.py --modality text --logging_fn ./saved/text/text_finetuning.log --model_fn ./saved/text/text_finetuning.pth` <br>

Since saved model is quite large, I except trained models from this repository.
<br>
To train multi modal(finetuned models are required)
<br>

`python train.py --modality both --multiModal_type cross --logging_fn ./saved/multi/full/multi_cross_6block.log --load_model_path ./saved/text/text.06.-0.93-0.88-.0.68-0.09-.pth`
