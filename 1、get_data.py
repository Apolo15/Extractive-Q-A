'''
@File  :Question Answering with a fine-tuned BERT.py
@Author:zhangxu
@Date  :2022/1/1317:35
@Desc  :

https://towardsdatascience.com/question-answering-with-a-fine-tuned-bert-bc4dafd45626
一个外国小姐姐写的博客，我试着是复现一下这个模型

该文件是从斯坦佛的网站上获取数据——coqa,并且按照csv的格式存在了本地，方便下次调用
'''

import pandas as pd
import numpy as np
import torch
from transformers import BertForQuestionAnswering
from transformers import BertTokenizer

#COQA coqa是斯坦福NLP在2019年发布的会话问答数据集
coqa =pd.read_json('http://downloads.cs.stanford.edu/nlp/data/coqa/coqa-train-v1.0.json')

print(coqa)
#coqa.head 是什么我知不知道
coqa.head()

del coqa["version"]

cols=["text","question","answer"]

#合集，complete_list的意思
comp_list=[]

#一个index,row是一个问题
for index,row in coqa.iterrows():
    #判断每个样本里面有几个questions，一个story对应好几个问题和对应的答案
    for i in range(len(row["data"]["questions"])):
        temp_list=[]
        temp_list.append(row["data"]["story"])
        temp_list.append(row["data"]["questions"][i]["input_text"])
        temp_list.append(row["data"]["answers"][i]["input_text"])

        comp_list.append(temp_list)

new_df=pd.DataFrame(comp_list,columns=cols)

#saving the dataframeto csv file for further loading
new_df.to_csv("CoQA_data.csv",index=False)