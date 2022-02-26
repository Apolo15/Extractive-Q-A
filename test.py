'''
@File  :test.py
@Author:zhangxu
@Date  :2022/2/1418:44
@Desc  :
'''
from transformers import BertModel,BertTokenizer

BERT_PATH = 'F:\Model\BERT\pytorch\bert-base-chinese'

tokenizer = BertTokenizer.from_pretrained("./bert_localpath/")

print(tokenizer.tokenize('I have a good time, thank you.'))

bert = BertModel.from_pretrained("./bert_localpath")

print('load bert model over')
