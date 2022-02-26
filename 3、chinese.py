'''
@File  :2、english.py
@Author:zhangxu
@Date  :2022/1/2417:36
@Desc  :
'''

import pandas as pd
import numpy as np
import torch
from transformers import BertForQuestionAnswering
from transformers import BertTokenizer

# data=pd.read_csv("CoQA_data.csv")
# print(data.head())
#获取总共有多少条数据：108647
# print("lenof question and answers: ",len(data))

model = BertForQuestionAnswering.from_pretrained("./bert_localpath/")
tokenizer = BertTokenizer.from_pretrained("./bert_localpath/")

def question_answer(question, text):
    # tokenize question and text as a pair
    input_ids = tokenizer.encode(question, text)

    # string version of tokenized ids
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    # segment IDs
    # first occurence of [SEP] token
    sep_idx = input_ids.index(tokenizer.sep_token_id)
    # number of tokens in segment A (question)
    num_seg_a = sep_idx + 1
    # number of tokens in segment B (text)
    num_seg_b = len(input_ids) - num_seg_a

    # list of 0s and 1s for segment embeddings
    segment_ids = [0] * num_seg_a + [1] * num_seg_b
    assert len(segment_ids) == len(input_ids)

    # model output using input_ids and segment_ids
    output = model(torch.tensor([input_ids]), token_type_ids=torch.tensor([segment_ids]))

    # reconstructing the answer
    answer_start = torch.argmax(output.start_logits)
    answer_end = torch.argmax(output.end_logits)
    if answer_end >= answer_start:
        answer = tokens[answer_start]
        for i in range(answer_start + 1, answer_end + 1):
            if tokens[i][0:2] == "##":
                answer += tokens[i][2:]
            else:
                answer += " " + tokens[i]

    if answer.startswith("[CLS]"):
        answer = "Unable to find the answer to your question."

    print("\nPredicted answer:\n{}".format(answer.capitalize()))


text = """当然，随着大范围的降雪出现，已经雪后的迅速放晴干冷气团占据上风，我国北方多地的“雪后寒”也迅速到来——今天清晨6点，
从24小时气温变化图上看，今天我国北方多地成为降温中心，内蒙古、山西、河北、北京、天津及黑龙江等地普遍出现了4～8度的明显降温，
部分地区24小时降温幅度更是高达15度以上，个别站点出现了18-20度的剧烈降温幅度，堪称“寒潮式降温”比如下了很大雪的北京，受雪后辐射降温影响，今天北京出现严寒天气，
北京南郊观象台凌晨出现了-11.3度的低温，为今冬新低，而即便是白天市区也普遍在零下，下午两点，山区最低甚至还低至-20度以下，可谓相当冰冷"""
question = "北京天气如何"
question_answer(question, text)