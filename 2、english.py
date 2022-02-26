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

model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

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


# text = """New York (CNN) -- More than 80 Michael Jackson collectibles --
# including the late pop star's famous rhinestone-studded glove from a 1983 performance --
# were auctioned off Saturday, reaping a total $2 million. Profits from the auction at the Hard Rock Cafe in New York's Times
# Square crushed pre-sale expectations of only $120,000 in sales. The highly prized memorabilia, which included items spanning
# the many stages of Jackson's career, came from more than 30 fans, associates and family members, who contacted Julien's Auctions
# to sell their gifts and mementos of the singer. Jackson's flashy glove was the big-ticket item of the night, fetching $420,000 from
# a buyer in Hong Kong, China. Jackson wore the glove at a 1983 performance during \"Motown 25,\" an NBC special where he debuted his
# revolutionary moonwalk. Fellow Motown star Walter \"Clyde\" Orange of the Commodores, who also performed in the special 26 years ago, said he asked for Jackson's autograph at the time, but Jackson gave him the glove instead. "The legacy that [Jackson] left behind is bigger than life for me,\" Orange said. \"I hope that through that glove people can see what he was trying to say in his music and what he said in his music.\" Orange said he plans to give a portion of the proceeds to charity. Hoffman Ma, who bought the glove on behalf of Ponte 16 Resort in Macau, paid a 25 percent buyer's premium, which was tacked onto all final sales over $50,000. Winners of items less than $50,000 paid a 20 percent premium."""
# question = "Where was the Auction held?"

text="""Of course, with the emergence of large-scale snowfall, the dry and cold air mass that has quickly cleared up after the snow has prevailed, and the "cold after snow" in many places in northern my country has also arrived quickly - this morning at 6 o'clock in the morning,
Judging from the 24-hour temperature change chart, many places in northern my country have become cooling centers today. Inner Mongolia, Shanxi, Hebei, Beijing, Tianjin and Heilongjiang have generally experienced significant cooling of 4 to 8 degrees.
The 24-hour cooling rate in some areas is as high as 15 degrees or more, and individual sites have experienced a severe cooling rate of 18-20 degrees, which can be called "cold wave cooling". severe cold weather,
The low temperature of -11.3 degrees appeared in the early morning of the southern suburbs observatory in Beijing, which is a new low for this winter, and even in the urban area during the day, it is generally below zero."""
question="how is the weather in beijing"
question_answer(question, text)