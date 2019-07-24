# -*- coding: utf-8 -*
import random
import pickle
import os
import re
import json
import time
import numpy as np
from tqdm import tqdm
import pickle

idiom_size=3848
word_size=585893
with open("/home/share/liyongqi/ChID/2id.pkl", 'rb') as f:
    idiom2id = pickle.load(f)
    word2id = pickle.load(f)
word_embedding=np.random.rand(word_size, 200)
idiom_embedding=np.random.rand(idiom_size, 200)



word_num=0
idiom_num=0
with open("/home/share/liyongqi/ChID/pre_trained word_embedding/Tencent_AILab_ChineseEmbedding.txt", 'r',encoding='utf8') as f:
    lines=f.readlines()
    for i in tqdm(range(1,len(lines))):
        line=lines[i].strip().split(" ")
        if(line[0] in word2id):
            word_embedding[word2id[line[0]]]=line[1:]
            word_num=word_num+1
        if(line[0] in idiom2id):
            idiom_embedding[idiom2id[line[0]]]=line[1:]
            idiom_num=idiom_num+1
print(float(word_num)/word_size)
print(float(idiom_num)/idiom_size)

with open("/home/share/liyongqi/ChID/word_embedding.pkl", 'wb') as f:
    pickle.dump(word_embedding, f)
    pickle.dump(idiom_embedding, f)