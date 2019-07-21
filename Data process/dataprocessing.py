# -*- coding: utf-8 -*
import random
import pickle
import os
import re
import json
import jieba
import time
import numpy as np
from tqdm import tqdm
import pickle

def fenci(s):
    seg_list = jieba.cut(s, cut_all=False, HMM=True)
    splited = " ".join(seg_list)

    try:
        return splited
    except(Exception):
        print(repr(Exception))
        return ""
'''
idiom2id={}  #
word2id={} #
word2id["<PAD>"]=0
word2id["<UNK>"]=1

with open("D://BaiduNetdiskDownload//open_data//train.txt", 'r',encoding='utf8') as f:
    lines = f.readlines()
    for line in tqdm(lines):
        line=eval(line)
        for s in line["candidates"]:
            if s not in idiom2id:
                idiom2id[s]=len(idiom2id)

        for s in line["content"]:
            for w in fenci(re.sub(r'#idiom\d+#','',s)).split(" "):
                if w not in word2id:
                    word2id[w]=len(word2id)
with open("D://BaiduNetdiskDownload//open_data//dev.txt", 'r',encoding='utf8') as f:
    lines = f.readlines()
    for line in tqdm(lines):
        line=eval(line)
        for s in line["candidates"]:
            if s not in idiom2id:
                idiom2id[s]=len(idiom2id)

        for s in line["content"]:
            for w in fenci(re.sub(r'#idiom\d+#','',s)).split(" "):
                if w not in word2id:
                    word2id[w]=len(word2id)
with open("D://BaiduNetdiskDownload//open_data//2id.pkl", 'wb') as f:
    pickle.dump(idiom2id, f)
    pickle.dump(word2id, f)
'''

with open("/home/share/liyongqi/ChID/2id.pkl", 'rb') as f:
    idiom2id = pickle.load(f)
    word2id = pickle.load(f)
ans={}
for line in open("/home/share/liyongqi/ChID/raw_data/train_answer.csv"):
    line = line.strip().split(',')
    ans[line[0]] = int(line[1])

train=[]
with open("/home/share/liyongqi/ChID/raw_data/train.txt", 'r',encoding='utf8') as f:
    lines = f.readlines()
    for line in tqdm(lines):
        line=eval(line)
        idioms=[]
        for s in line["candidates"]:
            idioms.append(idiom2id[s])
        for num in range(len(line["content"])):
            doc=[]
            loc=[]
            label=[]
            idiom_id=[]
            content = re.split(r'(#idiom\d+#)', line["content"][num])
            for s in content:
                if(s==""):
                    continue
                if re.match(r'#idiom\d+#', s) is not None:
                    loc.append(len(doc))
                    label.append(ans[s])
                    idiom_id.append(s)
                    doc.append(word2id["<UNK>"])
                else:
                    for each in fenci(re.sub(r'#idiom\d+#', '', s)).split(" "):
                        if each in word2id:
                            doc.append(word2id[each])
                        else:
                            doc.append(word2id["<UNK>"])
            for i in range(len(loc)):
                train.append((idioms,doc,loc[i],label[i],idiom_id[i]))  #[([idioms],[doc],)]

dev=[]
with open("/home/share/liyongqi/ChID/raw_data/dev.txt", 'r',encoding='utf8') as f:
    lines = f.readlines()
    for line in tqdm(lines):
        line=eval(line)
        idioms=[]
        for s in line["candidates"]:
            idioms.append(idiom2id[s])

        for num in range(len(line["content"])):
            doc=[]
            loc=[]
            label=[]
            idiom_id=[]
            content = re.split(r'(#idiom\d+#)', line["content"][num])
            for s in content:
                if(s==""):
                    continue
                if re.match(r'#idiom\d+#', s) is not None:
                    loc.append(len(doc))
                    idiom_id.append(s)
                    label.append(0)
                    doc.append(word2id["<UNK>"])                    
                else:
                    for each in fenci(re.sub(r'#idiom\d+#', '', s)).split(" "):
                        if each in word2id:
                            doc.append(word2id[each])
                        else:
                            doc.append(word2id["<UNK>"])
            for i in range(len(loc)):
                dev.append((idioms,doc,loc[i],label[i],idiom_id[i]))
with open("/home/share/liyongqi/ChID/dataset.pkl", 'wb') as f:
    pickle.dump(train, f)
    pickle.dump(dev, f)
print('len(train)',len(train))
print('len(dev)',len(dev))
print('ans["#idiom000000#"]',ans["#idiom000000#"])