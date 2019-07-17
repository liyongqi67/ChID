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

def fenci(s):
    seg_list = jieba.cut(s.strip(), cut_all=False, HMM=True)
    splited = " ".join(seg_list)

    try:
        return splited
    except(Exception):
        print(repr(Exception))
        return ""




with open("D://BaiduNetdiskDownload//open_data//2id.pkl", 'rb') as f:
    idiom2id = pickle.load(f)
    word2id = pickle.load(f)
ans={}
for line in open("D://BaiduNetdiskDownload//open_data//train_answer.csv"):
    line = line.strip().split(',')
    ans[line[0]] = int(line[1])

train=[]
with open("D://BaiduNetdiskDownload//open_data//train.txt", 'r',encoding='utf8') as f:
    lines = f.readlines()
    for line in tqdm(lines):
        line=eval(line)
        idioms=[]
        for s in line["candidates"]:
            idioms.append(idiom2id[s])

        doc=[]
        loc=[]
        label=[]
        for num in range(len(line["content"])):
            content = re.split(r'(#idiom\d+#)', line["content"][num])
            for s in content:
                if(s==""):
                    continue
                if re.match(r'#idiom\d+#', s) is not None:
                    loc.append(len(doc))
                    label.append(ans[s])
                else:
                    for each in fenci(re.sub(r'#idiom\d+#', '', s)).split(" "):
                        if each in word2id:
                            doc.append(word2id[each])
                        else:
                            doc.append(word2id["<UNK>"])
        for s in loc:
            train.append((idioms,doc,s,label))  #[([idioms],[doc],)]

dev=[]
with open("D://BaiduNetdiskDownload//open_data//dev.txt", 'r',encoding='utf8') as f:
    lines = f.readlines()
    for line in tqdm(lines):
        line=eval(line)
        idioms=[]
        for s in line["candidates"]:
            idioms.append(idiom2id[s])

        doc=[]
        loc=[]
        label=[]
        for num in range(len(line["content"])):
            content = re.split(r'(#idiom\d+#)', line["content"][num])
            for s in content:
                if(s==""):
                    continue
                if re.match(r'#idiom\d+#', s) is not None:
                    loc.append(len(doc))
                    label.append(0)
                else:
                    for each in fenci(re.sub(r'#idiom\d+#', '', s)).split(" "):
                        if each in word2id:
                            doc.append(word2id[each])
                        else:
                            doc.append(word2id["<UNK>"])
        for s in loc:
            dev.append((idioms,doc,s,label))
with open("D://BaiduNetdiskDownload//open_data//dataset.pkl", 'wb') as f:
    pickle.dump(train, f)
    pickle.dump(dev, f)