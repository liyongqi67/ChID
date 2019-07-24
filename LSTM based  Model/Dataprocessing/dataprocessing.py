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
val=[]
with open("/home/share/liyongqi/ChID/raw_data/train.txt", 'r',encoding='utf8') as f:
    lines = f.readlines()
    lines1=lines[10000:]
    lines2=lines[:10000]
    for line in tqdm(lines1):
        temp_data=eval(line)
        cans = temp_data["candidates"]
        cans = [idiom2id[each] for each in cans]

        for text in temp_data["content"]:
            content = re.split(r'(#idiom\d+#)', text)

            doc = []
            loc = []
            labs = []
            tags = []

            for i, segment in enumerate(content):
                if re.match(r'#idiom\d+#', segment) is not None:
                    tags.append(segment)
                    if segment in ans:
                        labs.append(ans[segment])
                    loc.append(len(doc))
                    doc.append(word2id["<UNK>"])
                else:
                    for each in fenci(re.sub(r'#idiom\d+#', '', segment)).split(" "):
                        if each in word2id:
                            doc.append(word2id[each])
                        else:
                            doc.append(word2id["<UNK>"])
            for i in range(len(tags)):
                train.append((doc, cans, labs[i], loc[i], tags[i]))

    for line in tqdm(lines2):
        temp_data=eval(line)
        cans = temp_data["candidates"]
        cans = [idiom2id[each] for each in cans]

        for text in temp_data["content"]:
            content = re.split(r'(#idiom\d+#)', text)

            doc = []
            loc = []
            labs = []
            tags = []

            for i, segment in enumerate(content):
                if re.match(r'#idiom\d+#', segment) is not None:
                    tags.append(segment)
                    if segment in ans:
                        labs.append(ans[segment])
                    loc.append(len(doc))
                    doc.append(word2id["<UNK>"])
                else:
                    for each in fenci(re.sub(r'#idiom\d+#', '', segment)).split(" "):
                        if each in word2id:
                            doc.append(word2id[each])
                        else:
                            doc.append(word2id["<UNK>"])
            for i in range(len(tags)):
                val.append((doc, cans, labs[i], loc[i], tags[i]))
dev=[]
with open("/home/share/liyongqi/ChID/raw_data/dev.txt", 'r',encoding='utf8') as f:
    lines = f.readlines()
    for line in tqdm(lines):
        temp_data=eval(line)
        cans = temp_data["candidates"]
        cans = [idiom2id[each] for each in cans]

        for text in temp_data["content"]:
            content = re.split(r'(#idiom\d+#)', text)

            doc = []
            loc = []
            labs = []
            tags = []

            for i, segment in enumerate(content):
                if re.match(r'#idiom\d+#', segment) is not None:
                    tags.append(segment)
                    labs.append(0)
                    loc.append(len(doc))
                    doc.append(word2id["<UNK>"])
                else:
                    for each in fenci(re.sub(r'#idiom\d+#', '', segment)).split(" "):
                        if each in word2id:
                            doc.append(word2id[each])
                        else:
                            doc.append(word2id["<UNK>"])
            for i in range(len(tags)):
                dev.append((doc, cans, labs[i], loc[i], tags[i]))
with open("/home/share/liyongqi/ChID/dataset.pkl", 'wb') as f:
    pickle.dump(train, f)
    pickle.dump(val, f)
    pickle.dump(dev, f)
print('len(train)',len(train))
print('len(dev)',len(dev))
print('ans["#idiom000000#"]',ans["#idiom000000#"])