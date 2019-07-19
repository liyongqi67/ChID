# -*- coding: utf-8 -*-
import os
import pickle
import numpy as np
import random
import re
import jieba
import time
from utils import Vocabulary

random.seed(time.time())


class DataManager:
    def __init__(self):
        self.vocab = Vocabulary()
        self.ans = {}
        for line in open("/home/share/liyongqi/ChID/raw_data/train_answer.csv"):
            line = line.strip().split(',')
            self.ans[line[0]] = int(line[1])

        print("*** Finish building vocabulary")


    def get_num(self):
        num_word, num_idiom = len(self.vocab.word2id) - 2, len(self.vocab.idiom2id) 
        print("Numbers of words and idioms: %d %d" % (num_word, num_idiom))
        return num_word, num_idiom


    def _prepare_data(self, temp_data):
        cans = temp_data["candidates"]
        cans = [self.vocab.tran2id(each, True) for each in cans]

        for text in temp_data["content"]:
            content = re.split(r'(#idiom\d+#)', text)

            doc = []
            loc = []
            labs = []
            tags = []

            for i, segment in enumerate(content):
                if re.match(r'#idiom\d+#', segment) is not None:
                    tags.append(segment)
                    if segment in self.ans:
                        labs.append(self.ans[segment])
                    loc.append(len(doc))
                    doc.append(self.vocab.tran2id('#idiom#'))
                else:
                    doc += [self.vocab.tran2id(each) for each in jieba.lcut(segment)]

            yield doc, cans, labs, loc, tags


    def train(self, dev=False):
        if dev:
            file = open("/home/share/liyongqi/ChID/raw_data/train.txt")
            lines = file.readlines()[:10000]
        else:
            file = open("/home/share/liyongqi/ChID/raw_data/train.txt")
            lines = file.readlines()[10000:]
            random.shuffle(lines)
        for line in lines:
            temp_data = eval(line)
            for doc, cans, labs, loc, tags in self._prepare_data(temp_data):
                yield doc, cans, labs, loc, tags


    def test(self, file):
        for line in open(file):
            temp_data = eval(line)
            for doc, cans, _, loc, tags in self._prepare_data(temp_data):
                yield doc, cans, loc, tags


    def get_embed_matrix(self):  # DataManager
        with open("/home/share/liyongqi/ChID/word_embedding.pkl", 'rb') as f:
              word_embedding= np.float32(pickle.load(f))
              idiom_embedding= np.float32(pickle.load(f))

        self.word_embed_matrix = word_embedding
        self.idiom_embed_matrix = idiom_embedding


        print("*** Embed matrixs built")
        return self.word_embed_matrix, self.idiom_embed_matrix
