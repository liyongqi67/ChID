# coding=UTF-8
import heapq
import random
import math
import tensorflow as tf
import numpy as np
import os
import io
import time
import datetime
import network
from tensorflow.contrib import learn
from tqdm import tqdm
import sklearn.metrics
# Parameters
# ==================================================
import sys

import csv
import pickle

#data file

time=time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())

print(time)





#下面是模型的超参
lr=0.001

settings = network.Settings()
idiom_num=10
doc_len=300
dev_batchSize=256

idiom_size=3848
word_size=585893
# Load data
print("Loading data...")

with open("/home/share/liyongqi/ChID/word_embedding.pkl", 'rb') as f:
  word_embedding= pickle.load(f)
  idiom_embedding= pickle.load(f)

with open("/home/share/liyongqi/ChID/dataset.pkl", 'rb') as f:
  train_data= pickle.load(f)
  dev_data= pickle.load(f)
print(len(dev_data))




def getBatch(batchSize,num,data):
        if((num+1)*batchSize>len(data)):
          temp_batchsize=len(data)-num*batchSize
          idiom_input=np.zeros([temp_batchsize,idiom_num],dtype=np.int)
          doc_input=np.zeros([temp_batchsize,doc_len],dtype=np.int)
          loc_input=np.zeros([temp_batchsize,1],dtype=np.int)
          label_input=np.zeros([temp_batchsize,idiom_num],dtype=np.int)
          sequence_len_input=np.zeros([temp_batchsize],dtype=np.int)

          for i in range(num*batchSize,len(data)):
              idiom_input[i%temp_batchsize]=data[i][0]
              loc_input[i%temp_batchsize]=data[i][2]
              label_input[i%temp_batchsize,data[i][3]]=1

              if(len(data[i][1])>doc_len):
                  doc_input[i%temp_batchsize]=data[i][1][:doc_len]
                  sequence_len_input[i%temp_batchsize]=doc_len
              else:
                  doc_input[i%temp_batchsize]=data[i][1]+[0]*(doc_len-len(data[i][1]))
                  sequence_len_input[i%temp_batchsize]=len(data[i][1])  
        else:
        
          idiom_input=np.zeros([batchSize,idiom_num],dtype=np.int)
          doc_input=np.zeros([batchSize,doc_len],dtype=np.int)
          loc_input=np.zeros([batchSize,1],dtype=np.int)
          label_input=np.zeros([batchSize,idiom_num],dtype=np.int)
          sequence_len_input=np.zeros([batchSize],dtype=np.int)

          for i in range(num*batchSize,(num+1)*batchSize):
              idiom_input[i%batchSize]=data[i][0]
              loc_input[i%batchSize]=data[i][2]
              label_input[i%batchSize,data[i][3]]=1

              if(len(data[i][1])>doc_len):
                  doc_input[i%batchSize]=data[i][1][:doc_len]
                  sequence_len_input[i%batchSize]=doc_len
              else:
                  doc_input[i%batchSize]=data[i][1]+[0]*(doc_len-len(data[i][1]))
                  sequence_len_input[i%batchSize]=len(data[i][1])        


        return idiom_input,doc_input,loc_input,label_input,sequence_len_input
        



# Placeholders for input, output and dropout
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
#config.gpu_options.allow_growth = True 
sess = tf.InteractiveSession(config=config)


model = network.Lstm(word_embedding,idiom_embedding,settings)



train_op=tf.train.AdamOptimizer(0.001).minimize(model.loss,global_step=model.global_step)
update_op = tf.group(*model.update_emas)

sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()

saver.restore(sess,  "./model_save/model.ckpt")

num=len(dev_data)/dev_batchSize
num=int(num)+1

submission=[]
for i in tqdm(range(num)):
    
    idiom_input,doc_input,loc_input,label_input,sequence_len_input=getBatch(dev_batchSize,i,dev_data)
    _,_,loss,y_pred =sess.run([train_op,update_op,model.loss,model.output],
      feed_dict = {
                   model.idiom_inputs: idiom_input, 
                   model.doc_inputs: doc_input,  
                   model.loc_inputs: loc_input, 
                   model.label_inputs: label_input, 
                   model.sequence_len_inputs:sequence_len_input,
                   model.tst: False, 
                   model.keep_prob: 1.0})
    submission.extend(np.argmax(y_pred,1).tolist())
   

with open("submission.csv","w",newline="", encoding="utf-8-sig") as f:
    csv_write = csv.writer(f)
    num=577157
    for s in submission:
      temp=[]
      temp.append("#idiom"+str(num)+"#")
      temp.append(str(s))
      csv_write.writerow(temp)
      num=num+1



