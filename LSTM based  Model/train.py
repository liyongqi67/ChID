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
doc_len=200 #avg_len=104
train_batchSize=64
val_batchsize=64
idiom_size=3848
word_size=585893
# Load data
print("Loading data...")
with open("/home/share/liyongqi/ChID/word_embedding.pkl", 'rb') as f:
    word_embedding= pickle.load(f)
    idiom_embedding= pickle.load(f)

with open("/home/share/liyongqi/ChID/dataset.pkl", 'rb') as f:
    train_data= pickle.load(f)
    val_data= pickle.load(f)
    dev_data= pickle.load(f)

def getBatch(batchSize,num,data):
    #(doc, cans, labs, loc, tags)
    if((num+1)*batchSize>len(data)):
      temp_batchsize=len(data)-num*batchSize
      idiom_input=np.zeros([temp_batchsize,idiom_num],dtype=np.int)
      doc_input=np.zeros([temp_batchsize,doc_len],dtype=np.int)
      loc_input=np.zeros([temp_batchsize,1],dtype=np.int)
      label_input=np.zeros([temp_batchsize,idiom_num],dtype=np.int)
      sequence_len_input=np.zeros([temp_batchsize],dtype=np.int)

      blank_keys=[]

      for i in range(num*batchSize,len(data)):
          idiom_input[i%batchSize]=data[i][1]
          loc_input[i%batchSize]=data[i][3]
          label_input[i%batchSize,data[i][2]]=1
          blank_keys.append(data[i][4])

          if(len(data[i][0])>doc_len):
              doc_input[i%batchSize]=data[i][0][:doc_len]
              sequence_len_input[i%batchSize]=doc_len
          else:
              doc_input[i%batchSize]=data[i][0]+[0]*(doc_len-len(data[i][0]))
              sequence_len_input[i%batchSize]=len(data[i][0])  

    else:    
      idiom_input=np.zeros([batchSize,idiom_num],dtype=np.int)
      doc_input=np.zeros([batchSize,doc_len],dtype=np.int)
      loc_input=np.zeros([batchSize,1],dtype=np.int)
      label_input=np.zeros([batchSize,idiom_num],dtype=np.int)
      sequence_len_input=np.zeros([batchSize],dtype=np.int)
      blank_keys=[]
      for i in range(num*batchSize,(num+1)*batchSize):
          idiom_input[i%batchSize]=data[i][1]
          loc_input[i%batchSize]=data[i][3]
          label_input[i%batchSize,data[i][2]]=1
          blank_keys.append(data[i][4])
          if(len(data[i][0])>doc_len):
              doc_input[i%batchSize]=data[i][0][:doc_len]
              sequence_len_input[i%batchSize]=doc_len
          else:
              doc_input[i%batchSize]=data[i][0]+[0]*(doc_len-len(data[i][0]))
              sequence_len_input[i%batchSize]=len(data[i][0])  

    return idiom_input,doc_input,loc_input,label_input,sequence_len_input,blank_keys

def eva(sess,model):
    total_count=len(val_data)/val_batchsize
    total_count=int(total_count)+1
    y_true=[]
    y_pred=[]
    for num in tqdm(range(total_count)):
        
        idiom_input,doc_input,loc_input,label_input,sequence_len_input,blank_keys=getBatch(val_batchsize,num,val_data)
        loss,y_out =sess.run([model.loss,model.pred],
          feed_dict = {
                       model.idiom_inputs: idiom_input, 
                       model.doc_inputs: doc_input,  
                       model.loc_inputs: loc_input, 
                       model.label_inputs: label_input, 
                       model.sequence_len_inputs:sequence_len_input,
                       model.tst: True, 
                       model.keep_prob: 1.0})

        y_true.extend(list(np.argmax(label_input,1)))
        y_pred.extend(list(y_out))


    acc=sklearn.metrics.accuracy_score(y_true,y_pred)
    return acc



# Placeholders for input, output and dropout
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True 
sess = tf.InteractiveSession(config=config)



model = network.Lstm(word_embedding,idiom_embedding,settings)



train_op=tf.train.AdamOptimizer(0.001).minimize(model.loss,global_step=model.global_step)
update_op = tf.group(*model.update_emas)

sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()

#saver.restore(sess,  "./model_save/model.ckpt")


with open(time+".txt", "a") as f:
#    acc=eva(sess,model)
    best_score=0
    for epoch in range(10000000):
        total_count=len(train_data)/train_batchSize
        total_count=int(total_count)+1

        for num in tqdm(range(total_count)):

            idiom_input,doc_input,loc_input,label_input,sequence_len_input,blank_keys=getBatch(train_batchSize,num,train_data)
            _,_,loss,y_pred =sess.run([train_op,update_op,model.loss,model.pred],
              feed_dict = {
                           model.idiom_inputs: idiom_input, 
                           model.doc_inputs: doc_input,  
                           model.loc_inputs: loc_input, 
                           model.label_inputs: label_input, 
                           model.sequence_len_inputs:sequence_len_input,
                           model.tst: False, 
                           model.keep_prob: 0.5})
            if num%100==1:
                
                acc=sklearn.metrics.accuracy_score(np.argmax(label_input,1),y_pred)
                print('acc in train:', acc)
                print("epoch:",epoch,' num:',num,' loss:',loss)
                f.write("epoch:"+str(epoch)+' num:'+str(num)+' loss:'+str(loss)+' acc:'+str(acc)+"\n")
                f.flush()
                if num%1000==1 and not num==1:
                    acc=eva(sess,model)
                    if(acc>best_score):
                        save_path = saver.save(sess, "./model_save/model.ckpt")
                        best_score=acc
                    print('epoch:',epoch,'acc on val:',acc,'best_score',best_score)
                    f.write('epoch:'+str(epoch)+' acc on val:'+str(acc)+' best_score:'+str(best_score)+"\n" )
                    f.flush() 
sess.close()

