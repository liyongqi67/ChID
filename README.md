# ChID
成语阅读理解大赛

## raw data 
train.txt<br/>
train_answer.txt<br/>
dev.txt<br/>
idiomDict.json<br/>

## Steps
## 1 Data processing <br/>
run python3 dataprocessing.py<br/>
You will obtain the new 2id.pkl and dataset.pkl, dataset.pkl contains of train and dev, train is a list where the element is a tuple(idioms, ducuments, label, loc), dev is same to train(but label is always 0 because we do not have the correct answer)<br/>
## 2 LSTM based Model<br/>
run python3 LSTM based Model/train.py<br/>
start to train the model on dataset.pkl, and the parameters will be saved in model_save folder<br/>

