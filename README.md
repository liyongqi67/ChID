# ChID
成语阅读理解大赛

## raw data 
train.txt<br/>
train_answer.txt<br/>
dev.txt<br/>
idiomDict.json<br/>

## Steps
1 run python3 dataprocessing.py
you will obtain the new 2id.pkl and dataset.pkl, dataset.pkl contains of train and dev, train is a list where the element is a tuple(idioms, ducuments, label, loc), dev is same to train(but label is always 0 because we do not have the correct answer)
