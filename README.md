# ChID
成语阅读理解大赛

## raw data 
train.txt<br/>
train_answer.txt<br/>
dev.txt<br/>
idiomDict.json<br/>

## Steps

### 1 LSTM based Model<br/>
run python3 dataprocessing.py<br/>
run python3 getw2v.py<br/>
run python3 train.py<br/>
start to train the model on dataset.pkl, and the parameters will be saved in model_save folder<br/>
will obtain the submission.csv



### 2 AR Model<br/>
run python3 AR/main.py<br/>
start to train the model, and the parameters will be saved in train folder<br/>

change "flags.DEFINE_boolean("is_train", False, "training or testing a model")" to "flags.DEFINE_boolean("is_train", Tuue, "training or testing a model")" in AR/Flags.py and run python3 main.py, will obtain the submission.csv


### 3 Interaction based Model<br/>
run python3 dataprocessing.py<br/>
run python3 getw2v.py<br/>
run python3 train.py<br/>
start to train the model on dataset.pkl, and the parameters will be saved in model_save folder<br/>
will obtain the submission.csv

### 4 NCF based Model<br/>
run python3 dataprocessing.py<br/>
run python3 getw2v.py<br/>
run python3 train.py<br/>
start to train the model on dataset.pkl, and the parameters will be saved in model_save folder<br/>
will obtain the submission.csv

### 5 Bert based Model<br/>
run run.sh

## Results

| Models            | Dev      | Test | Out  |
| ---------         | -------- | ---- | ---- |
| LSTM based Model  | 64.960236| -    | -    |
| AR Model          | 65.355707| -    | -    |
| Interaction based | 55.099735| -    | -    |
| NCF based Model   | 63.782538| -    | -    |
| Bert based Model  | 73.495286| -    | -    |
