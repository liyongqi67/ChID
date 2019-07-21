# ChID
成语阅读理解大赛

## raw data 
train.txt<br/>
train_answer.txt<br/>
dev.txt<br/>
idiomDict.json<br/>

## Steps
### 1 Data processing <br/>
run python3 dataprocessing.py<br/>
run getw2v.py<br/>
### 2 LSTM based Model<br/>
run python3 LSTM based Model/train.py<br/>
start to train the model on dataset.pkl, and the parameters will be saved in model_save folder<br/>

run ptthon3 LSTM based Model/getAnswer.py<br/>
will obtain the submission.csv



### 2 AR Model<br/>
run python3 AR/main.py<br/>
start to train the model, and the parameters will be saved in train folder<br/>

change "flags.DEFINE_boolean("is_train", False, "training or testing a model")" to "flags.DEFINE_boolean("is_train", Tuue, "training or testing a model")" in AR/Flags.py and run python3 main.py, will obtain the submission.csv

## Results

| Models            | Dev      | Test | Out  |
| ---------         | -------- | ---- | ---- |
| LSTM based Model  | 30.22033 | -    | -    |
| AR Model          | 65.35570 | -    | -    |
