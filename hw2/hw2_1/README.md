
# Run

#Please run all code on current PythonPath

##  Part 1 training, DATAPATH should include training_label.json and ./training_data/feat
```bash
python train.py $DATAPATH
```
### the total training time should be less than 30 mins for 400 epochs at RTX 4090

## Part 2 testing, The default model path is s2vt_model.pth in current PythonPath
### If you want to set the specific model_path, please modified the hw2_seq2seq.sh. Otherwise, it will use default model './s2vt_model.pth'
```bash
bash ./hw2_seq2seq.sh $TESTDATAPATH $OUTPUT_TXT
```
### the total training time should be less than 1 mins at RTX 4090
