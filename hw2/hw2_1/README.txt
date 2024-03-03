
# Run
```bash
#Please run all code on current PythonPath

#  Part 1 training, DATAPATH should include training_label.json and ./training_data/feat
python train.py $DATAPATH

# Part 2 testing, The default model path is s2vt_model.pth in current PythonPath
./hw2_seq2seq.sh $TESTDATAPATH $OUTPUT_TXT
```