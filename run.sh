#export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7,8,9
echo $CUDA_VISIBLE_DEVICES
accelerate launch run.py
accelerate launch inference.py
python make_submission.py
kaggle competitions submit -c stanford-ribonanza-rna-folding -f test.parquet -m "test27"