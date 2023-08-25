#!/bin/bash

chmod +x ./s5cmd
./s5cmd sync s3://$MODEL_S3_BUCKET/$MODEL_NAME_S3/pretrain/* /tmp/whisper-large-v2/
./s5cmd sync $DATA_S3/* /tmp/data/
# /usr/bin/python -m torch.distributed.launch \
#     --nproc_per_node 8 train.py --dataloader_num_workers 16 --eval_batch_size 1 --language Marathi --model_name /tmp/whisper-large-v2/ --model-dir /tmp/whisper_out --num_train_epochs 3 --train_batch_size 1

/usr/bin/python train.py --dataloader_num_workers 16 --eval_batch_size 8 --language Marathi --model_name /tmp/whisper-large-v2/ --model-dir /tmp/whisper_out --out-dir /tmp --num_train_epochs 10 --train_batch_size 8 --is_8_bit True --training_dir /tmp/data/

./s5cmd sync /tmp/whisper_out s3://$MODEL_S3_BUCKET//$MODEL_NAME_S3/output/$(date +%Y-%m-%d-%H-%M-%S)/