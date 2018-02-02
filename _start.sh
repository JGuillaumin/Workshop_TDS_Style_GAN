#!/usr/bin/env bash


# source activate carnd-term1

cd /mnt/s0
CUDA_VISIBLE_DEVICES='0' jupyter notebook --port=8888 &
PID_JUPYTER_S0=$!

cd /mnt/s1
CUDA_VISIBLE_DEVICES='1' jupyter notebook --port=8889 &
PID_JUPYTER_S1=$!

cd /mnt/s2
CUDA_VISIBLE_DEVICES='2' jupyter notebook --port=8890 &
PID_JUPYTER_S2=$!

cd /mnt/s3
CUDA_VISIBLE_DEVICES='3' jupyter notebook --port=8891 &
PID_JUPYTER_S3=$!

