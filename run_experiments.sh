#!/bin/bash

#Declare a string array
declare -a seedArray=("200" "300" "400" "500" "600" "700")

EPOCHS=250

EXPNAME=20231108_test

for seedVal in "${seedArray[@]}"
do
    docker run --mount type=bind,source=/home/,target=/home --rm --gpus 4 --env NCCL_DEBUG=INFO --env NCCL_SOCKET_IFNAME=eth0 --env NCCL_P2P_LEVEL=NODE --shm-size=8G --privileged=true -it equity:0.1 bash -c "cd [HOMEDIR]/InfoMaxCutOut/; python3 main.py --batch_size=48 --test_batch_size=48 --lr=5e-5 --epochs=250 --seed=$seedVal --randaug --cutout --exp_name=${EXPNAME}_${seedVal} --image_path=[DATAPATH] --label_path=[LABEL_PATH]" 2>&1 | tee "${EXPNAME}_${seedVal}.txt"
done