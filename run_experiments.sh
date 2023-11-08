#!/bin/bash

#Declare a string array
declare -a seedArray=("200" "300" "400" "500" "600" "700")

EPOCHS=250
#EXPNAME=20221015_iic_do_unet_0_5_randaugm4n10_convnext_stableadamw_meansep_ycbcrgn

#for seedVal in "${seedArray[@]}"
#do
#    docker run --mount type=bind,source=/home/,target=/home --rm --gpus 4 --env NCCL_DEBUG=INFO --env NCCL_SOCKET_IFNAME=eth0 --env NCCL_P2P_LEVEL=NODE --shm-size=8G --privileged=true -it equity:0.1 bash -c "cd /home/mdominguez/SoftGraphCutNet/; python3 main.py --batch_size=48 --test_batch_size=48 --lr=5e-5 --epochs=250 --seed=$seedVal --randaug --exp_name=${EXPNAME}_${seedVal} --image_path=/home/mdominguez/EquityLoss/data/data/finalfitz17k/ --label_path=/home/mdominguez/EquityLoss/data/fitzpatrick17k.csv" 2>&1 | tee "${EXPNAME}_${seedVal}.txt"
#done

EXPNAME=20231107_iic_do_unet_0_5_randaugm15n1_convnext_stableadamw_ycbcrgnfixeddatanorm_iic_butnoim_meanseplatecoords_cutout190_unetdo_classbackprop50percent

for seedVal in "${seedArray[@]}"
do
    docker run --mount type=bind,source=/home/,target=/home --rm --gpus 4 --env NCCL_DEBUG=INFO --env NCCL_SOCKET_IFNAME=eth0 --env NCCL_P2P_LEVEL=NODE --shm-size=8G --privileged=true -it equity:0.1 bash -c "cd /home/mdominguez/SoftGraphCutNet/; python3 main.py --batch_size=48 --test_batch_size=48 --lr=5e-5 --epochs=250 --seed=$seedVal --randaug --cutout --exp_name=${EXPNAME}_${seedVal} --image_path=/home/mdominguez/EquityLoss/data/data/finalfitz17k/ --label_path=/home/mdominguez/EquityLoss/data/fitzpatrick17k.csv" 2>&1 | tee "${EXPNAME}_${seedVal}.txt"
done