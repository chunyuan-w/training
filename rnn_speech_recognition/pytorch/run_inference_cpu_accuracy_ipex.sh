#!/bin/bash

export DNNL_PRIMITIVE_CACHE_CAPACITY=1024

SEED=2020
BATCH_SIZE=1

CONFIG_FILE=""

ARGS=""

ARGS="$ARGS --dataset_dir $1 --ckpt $2"
echo "### dataset path: $1"
echo "### checkpoint path: $2"

VAL_DATASET="$1/librispeech-dev-clean-wav.json"


if [ "$3" == "ipex" ]; then
    ARGS="$ARGS --ipex"
    echo "### running ipex mode"
fi

if [ "$4" == "int8" ]; then
    ARGS="$ARGS --int8"
    CONFIG_FILE="$CONFIG_FILE --configure-dir $5"

    if [ "$6" == "calibration" ]; then
        # TODO: why 2 in RN50? 
        BATCH_SIZE=2
        ARGS="$ARGS --calibration"
        echo "### running int8 calibration"
    elif [ "$6" == "jit" ]; then 
        ARGS="$ARGS --jit"
        echo "### running jit path"
        echo "### running int8 inference"
    else
        echo "### running int8 inference"
    fi
elif [ "$4" == "bf16" ]; then
    ARGS="$ARGS --mix-precision"
    echo "### running bf16 inference"
    if [ "$5" == "jit" ]; then
        ARGS="$ARGS --jit"
        echo "### running jit path"
    fi
elif [ "$4" == "fp32" ]; then
    echo "### running fp32 inference"
    if [ "$5" == "jit" ]; then
        ARGS="$ARGS --jit"
        echo "### running jit path"
    fi
fi


CORES=`lscpu | grep Core | awk '{print $4}'`
SOCKETS=`lscpu | grep Socket | awk '{print $2}'`
TOTAL_CORES=`expr $CORES \* $SOCKETS`

KMP_SETTING="KMP_AFFINITY=granularity=fine,compact,1,0"


export OMP_NUM_THREADS=$TOTAL_CORES
export $KMP_SETTING

echo -e "### using OMP_NUM_THREADS=$TOTAL_CORES"
echo -e "### using $KMP_SETTING\n\n"
sleep 3

python -u inference.py $ARGS $CONFIG_FILE --val_manifest $VAL_DATASET --model_toml configs/rnnt_ckpt.toml --batch_size $BATCH_SIZE --seed $SEED