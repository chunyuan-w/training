#!/bin/bash

export DNNL_PRIMITIVE_CACHE_CAPACITY=1024

SEED=2020

CONFIG_FILE=""
precision=""

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
    precision="int8"

    if [ "$6" == "calibration" ]; then
        # TODO: why 2 in RN50? 
        BATCH_SIZE=2
        ARGS="$ARGS --calibration"
        echo "### running int8 calibration"
    elif [ "$6" == "jit" ]; then 
        ARGS="$ARGS --jit"
        echo "### running jit path"
        echo "### running int8 inference"
        if [ "$7" == "print" ]; then
            ARGS="$ARGS --print_time"
        fi
    else
        echo "### running int8 inference"
    fi
elif [ "$4" == "bf16" ]; then
    ARGS="$ARGS --mix-precision"
    echo "### running bf16 inference"
    precision="bf16"
    if [ "$5" == "jit" ]; then
        ARGS="$ARGS --jit"
        echo "### running jit path"
        if [ "$6" == "print" ]; then
            ARGS="$ARGS --print_time"
        fi
    fi
elif [ "$4" == "fp32" ]; then
    echo "### running fp32 inference"
    precision="fp32"
    if [ "$5" == "jit" ]; then
        ARGS="$ARGS --jit"
        echo "### running jit path"
        if [ "$6" == "print" ]; then
            ARGS="$ARGS --print_time"
        fi
    fi
fi

CORES=`lscpu | grep Core | awk '{print $4}'`
SOCKETS=`lscpu | grep Socket | awk '{print $2}'`
TOTAL_CORES=`expr $CORES \* $SOCKETS`

# change this number to adjust number of instances
CORES_PER_INSTANCE=$CORES

KMP_SETTING="KMP_AFFINITY=granularity=fine,compact,1,0"

BATCH_SIZE=64

export OMP_NUM_THREADS=$CORES_PER_INSTANCE
export $KMP_SETTING

echo -e "### using OMP_NUM_THREADS=$CORES_PER_INSTANCE"
echo -e "### using $KMP_SETTING\n\n"
sleep 3

INSTANCES=`expr $TOTAL_CORES / $CORES_PER_INSTANCE`
LAST_INSTANCE=`expr $INSTANCES - 1`
INSTANCES_PER_SOCKET=`expr $INSTANCES / $SOCKETS`
for i in $(seq 1 $LAST_INSTANCE); do
    numa_node_i=`expr $i / $INSTANCES_PER_SOCKET`
    start_core_i=`expr $i \* $CORES_PER_INSTANCE`
    end_core_i=`expr $start_core_i + $CORES_PER_INSTANCE - 1`
    LOG_i=throughput_log_ins${i}.txt

    echo "### running on instance $i, numa node $numa_node_i, core list {$start_core_i, $end_core_i}..."
    numactl --physcpubind=$start_core_i-$end_core_i --membind=$numa_node_i python -u inference.py \
     $ARGS $CONFIG_FILE --val_manifest $VAL_DATASET --model_toml configs/rnnt_ckpt.toml --batch_size $BATCH_SIZE --seed $SEED --warm_up 3 2>&1 | tee $LOG_i &
done

numa_node_0=0
start_core_0=0
end_core_0=`expr $CORES_PER_INSTANCE - 1`
LOG_0=throughput_log_ins0.txt

echo "### running on instance 0, numa node $numa_node_0, core list {$start_core_0, $end_core_0}...\n\n"
numactl --physcpubind=$start_core_0-$end_core_0 --membind=$numa_node_0 python -u inference.py \
     $ARGS $CONFIG_FILE --val_manifest $VAL_DATASET --model_toml configs/rnnt_ckpt.toml --batch_size $BATCH_SIZE --seed $SEED --warm_up 3 2>&1 | tee $LOG_0

sleep 10
throughput=$(grep 'Throughput:' ./throughput_log_ins* |sed -e 's/.*Throughput//;s/[^0-9.]//g' |awk '
BEGIN {
        sum = 0;
i = 0;
      }
      {
        sum = sum + $1;
i++;
      }
END   {
sum = sum / i;
        printf("%.3f", sum);
}')
echo ""RNN-T";"throughput";${precision};${BATCH_SIZE};${throughput}" | tee -a ${work_space}/summary.log
