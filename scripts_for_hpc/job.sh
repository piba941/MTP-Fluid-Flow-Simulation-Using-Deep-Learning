#!/bin/bash
#PBS -q gpuq
#PBS -o out.o
#PBS -e out.e
#PBS -N 26_dec
#PBS -j oe
#PBS -l nodes=4:ppn=1
#PBS -V
cd ${PBS_O_WORKDIR}
echo "Running on: "
cat ${PBS_NODEFILE}
cat $PBS_NODEFILE > machines.list
echo "Program Output begins: "
conda init bash
source activate tflow
python -m learning_to_simulate.train  --data_path=/home/b17020/wd/main/datasets/WaterRamps --model_path=/home/b17020/wd/main/models/wr
#python -m learning_to_simulate.train --mode="eval" --data_path=/home/b17020/wd/main/datasets/WaterRamps
