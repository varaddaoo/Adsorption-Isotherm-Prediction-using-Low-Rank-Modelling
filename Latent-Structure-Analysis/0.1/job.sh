#!/bin/bash
#PBS -N SMED-N2-0.1
#PBS -l select=1:ncpus=8,walltime=100:00:00
#PBS -j oe
#PBS -e err.sm
#PBS -o out.sm

cd $PBS_O_WORKDIR

python-jl RUNM-17-08-2026.py --input CoRE2019-N2M-77K-mod.csv --sample RANDOM > out_0.1.log 2> warning_0.1.log
