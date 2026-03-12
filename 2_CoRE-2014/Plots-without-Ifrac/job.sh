#!/bin/bash
#PBS -N NoIfrac
#PBS -l select=1:ncpus=2,walltime=1000:00:00
#PBS -j oe
#PBS -e err.sm
#PBS -o out.sm

cd $PBS_O_WORKDIR

time python-jl RUNM-18-02-2025-CoRE-2019-Ar.py > out_NoIfrac.log 2> warning_NoIfrac.log
