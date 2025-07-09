#!/bin/bash
#SBATCH --job-name=exo_bio
#SBATCH --output=logs/exo_bio_%a.out   
#SBATCH --error=logs/exo_bio_%a.err 
#SBATCH --array=0-175%88  
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1     
#SBATCH --mem=10G          
#SBATCH --time=12:00:00 
#SBATCH --partition=shared-cpu

python BioClim_coupling.py $SLURM_ARRAY_TASK_ID
