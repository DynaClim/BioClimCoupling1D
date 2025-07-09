#!/bin/bash
#SBATCH --job-name=GEL_exo_bio
#SBATCH --output=logs/GEL_exo_bio_%a.out   
#SBATCH --error=logs/GEL_exo_bio_%a.err  
#SBATCH --array=0-49%50 
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1  
#SBATCH --mem=10G  
#SBATCH --time=12:00:00 
#SBATCH --partition=shared-cpu

python BioClim_coupling_Vocean.py $SLURM_ARRAY_TASK_ID
