#!/bin/bash
#SBATCH --job-name=exo_bio
#SBATCH --output=logs/exo_bio_%a.out   # Fichiers de sortie individuels par tâche
#SBATCH --error=logs/exo_bio_%a.err    # Fichiers d'erreur individuels par tâche
#SBATCH --array=0-175%88                      # Lancer les tâches 0 à 9 (modifiable)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1                # À adapter selon ton script
#SBATCH --mem=10G                          # Mémoire par tâche
#SBATCH --time=12:00:00                   # Durée max (hh:mm:ss)
#SBATCH --partition=shared-cpu              # Nom de la partition 

# Charger les modules nécessaires (à adapter à ton environnement)
# module load python/3.10  # Exemple

# Lancer le script Python avec l'identifiant de tâche SLURM
python Couplage_exok_bio_principal_new_RK.py $SLURM_ARRAY_TASK_ID
