#!/bin/bash
#SBATCH --job-name=GEL_exo_bio
#SBATCH --output=logs/GEL_exo_bio_%a.out   # Fichiers de sortie individuels par tâche
#SBATCH --error=logs/GEL_exo_bio_%a.err    # Fichiers d'erreur individuels par tâche
#SBATCH --array=0-49%50                      # Lancer les tâches 0 à 9 (modifiable)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1                # À adapter selon ton script
#SBATCH --mem=10G                          # Mémoire par tâche
#SBATCH --time=12:00:00                   # Durée max (hh:mm:ss)
#SBATCH --partition=shared-cpu              # Nom de la partition

# Charger les modules nécessaires (à adapter à ton environnement)
# module load python/3.10  # Exemple

# Lancer le script Python avec l'identifiant de tâche SLURM
python Couplage_exok_bio_principal_new_RK_Vocean_var.py $SLURM_ARRAY_TASK_ID
