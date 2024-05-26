#!/bin/bash
#SBATCH -J job_name
#SBATCH --output=../Plots/out_%A.out
#SBATCH --error=../Plots/err_%A.err
#SBATCH -A <Project name MPHIL-DIS-SL2-CPU>
#SBATCH --time=1:00:00
#SBATCH -p icelake
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

#! Optionally modify the environment seen by the application
#! (note that SLURM reproduces the environment at submission irrespective of ~/.bashrc):
. /etc/profile.d/modules.sh                # Leave this line (enables the module command)
module purge                               # Removes all modules still loaded
module load rhel8/default-amp              # REQUIRED - loads the basic environment

python spectral_clust_bl.py 0.5
