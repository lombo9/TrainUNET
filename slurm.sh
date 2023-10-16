#!/bin/bash 
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --job-name=test
#SBATCH --cpus-per-task 1
#SBATCH --mail-type=All
#SBATCH --mail-user=my_email_address
#SBATCH -o test_out.txt
#SBATCH -e test_err.txt
#SBATCH --partition=vgpu
#SBATCH --gres=gpu:1


source /home/Student/s4585713/anaconda/bin/activate /home/Student/s4585713
python testTrain.py