#!/bin/bash
#SBATCH --job-name=faces_extraction
#SBATCH --output=%x-%j.out
#SBATCH -n 1 # Number of tasks
#SBATCH --cpus-per-task 1  # number cpus per task
#SBATCH --mem=16384 # Memory - Use up to 2GB per requested CPU as a rule of thumb
#SBATCH --time=0 # No time limit
#SBATCH --gres=gpu:3g.20gb:1
eval "$(conda shell.bash hook)"
# activate desired environment
conda activate test
# change dir to where we want to run scripts
cd ~/test/src/extract-faces
# run program
python mediaeval_faces.py
