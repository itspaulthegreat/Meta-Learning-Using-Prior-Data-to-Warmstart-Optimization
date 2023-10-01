#!/bin/bash
#SBATCH --job-name=acq_job
#SBATCH --partition=gpu_8
# del SBATCH --partition=gpu_4_a100
# del SBATCH --partition=dev_gpu_4_a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=20G
#SBATCH --time=8:30:00
#SBATCH --output=output_default_acq_acqmax.log
#SBATCH --gres=gpu:1

#SBATCH --mail-user=ali.sarlak@protonmail.com
#SBATCH --mail-type=END, FAIL, TIME_LIMIT

# Run the script
export PYTHONPATH="/pfs/data5/home/fr/fr_fr/fr_as1829/projects/automl"

module load jupyter/base/2023-03-24

nvidia-smi
python --version
pip install --upgrade pip
pip install pandas==2.0.3
pip install -r requirements.txt
nvidia-smi
pip list

python warmstart_template.py --warm-start="none" --experiment-name="different_acq" --strategy="HPO"
