#!/bin/bash
#
#SBATCH --job-name=cellpy
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20GB
#SBATCH --time=8:00:00

cd /home/groups/smwu/Sidra_FXG_Python/machine_learning_env/bin

module load python/3.6.1

source activate

cd

cd /scratch/groups/smwu/sidraxu

python /Users/sidraxu/Documents/GitHub/CellPy/full.py --runMode trainAndPredict --trainNormExpr /Users/sidraxu/Downloads/CELLPY_TEST_PBMC_DATA/zheng_pbmc_2.5K.csv --labelInfo /Users/sidraxu/Downloads/zheng_labelinfo.csv --trainMetadata /Users/sidraxu/Downloads/CELLPY_TEST_PBMC_DATA/zheng_pbmc_2.5K_metadata.csv --testSplit 0.1 --featureRanking on --rejectionCutoff 0.5 --predNormExpr /Users/sidraxu/Downloads/CELLPY_TEST_PBMC_DATA/pbmc_10k_normalized.csv --predMetadata /Users/sidraxu/Downloads/CELLPY_TEST_PBMC_DATA/pbmc_10k_metadata.csv
