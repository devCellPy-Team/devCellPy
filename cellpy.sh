#!/bin/bash
#
#SBATCH --job-name=cellpy
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20GB
#SBATCH --time=8:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=21sidrax@students.harker.org

cd /home/groups/smwu/Sidra_FXG_Python/machine_learning_env/bin

module load python/3.6.1

source activate

cd

cd /scratch/groups/smwu/sidraxu/abcd

python /scratch/groups/smwu/sidraxu/abcd/full.py --runMode trainAndPredict --trainNormExpr /scratch/groups/smwu/sidraxu/abcd/cui_normexpr.csv --labelInfo /scratch/groups/smwu/sidraxu/abcd/cui_labelinfo.csv --trainMetadata /scratch/groups/smwu/sidraxu/abcd/cui_metadata.csv --testSplit 0.1 --featureRanking off --rejectionCutoff 0.5 --predNormExpr /scratch/groups/smwu/sidraxu/abcd/lmna_normexpr.csv
