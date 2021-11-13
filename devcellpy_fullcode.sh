#!/bin/bash
#
#SBATCH --job-name=devcellpy
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20GB
#SBATCH --time=8:00:00

cd /home/groups/smwu/Sidra_FXG_Python/machine_learning_env/bin

module load python/3.6.1

source activate

cd

cd /scratch/groups/smwu/sidraxu

devcellpy --runMode trainAll --trainNormExpr /scratch/groups/smwu/sidraxu/zheng_pbmc_10K.csv --labelInfo /scratch/groups/smwu/sidraxu/zheng_labelinfo.csv --trainMetadata /scratch/groups/smwu/sidraxu/zheng_pbmc_10K_metadata.csv --testSplit 0.1 --rejectionCutoff 0.5

devcellpy --runMode predictOne --rejectionCutoff 0.5 --predNormExpr /scratch/groups/smwu/sidraxu/pbmc_10k_normalized.csv --predMetadata /scratch/groups/smwu/sidraxu/pbmc_10k_metadata.csv --layerObjectPaths /scratch/groups/smwu/sidraxu/cellpy_results_20210720155257/training/Root_object.pkl,/scratch/groups/smwu/sidraxu/cellpy_results_20210720155257/training/CD4_object.pkl,/scratch/groups/smwu/sidraxu/cellpy_results_20210720155257/training/CD8_object.pkl,/scratch/groups/smwu/sidraxu/cellpy_results_20210720155257/training/T-cell_object.pkl

devcellpy --runMode predictAll --rejectionCutoff 0.5 --predNormExpr /scratch/groups/smwu/sidraxu/pbmc_10k_normalized.csv --layerObjectPaths /scratch/groups/smwu/sidraxu/cellpy_results_20210720155257/training/Root_object.pkl,/scratch/groups/smwu/sidraxu/cellpy_results_20210720155257/training/CD4_object.pkl,/scratch/groups/smwu/sidraxu/cellpy_results_20210720155257/training/CD8_object.pkl,/scratch/groups/smwu/sidraxu/cellpy_results_20210720155257/training/T-cell_object.pkl

devcellpy --runMode predictOne --rejectionCutoff 0.5 --predNormExpr /scratch/groups/smwu/sidraxu/cardiac_normalized.csv --predMetadata /scratch/groups/smwu/sidraxu/cardiac_metadata.csv --layerObjectPaths cardiacDevAtlas --timePoint 10

devcellpy --runMode predictAll --rejectionCutoff 0.5 --predNormExpr /scratch/groups/smwu/sidraxu/cardiac_normalized.csv --layerObjectPaths cardiacDevAtlas --timePoint 13

devcellpy --runMode featureRankingOne --trainNormExpr /scratch/groups/smwu/sidraxu/zheng_pbmc_10K.csv --trainMetadata /scratch/groups/smwu/sidraxu/zheng_pbmc_10K_metadata.csv --layerObjectPaths /scratch/groups/smwu/sidraxu/cellpy_results_20210720155257/training/Root_object.pkl,/scratch/groups/smwu/sidraxu/cellpy_results_20210720155257/training/CD4_object.pkl,/scratch/groups/smwu/sidraxu/cellpy_results_20210720155257/training/CD8_object.pkl,/scratch/groups/smwu/sidraxu/cellpy_results_20210720155257/training/T-cell_object.pkl --featureRankingSplit 0.1