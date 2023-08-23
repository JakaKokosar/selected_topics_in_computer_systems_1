#!/bin/bash
#SBATCH --time=00:00:30
#SBATCH --job-name=perm_32-32
#SBATCH --output=stdout/%x-%A_%a.out
#SBATCH --error=stderr/%x-%A_%a.err
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=32
#SBATCH --ntasks-per-node=4
#SBATCH --nodes=8
#SBATCH --array=1-1000
#SBATCH --exclude=wn[051,052,053,061,062,064,065]


# Load module
module load Anaconda3/2022.05

# Specify the path to the Python interpreter of the conda environment
PYTHON_ENV_PATH="/d/hpc/home/jkokosar/miniconda3/envs/dev/bin/python"

tcga_project=$1
# data_input_file="data/${tcga_project}/${tcga_project}-data.parquet"
data_input_file="data/${tcga_project}/${tcga_project}-data.csv"
gene_input_file="data/${tcga_project}/all-combinations.csv"

# num_of_interactions=$(wc -l < $gene_input_file)

# Run the script with the Python interpreter of the environment  # mprof run -M 
srun $PYTHON_ENV_PATH compute.py --tcga_project "$tcga_project" --input_data "$data_input_file"  --input_genes "$gene_input_file" # --input_size "$num_of_interactions"

# Print detailed resource usage using sacct
echo "Detailed Resource Usage:"
sacct -j $SLURM_JOB_ID --format=user%10,jobname%10,node%10,start%10,end%10,elapsed%10,ElapsedRaw%10,MaxRSS%10,CPUTime%10,AveRSS%10,AveVMSize%10


