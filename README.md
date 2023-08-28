# Problem

Understanding the interactions between genes is pivotal in unraveling the complexities of biology. Working with high-dimensional datasets, particularly those comprising roughly 20,000 known genes, presents a computational challenge. For example, we need to check 199,980,000 combinations when investigating all pairwise interactions. This complexity is further amplified as multiple experiments are often conducted with different permutations of data to assess the significance of the findings empirically.

Luckily, these experiments are easily parallelizable, as each basic computation step (e.g., calculation of pairwise interaction scores) is independent of the others. However, the high dimensionality of genomic data renders local machine calculations impractical. Implementing a programmatic approach that enables parallelization across multiple compute cores and distributes computations across nodes on compute clusters running SLURM can overcome this challenge.

This exercise aims to uncover gene interactions based on survival data, utilizing gene expression information and corresponding survival times from patients across the TCGA cohort for multiple cancer types.

# Data

This exercise focuses on 33 TCGA experiments, specifically TCGA-HNSC (head and neck squamous cell carcinoma). The data is further refined by selecting well-known [L1000](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5990023/) landmark genes, reducing noise through normalizing and standardizing gene expression values and discarding genes with low expression levels. We perform all these preprocessing steps before leveraging the compute cluster.

For slected TCGA experiment (HNSC) the final processed data consists of two files:

1. **TCGA-HNSC-data.csv**: After preprocessing, this dataset includes 519 samples and 857 genes + survival data.
2. **TCGA-HNSC-combinations.csv**: This file contains 365,085 unique gene pairs, where each row represents a unique combination.

These numbers differ across the various TCGA datasets, reflecting each dataset's specific characteristics and requirements.

# Methods

We implemented a python script which has two inputs; **data** and all **pair-wise combinations of genes**. We then split input genes in two distinct ways; given a number of tasks or CPUs. 

![](images/image0.png)

We utilize the following SLURM environment variables: `SLURM_PROCID`, `SLURM_NTASKS`, and `SLURM_CPUS_PER_TASK`. The input gene array is divided based on the number of tasks. Subsequently, we index the corresponding segment using `SLURM_PROCID`. After obtaining the necessary gene segment, we employ the Joblib Python library to further distribute the genes across the available compute cores.

Ultimately, we would like to use this environment to speed up our experiments. Here, we are focusing on a specific dataset and a reduced number of genes. On our local machine, M1 Macs, there are 8 high-performance cores. If we use all the compute power on this specific dataset, we roughly need around 3 minutes. However, we have to consider the following:

- There are 33 different TCGA datasets (different cancer types).
- At some point, we would like to work with larger sets of genes.
- We want to perform permutation tests, which add additional computational needs.

Just to run one experiment of this size and 1000 permutation tests, it would take us more than 48 hours. Access to such computer systems can significantly contribute to the accelerated execution of these kinds of experiments. Below, we present several different scenarios and benchmarks that have helped us in understanding and managing these kinds of systems.


# Benchmark Setup

## sccat vs time library
We decided to use `sccat` to extract the runtimes of the python script. What we found is that those times differ significantly from time measured in python script. We suspect this is because `sccat` times probably also consider the time it takes for the resources to be allocated. For example, when we ran a `sbatch` with 1024 tasks and 1 CPU core and printed compute times for all tasks (using the time library in python), we got ranging values from 4 to 5s (per task). If we do `sccat` of that job, the raw elapsed time is around 14s. This was just one example, but we consistently noticed the difference in our experiments.

## slower nodes
We also noticed that the same jobs would have slower runtimes on smaller nodes. In our experiments, we decided to always use the following flag:
```#SBATCH --exclude=wn[051,052,053,061,062,064,065]```

## baseline
To set a baseline for our experiment we measured 3 separate runs on 1, 2, 4, 16, 32, 64 and 128 cores (as this is the maximum CPU logical cores in compute node).

The final config was the following:

``` bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=<number_of_cpus>
#SBATCH --array=1-3
#SBATCH --exclude=wn[051,052,053,061,062,064,065]
#SBATCH --nodes=1
#SBATCH --exclusive
```

Note that we use the `--exclusive` option to reserve a full node for the duration of our computation. The reason for this is that we have observed varying compute times across different jobs with the same configuration. This likely indicates that different processes were sharing resources on a node. We reserved the entire node exclusively for this baseline runs, even though it made queueing more challenging, with wait times extending to days or even weeks at times. When using  `--exclusive` we observed much more consistent results.

![](images/image1.png)

We observe that adding additional cores for the given number of inputs yields diminishing returns after certain point. We suspect that given a larger input size this would not be the case.


We also wanted to compare this results if we only used different number of tasks (average of 5 runs) and one CPU per task:
``` bash
#SBATCH --ntasks=<num_of_tasks>
#SBATCH --cpus-per-task=1
#SBATCH --array=1-5
#SBATCH --exclude=wn[051,052,053,061,062,064,065]
```

![](images/image2.png)

We can observe that there is additional overhead when using tasks versus only CPU cores. But after around 128 tasks this starts paying off.

We noticed that at some point, the overhead of scheduling tasks overshadows the speedups gained from reducing input sizes. This becomes evident when we measure the time within the Python script and compare it to what sccat reports, a discrepancy that is clearly visible. To illustrate this, take a look at the zoomed-in version of the graph above:

![](images/sccat_time.png)

If we measure the time within the Python script and compare it to what sccat reports, this discrepancy is indeed clearly visible.


## 1000 permutations
We used array jobs to perform experiments with 1,000 permutations. We used the following configuration:

``` bash
#!/bin/bash
#SBATCH --time=00:00:30
#SBATCH --job-name=perm_<num_tasks>-<num_cores>
#SBATCH --output=stdout/%x-%A_%a.out
#SBATCH --error=stderr/%x-%A_%a.err
#SBATCH --ntasks=<num_tasks>
#SBATCH --cpus-per-task=<num_cores>
#SBATCH --ntasks-per-node=<depends_on_config>
#SBATCH --nodes=<depends_on_config>
#SBATCH --array=1-1000
#SBATCH --exclude=wn[051,052,053,061,062,064,065]
```

We use the `SLURM_ARRAY_TASK_ID` environment variable to keep track of permutation tests. That is, when the array task ID is 1, we operate on the original data, and all subsequent IDs represent permutation tests.

We tried several different configurations:
- 1024 tasks, 1 CPU core

    **Job submition time:** 17:12:45 <br>
    **ob end time:** 10:12:32 <br>
    **Duration of a job:** 16:59:47 <br>
    **Avg. runtime of python scripts:** 15.583 seconds<br>

- 16 tasks, 64 CPU cores (2 tasks per node):

    **Job submition time:** 22:33:27 <br>
    **Job end time:** 07:00:44 <br>
    **Duration of a job:** 08:27:17 <br>
    **Avg. runtime of python scripts:** 18.343 seconds

- 32 tasks, 32 CPU cores (4 task per node)

    **Job submition time:** 07:00:44  <br>
    **Job end time:** 23:35:00  <br>
    **Duration of a job:** 16:34:16  <br>
    **Avg. runtime of python script:** 18.348 seconds


We significantly reduced the amount of time required to conduct numerous experiments. Instead of days, we now measure the time in hours. However, these durations appear to vary considerably, depending on how the SLURM scheduler handles our tasks.