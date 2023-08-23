import os
import time
import argparse
import numpy as np
import pandas as pd

from collections import namedtuple
from joblib import Parallel, delayed

# import psutil
# from multiprocessing import Pool #, cpu_count, current_process


slurm_proc_id = int(os.environ.get('SLURM_PROCID'))
cpus_per_task = int(os.environ.get('SLURM_CPUS_PER_TASK'))
num_of_tasks = int(os.environ.get('SLURM_NTASKS'))
array_task_id = os.environ.get("SLURM_ARRAY_TASK_ID")


Results = namedtuple(
    'Results',
    [
        'feature1_rmst',
        'feature2_rmst',
        'additive_rmst',
        'additive_interaction_score',
        'competing_rmst',
        'competing_interaction_score',
        'xor_rmst',
        'xor_interaction_score',
    ],
)


def fit_KM_sequence(sample_indicator: np.ndarray, event_indicator: np.ndarray):
    n_series = np.cumsum(sample_indicator[::-1])[::-1]
    return np.append(
        [1], np.cumprod((n_series - event_indicator * sample_indicator) / n_series)
    )


def difference_RMST(km1: np.ndarray, km2: np.ndarray, time_values: np.ndarray):
    LEN = len(time_values)  # the last value is the maximal (resticted at) time
    dt = np.ediff1d(time_values, to_begin=time_values[0])
    return np.abs(np.sum((km1[:LEN] - km2[:LEN]) * dt))


def rmst_diff_median_split(feature_values, sorted_time, sorted_events, TIME_LIMIT):
    cutoff = np.median(feature_values)
    strata = (feature_values > cutoff).astype(bool)
    time_limit = min(sorted_time[strata].max(), sorted_time[~strata].max())

    if TIME_LIMIT <= time_limit:
        time_limit = TIME_LIMIT

    km1 = fit_KM_sequence(strata, sorted_events)
    km2 = fit_KM_sequence(~strata, sorted_events)
    dif_rstm = difference_RMST(
        km1, km2, list(sorted_time[sorted_time <= time_limit]) + [time_limit]
    )
    return dif_rstm



def compute_interactions(feature1, feature2, time_col, event_col):
    TIME_LIMIT = np.percentile(time_col, 75)

    feature1_rmst = np.round(
        rmst_diff_median_split(feature1, time_col, event_col, TIME_LIMIT), 4
    )
    feature2_rmst = np.round(
        rmst_diff_median_split(feature2, time_col, event_col, TIME_LIMIT), 4
    )
    max_rmst_no_interaction = np.max([feature1_rmst, feature2_rmst])

    # interaction terms
    xor_rmst = np.round(
        rmst_diff_median_split(feature1 * feature2, time_col, event_col, TIME_LIMIT), 4
    )
    additive_rmst = np.round(
        rmst_diff_median_split(feature1 + feature2, time_col, event_col, TIME_LIMIT), 4
    )
    competing_rmst = np.round(
        rmst_diff_median_split(feature1 - feature2, time_col, event_col, TIME_LIMIT), 4
    )

    xor_interaction_score = np.round(xor_rmst - max_rmst_no_interaction, 4)
    additive_interaction_score = np.round(additive_rmst - max_rmst_no_interaction, 4)
    competing_interaction_score = np.round(competing_rmst - max_rmst_no_interaction, 4)

    return Results(
        feature1_rmst,
        feature2_rmst,
        additive_rmst,
        additive_interaction_score,
        competing_rmst,
        competing_interaction_score,
        xor_rmst,
        xor_interaction_score,
    )


def worker(
    feature1_label,
    feature1_values,
    feature2_label,
    feature2_values,
    time_col,
    event_col
):
    results = compute_interactions(feature1_values, feature2_values, time_col, event_col)
    # f1, f2, f1_f2, interaction_score = compute_interactions(feature1_values, feature2_values, time_col, event_col, type)
    return feature1_label, feature2_label, *results  # f1, f2, f1_f2, interaction_score


if __name__ == '__main__':
    start_time = time.time()

    # Define the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--tcga_project", help="Name of TCGA project", required=True)
    parser.add_argument("--input_data", help="Expression data", required=True)
    parser.add_argument("--input_genes", help="Gene inputs", required=True)
    # parser.add_argument("--input_size", help="Size of genes on input", required=False)

    # Parse the command line arguments
    args = parser.parse_args()
    tcga_project = args.tcga_project
    input_data = args.input_data
    input_genes = args.input_genes

    # df = pd.read_parquet(input_data)
    df = pd.read_csv(input_data)
    
    # permute the data if array_task_id is not None and array_task_id != 1
    if array_task_id is not None and array_task_id != 1:
        time_event_df = df[['time', 'event']]
        shuffled_df = df.drop(columns=['time', 'event']).sample(frac=1, random_state=int(array_task_id)).reset_index(drop=True)
        df = pd.concat([shuffled_df, time_event_df], axis=1)

    genes = pd.read_csv(input_genes, header=None)

    if num_of_tasks == 1:
        genes = genes.values
    else:
        genes = np.array_split(genes, num_of_tasks)[slurm_proc_id].values
    
    if cpus_per_task == 1:
        parallel_kwargs = {'return_as': 'generator'}
    else:
        parallel_kwargs = {'n_jobs': cpus_per_task, 'return_as': 'generator', 'batch_size': len(genes)//cpus_per_task, 'pre_dispatch': 'all'}


    # run the parallel computation
    parallel = Parallel(**parallel_kwargs)
    results = parallel(
        delayed(worker)(g1, df[g1], g2, df[g2], df['time'], df['event'])
        for g1, g2 in genes
    )

    # save results to csv
    df_results = pd.DataFrame(
        results,
        columns=[
            'feature1',
            'feature2',
            'feature1_rmst',
            'feature2_rmst',
            'additive_rmst',
            'additive_interaction_score',
            'competing_rmst',
            'competing_interaction_score',
            'xor_rmst',
            'xor_interaction_score',
        ],
    )

    # results_dir = f'results/{tcga_project}/'
    # os.makedirs(results_dir, exist_ok=True)
    # df_results.to_csv(f'{results_dir}/task_{slurm_proc_id}.csv', index=False)

    end_time = time.time()
    print(
        f"Processing all pairs on task with"
        f"id {slurm_proc_id} took {end_time - start_time} seconds."
    )
