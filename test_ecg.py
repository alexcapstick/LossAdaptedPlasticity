# python test_ecg.py -v --seed 2 4 8 16 32 64 128 256 512 1024 --n-jobs 6 --ds 1 & python test_ecg.py -v --seed 2 4 8 16 32 64 128 256 512 1024 --n-jobs 6 --ds 0

# importing required packages and models
import pandas as pd
import torch
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import datetime as dt
import argparse
from experiment_code.models.model_code.resnet import ResNetLearning
import os
from pathlib import Path
from experiment_code.data_utils.dataset_loaders import WrapperDataset, PTB_XL, MemoryDataset
from experiment_code.data_utils.corruption_functions import ECGCorruptor
from experiment_code.data_utils.dataloader_loaders import GroupBatchSampler
from experiment_code.utils.parallel_functions import p_apply
from experiment_code.utils.parallel import ParallelModelling
from experiment_code.testing_utils.testing_functions import auc_precision_recall_curve


# parsing user options
parser = argparse.ArgumentParser(description='Train a chosen model on ECG data')
parser.add_argument('--seed', help='random seed', nargs='+', type=int, default=None)
parser.add_argument('--n-jobs', help='The number of jobs for parallel tasks', type=int, default=1)
parser.add_argument('--ds', help='The depression_strength', type=float, default=1)
parser.add_argument('--axis', help='The axis to apply corruption to. Can be x, y, or both', 
    type=str, default='both')
parser.add_argument('--noise-level', help='The noise level', type=float, default=0.5)
parser.add_argument('--n-corrupt-sources', help='The number of corrupt sources', type=int, default=4)
parser.add_argument('-v', '--verbose', help='Whether to print information as the script runs', action='store_true')
parser.add_argument('--data-dir', help='Directory for the data to be saved and loaded', 
    type=str, default='./data/')
parser.add_argument('--disk-dataset', help='Keep the datasets on disk', action='store_true')
parser.add_argument('--test-dir', help='The directory to save the model test results',
    type=str, default='./outputs/ecg_results/')

args = parser.parse_args()

if not Path(args.test_dir).exists():
    os.makedirs(args.test_dir)


# getting the time to add to the results files
post_fix = dt.datetime.now().strftime('%Y%m%d-%H%M%S')

# printing information about the experiments
if args.verbose:
    print(f'Running with seeds: {args.seed}')
    print(f'Running with n_jobs: {args.n_jobs}')
    print(f'Saving dataset into memory: {not args.disk_dataset}')


# set values for all experiments
lap_n = 25
depression_strength = args.ds
strictness = 0.8
hold_off = 0
batch_size = 64
n_epochs = 40
data_subset = False
sampling_rate = 100 # 500 # 100
corruption_std = 0.2 # 0.1

noise_level = args.noise_level
n_corrupt_sources = args.n_corrupt_sources




########################### getting datasets

device_func = lambda x: x.cuda() if torch.cuda.is_available() else lambda x: x

# wrapping function sends data to GPU if available
train_dataset = WrapperDataset( 
    # the ECG dataset, which will be downloaded 
    # if it does not exist at the given path
    PTB_XL(
        data_path=args.data_dir, 
        train=True, 
        source_name='nurse', 
        sampling_rate=sampling_rate,
        return_sources=True,
        binary=True,
        subset=data_subset,
        ), 
    functions_index=[0,1],
    functions=device_func,
    )

# wrapping function sends data to GPU if available
test_dataset = WrapperDataset(
    # the ECG dataset, which will be downloaded 
    # if it does not exist at the given path
    PTB_XL(
        data_path=args.data_dir, 
        train=False, 
        source_name='nurse', 
        sampling_rate=sampling_rate,
        return_sources=False,
        binary=True,
        subset=data_subset,
        ), 
    functions_index=[0,1],
    functions=device_func,
    )


# load the dataset to memory if required
if not args.disk_dataset:
    train_dataset = MemoryDataset(
        train_dataset,
        now=True, 
        verbose=args.verbose,
        )

    test_dataset = MemoryDataset(
        test_dataset,
        now=True, 
        verbose=args.verbose,
        )

########################### model arguments


model_args = {
    'input_dim': 10*sampling_rate,
    'input_channels': 12,
    'kernel_size': 15,
    'n_output': 2,
    'train_criterion': 'CE',
    'n_epochs': n_epochs,
    'source_fit': True,
    'verbose': False,
    'train_optimizer': {
        'adam_lap': {
            'params': ['all'],
            'lr': 0.001,
            'lap_n': lap_n,
            'depression_strength': depression_strength,
            'depression_function': 'discrete_ranking_std',
            'depression_function_kwargs':{
                'strictness': strictness,
                'hold_off': hold_off,
                }
            },
        },
    }




########################### data loading

# lists to store the data loaders for each model being tested
train_loaders = []
val_loaders = []

# the models and dataloaders are unique to the seeds
for seed in args.seed:

    # randomly choosing which of the sources to be corrupt
    rng = np.random.default_rng(seed=args.seed)
    unique_sources = np.unique(train_dataset.sources)
    corrupt_sources = rng.choice(
        unique_sources, 
        size=n_corrupt_sources, 
        replace=False,
        )

    # corrupting the ECG data with the given arguments and options
    train_dataset_corrupt = ECGCorruptor(
        dataset=train_dataset, 
        corrupt_sources=corrupt_sources, 
        noise_level=noise_level,
        seed=seed,
        axis=args.axis,
        x_noise_std=corruption_std,
        )

    # splitting the data into training and validation sets.
    # This is done after the corruption function to allow the 
    # validation set to be corrupted
    train_dataset_s, val_dataset_s = torch.utils.data.random_split(
        train_dataset_corrupt,
        lengths=[
            int(len(train_dataset_corrupt)*0.75),
            len(train_dataset_corrupt) - int(len(train_dataset_corrupt)*0.75)
            ],
        generator=torch.Generator().manual_seed(seed)
        )

    # after data shuffling, get the sources of the training data
    # and the validation data
    train_sources = [s for d, t, s in train_dataset_s]
    val_sources = [s for d, t, s in val_dataset_s]

    # add the train dataloader to the list of train dataloaders
    train_loaders.append(torch.utils.data.DataLoader(
        train_dataset_s, 
        # sample batches by source, so that 
        # each batch contains a single source
        batch_sampler=GroupBatchSampler(
            group=train_sources,
            seed=seed,
            batch_size=batch_size,
            # upsample sources to ensure the 
            # same number of data points in each source
            upsample=True, 
            )
        ))

    # add the val dataloader to the list of val dataloaders
    val_loaders.append(torch.utils.data.DataLoader(
        val_dataset_s, 
        # sample batches by source, so that 
        # each batch contains a single source
        batch_sampler=GroupBatchSampler(
            group=val_sources,
            seed=seed,
            batch_size=batch_size,
            # upsample sources to ensure the 
            # same number of data points in each source
            upsample=True,
            )
        ))

# build the test loader, which is shared across all seeds
test_loader = torch.utils.data.DataLoader(
    test_dataset, 
    batch_size=batch_size,
    )


########################### model arguments


# allow for the models to be trained in parallel
p_model = ParallelModelling(
    models=[
        ResNetLearning(
            seed=seed, 
            model_name=f'RN-ecg-s_{seed}-ds_{depression_strength}',
            **model_args
            )
        for seed in args.seed
        ],
    n_jobs=args.n_jobs,
    )



########################### model fitting

p_model.fit(
    list__train_loader=train_loaders, 
    list__val_loader=val_loaders, 
    metrics_track=['accuracy'],
    )

########################### model evaluation

# test predictions for each model and the true test values
y_test_pred = p_model.predict(test_loader=test_loader)
y_test_true = torch.concat([y for x, y in test_loader],dim=0)

# val predictions for each model and the true val values
# this function is parallelised since all of the val loaders
# are unique to the seed
y_val_pred = p_model.predict(list__test_loader=val_loaders)
y_val_true = p_apply(
    func=lambda loader: torch.concat([y for _, y, _ in loader],dim=0),
    list__loader=val_loaders,
    verbose=False,
    n_jobs=args.n_jobs,
    )
s_val = p_apply(
    func=lambda loader: torch.concat([s for _, _, s in loader],dim=0),
    list__loader=val_loaders,
    verbose=False,
    n_jobs=args.n_jobs,
    )

# metrics to evaluate the model predictions with
metric_dict = {
    'accuracy_score': accuracy_score,
    'recall_score': recall_score,
    'precision_score': precision_score,
    'f1_score': f1_score,
    }

# metrics to evaluate the model probability outputs with
metric_prob_dict = {
'auc_precision_recall_curve': auc_precision_recall_curve,
    }

# functions that will evaluate the models with the given metrics
# for each source in the validation set
metric_source_calc_func = lambda x,y,s: [
    {   
        'source': source,
        'metric': func_name,
        'value': func(
            x[s == source].cpu().numpy(), 
            torch.max(y[s == source], dim=1)[1].cpu().numpy(),
            ),
        'context': 'validation',
        }
    for source in np.unique(s)
    for func_name, func in metric_dict.items()
    ]
metric_prob_source_calc_func = lambda x,y,s: [
    {   
        'source': source,
        'metric': func_name,
        'value': func(
            x[s == source].cpu().numpy(), 
            y[s == source].cpu().numpy(),
            ),
        'context': 'validation',
        }
    for source in np.unique(s)
    for func_name, func in metric_prob_dict.items()
    ]


# functions that will evaluate the models with the given metrics
# for the test set. This is separated as the test set doesn't contain
# sources.
metric_calc_func = lambda x,y: [
    {  
        'metric': func_name,
        'value': func(
            x.cpu().numpy(), 
            torch.max(y, dim=1)[1].cpu().numpy(),
            ),
        'context': 'test',
        }
    for func_name, func in metric_dict.items()
    ]
metric_prob_calc_func = lambda x,y: [
    {  
        'metric': func_name,
        'value': func(
            x.cpu().numpy(), 
            y.cpu().numpy(),
            ),
        'context': 'test',
        }
    for func_name, func in metric_prob_dict.items()
    ]

# running the evaluation functions
evaluation_args = [
    {
        'func': metric_source_calc_func, 
        'list__x': y_val_true,
        'list__y': y_val_pred,
        'list__s': s_val,
        },

    {
        'func': metric_prob_source_calc_func, 
        'list__x': y_val_true,
        'list__y': y_val_pred,
        'list__s': s_val,
        },

    {
        'func': metric_calc_func, 
        'x': y_test_true,
        'list__y': y_test_pred,
        },

    {
        'func': metric_prob_calc_func, 
        'x': y_test_true,
        'list__y': y_test_pred,
        },
    ]

# parallelisation of the metric calculations 
# over the different lists of predictions
results_list = [
    p_apply(
        verbose=False, 
        n_jobs=args.n_jobs, 
        **kwargs,
        ) 
        for kwargs in evaluation_args
    ]

# collating results into a dataframe
results = pd.DataFrame()
for result_context_metric in results_list:
    for nm, results_metric in enumerate(result_context_metric):
        results_metric = pd.json_normalize(results_metric)
        results_metric['run'] = nm + 1
        results_metric['seed'] = args.seed[nm]
        # finding whether a source was corrupt or not, for val results
        if 'source' in results_metric.columns:
            results_metric['corrupt_source'] = (
                results_metric['source']
                .isin(
                    val_loaders[nm].dataset.dataset.corrupt_sources
                    )
                )
        results = pd.concat([results, results_metric])

# adding the values that were the same for all experiments
results = results.reset_index(drop=True)
results['lap_n'] = 25
results['depression_strength'] = depression_strength
results['strictness'] = strictness
results['hold_off'] = hold_off
results['batch_size'] = batch_size
results['noise_level'] = noise_level
results['n_corrupt_sources'] = n_corrupt_sources


save_path = (
    f"{args.test_dir}"
    + f"ecg-s_{''.join([str(s) for s in args.seed])}"
    + f"-ds_{depression_strength}"
    + f"-{post_fix}"
    + ".csv"
    )

# saving results dataframe as a pickle file
results.to_csv(
    save_path,
    mode='a', 
    header=not Path(save_path).is_file(),
    index=False,
    )
