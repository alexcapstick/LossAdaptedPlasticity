# python test_cifarn_different_noise.py --seed 2 4 8 16 32 64 128 256 512 1024 

import numpy as np
import pandas as pd
import torch
from itertools import product
import tqdm
import os
import argparse
from pathlib import Path
import os
from experiment_code.models.model_code import get_model
from experiment_code.data_utils.dataloader_loaders import SourceBatchSampler 
from experiment_code.data_utils.dataset_loaders import CIFAR10_N
from experiment_code.testing_utils.testing_functions import accuracy_topk

parser = argparse.ArgumentParser(description='Train a convolutional model on CIFAR-10N')
parser.add_argument('--seed', help='Random seed or seeds', nargs='+', type=int, default=None)
parser.add_argument('--data-dir', 
    help='The data path to the CIFAR-10 and CIFAR-10N dataset. '\
            'The CIFAR-10N dataset can be downloaded from http://noisylabels.com/.',
            default='./data/')
parser.add_argument('--test-dir', help='The directory to save the model test results',
    type=str, default='./outputs/cifarn_results/')

args = parser.parse_args()

if not Path(args.test_dir).exists():
    os.makedirs(args.test_dir)

if args.seed is None:
    args.seed = [np.random.randint(0,1e6)]


model_class = get_model(model_config={'model_name': 'Conv3Net'})

hparam_list_list = [
    [1,],
    [0.5],
    #[1, 2, 3, 4,], # n_corrupt_sources # [1,],
    #[0.25, 0.5, 0.75, 1.0,], # noise_level # [0.5],
    [0, 1,], # depression_strength
    [25,], # lap_n
    [0.8,], # strictness
    [0], # hold_off
    ]

hparam_runs = list(product(*hparam_list_list))
hparam_runs.extend([[0,0,0,25,0.8,0]]) # no corruption, nor depression

n_experiments = len(hparam_runs)

# set values for all experiments
batch_size = 128
n_epochs = 2# 25

pbar = tqdm.tqdm(desc='Running Experiments', total=n_experiments*len(args.seed))

for seed in args.seed:
    for hparams in hparam_runs:

        ( 
            n_corrupt_sources, 
            noise_level,
            depression_strength, 
            lap_n, 
            strictness, 
            hold_off,
            ) = hparams

        # constructing dataset -------------------------------
        train_dataset = CIFAR10_N(
            args.data_dir,
            train=True,
            n_sources=10,
            n_corrupt_sources=n_corrupt_sources,
            noise_level=noise_level,
            seed=seed,
            )
        test_dataset = CIFAR10_N(args.data_dir, train=False)

        train_split = [
            int(0.75*len(train_dataset)), 
            len(train_dataset)-int(0.75*len(train_dataset))
            ]

        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, 
            lengths=train_split, 
            generator=torch.Generator().manual_seed(int(seed))
            )

        train_sources = [x[2] for x in train_dataset]
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, 
            batch_sampler=SourceBatchSampler(
                sources=train_sources,
                seed=None,
                batch_size=batch_size,
                )
            )

        val_sources = [x[2] for x in val_dataset]
        val_loader = torch.utils.data.DataLoader(
            dataset=val_dataset, 
            batch_sampler=SourceBatchSampler(
                sources=val_sources,
                seed=None,
                batch_size=batch_size,
                )
            )

        true_label_val = [x[1] for x in val_dataset]
        val_no_source_loader = torch.utils.data.DataLoader(
            dataset=val_dataset,
            batch_size=batch_size,
            shuffle=False,
            )

        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            shuffle=False,
            )
        true_label_test = [x[1] for x in test_dataset]


        # constructing model -------------------------------
        model = model_class(
            **{
                'model_name': f'C3N'\
                    '-cifar10N'\
                    f'-ds_{depression_strength}'\
                    f'-ln_{lap_n}'\
                    f'-stn_{strictness}'\
                    f'-ho_{hold_off}'\
                    f'-ncs_{n_corrupt_sources}'\
                    f'-nl_{noise_level}'\
                    f'-sd_{seed}',
                'input_dim': 32,
                'in_channels': 3,
                'channels': 32,
                'n_out': 10,
                'n_epochs': n_epochs,
                'verbose': True,
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
        )

        
        # experiments -------------------------------
        model.fit(train_loader=train_loader, val_loader=val_loader)
        results_temp = {
            'Seed': [seed]*4,
            'LAP N': [lap_n]*4,
            'Depression Strength': [depression_strength]*4,
            'Strictness': [strictness]*4,
            'Hold Off': [hold_off]*4,
            'Number of Corrupted Sources': [n_corrupt_sources]*4,
            'Noise Level': [noise_level]*4
            }


        # val predictions -------------------------------
        results_temp['Context'] = ['Validation']*2
        outputs_val = model.predict(val_no_source_loader)
        probabilities_val, predictions_val = outputs_val.max(dim=1)
        results_temp['Metric'] = ['Accuracy', 'Top 2 Accuracy']
        accuracy_val_top1, accuracy_val_top2 = accuracy_topk(outputs_val, torch.tensor(true_label_val), topk=(1,2,))
        results_temp['Value'] = [accuracy_val_top1.item(), accuracy_val_top2.item()]


        # test predictions -------------------------------
        results_temp['Context'].extend(['Test']*2)
        outputs_test = model.predict(test_loader)
        probabilities_test, predictions_test = outputs_test.max(dim=1)
        results_temp['Metric'].extend(['Accuracy', 'Top 2 Accuracy'])
        accuracy_test_top1, accuracy_test_top2 = accuracy_topk(outputs_test, torch.tensor(true_label_test), topk=(1,2,))
        results_temp['Value'].extend([accuracy_test_top1.item(), accuracy_test_top2.item()])

        # saving results to file -------------------------------
        output_path = f'{args.test_dir}CIFAR10N_results-seed_{seed}.csv'
        pd.DataFrame(results_temp).to_csv(
            output_path, 
            mode='a', 
            header=not Path(output_path).is_file(),
            index=False,
            )

        model.to('cpu')
        del model
    
        pbar.update(1)
        pbar.refresh()

pbar.close()