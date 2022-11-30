import argparse
import numpy as np
import pandas as pd
import torch
import yaml
from itertools import product
from experiment_code.data_utils.dataset_loaders import get_dataset_function
from experiment_code.data_utils.corruption_manager import SourceCorrupter
from experiment_code.models.model_code import get_model
from experiment_code.testing_utils.testing_functions import accuracy_topk
from experiment_code.utils.utils import update_dict
import random
import os
from pathlib import Path
import tqdm



####### possible arguments #######
parser = argparse.ArgumentParser(description='Train a chosen model with different algorithms')
parser.add_argument('--dataset-name', help='The dataset to use', type=str, default='mnist')
parser.add_argument('--model-name', help='The model Name', required=True)
parser.add_argument('--seed', help='random seed', nargs='+', type=int, default=None)
parser.add_argument('--data-dir', help='Directory for the data to be saved and loaded', 
    type=str, default='../../data/')
parser.add_argument('-v', '--verbose', help='Whether to print information as the script runs',
    action='store_true')
parser.add_argument('--config-file', help='The config file containing the model parameters and training methods',
    type=str, default='./synthetic_config.yaml')
parser.add_argument('--device', help='Device to run the models on.',
    type=str, default='auto')
parser.add_argument('--n-sources', help='The number of sources used in the training data',
    nargs='+', type=int, default=None)
parser.add_argument('--n-corrupt-sources', help='The number of corrupt sources used in the training data',
    nargs='+', type=int, default=None)
parser.add_argument('--source-size', help='The number of data points in each source batch',
    nargs='+', type=int, default=None)
parser.add_argument('--lr', help='The learning rate of the training',
    nargs='+', type=float, default=None)
parser.add_argument('--depression-strength', help='The depression strength of the training',
    nargs='+', type=float, default=None)
parser.add_argument('--lap-n', help='The number of previous losses to use in the depression ranking',
    nargs='+', type=int, default=None)
parser.add_argument('--strictness', help='The strictness used when calculating which sources to '\
    'to apply depression to. It is used in mean+strictness*std.',
    nargs='+', type=float, default=None)
parser.add_argument('--hold-off', help='The number of steps to hold off on applying depression '\
    'after full history is built.',
    nargs='+', type=int, default=None)
parser.add_argument('--n-epochs', help='The number of epochs to train for',
    nargs='+', type=int, default=None)
parser.add_argument('--corruption-level', help='How much corruption in a source. You should '\
    'pass as many floats as the number of corrupt sources.',
    nargs='+', type=float, default=None)
parser.add_argument('--test-dir', help='The directory to save the model test results',
    type=str, default='./outputs/different_n_p/')


args = parser.parse_args()

if not Path(args.test_dir).exists():
    os.makedirs(args.test_dir)

# setting the device for running the experiment
device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device

# the hparam names that will be used when saving the models and results
hparam_short_name = {
    'n_sources': 'ns', 
    'n_corrupt_sources': 'ncs', 
    'source_size': 'ssize',
    'lr': 'lr',
    'depression_strength': 'ds',
    'strictness': 'stns',
    'lap_n': 'lap_n',
    'hold_off': 'ho',
    'n_epochs': 'ne',
    }

hparam_list_list = []
for arg_name in hparam_short_name.keys():
    hparam_list = getattr(args, arg_name)
    if hparam_list is None:
        continue
    else:
        hparam_list_list.append(
            [(arg_name, hparam) for hparam in hparam_list]
        )

# all hparam combinations stored here
hparam_runs = list(product(*hparam_list_list))

n_experiments = len(hparam_runs)
pbar = tqdm.tqdm(desc='Running Experiments', total=n_experiments*len(args.seed))

if args.seed is None:
    args.seed = [np.random.randint(0,1e6)]

args.seed_list = args.seed
corruption_type = args.model_name.split('-')[1]

if args.verbose:
    print(' --------- Running {} experiments  --------- '.format(len(args.seed_list)))

# loop over the hparams
for hparams in hparam_runs:
    for seed in args.seed_list:
        args.seed = seed

        ####### Running with one of the seeds #######
        if args.verbose:
            print(' --------- Running with seed {} --------- '.format(args.seed))

        # setting options for reproducibility
        random.seed(args.seed)     # python random generator
        np.random.seed(args.seed)  # numpy random generator
        torch.manual_seed(args.seed) # torch seed
        torch.cuda.manual_seed_all(args.seed) # cuda seed
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        ####### model config from file #######
        model_config = yaml.load(open(args.config_file, 'r'), Loader=yaml.FullLoader)[args.model_name]

        # this updates the model config with the options given in the arguments
        for hparam in hparams:
            param_name, param = hparam
            if param_name in ['n_epochs']:
                update_dict(model_config['model_params'], {param_name: param})
            elif param_name in ['n_sources', 'n_corrupt_sources', 'source_size']:
                update_dict(model_config, {'train_params': {param_name: param}})
            elif param_name in ['lr', 'depression_strength', 'lap_n']:
                for optim_name in model_config['model_params']['train_optimizer'].keys():
                    update_dict(model_config, {'model_params': {'train_optimizer': {optim_name: {param_name: param}}}})
            elif param_name in ['strictness', 'hold_off']:
                for optim_name in model_config['model_params']['train_optimizer'].keys():
                    update_dict(model_config, {
                        'model_params': {
                            'train_optimizer': {
                                optim_name: {
                                    'depression_function_kwargs': {
                                        param_name: param
                        }}}}})
            else:
                raise NotImplementedError('Please point the hparam to the correct model_config param.')
            
            # adding hparams to model_name
            if param_name in hparam_short_name:
                model_config['model_name'] += '-{}_{}'.format(hparam_short_name[param_name], param)
            else:
                model_config['model_name'] += '-{}_{}'.format(param_name, param)


        ####### collating training data #######

        if args.verbose:
            print(' --------- Extracting the data --------- ')

        dataset_func = get_dataset_function(args.dataset_name)
        (
            train_dataset, test_dataset, 
            train_dataset_targets, test_targets) = dataset_func(path=args.data_dir, return_targets=True)

        train_loaders = []

        train_dataset = SourceCorrupter(
            dataset=train_dataset, 
            n_sources=model_config['train_params']['n_sources'], 
            source_size=model_config['train_params']['source_size'], 
            n_corrupt_sources=model_config['train_params']['n_corrupt_sources'], 
            corruption_function=model_config['train_params']['corruption_function'],
            corruption_function_kwargs=model_config['train_params']['corruption_function_kwargs'],
            shuffle_sources=False, 
            shuffle_dataset=True, 
            return_sources=True,
            return_bool_source=False,
            seed = args.seed,
            corrupt_for_n_steps=None,
            corruption_level=args.corruption_level,
            )

        split_prop = model_config['train_params']['validation']['train_split']

        train_split = [
            int(split_prop*len(train_dataset)), 
            len(train_dataset)-int(split_prop*len(train_dataset))
            ]

        train_idx = np.arange(
            int(
                len(train_dataset)
                *split_prop
                //train_dataset.source_size
                )
            *train_dataset.source_size
            )

        val_idx =  np.arange(
            int(
                len(train_dataset)
                *split_prop
                //train_dataset.source_size
                )
            *train_dataset.source_size, 
            len(train_dataset)
            )

        train_dataset_out = torch.utils.data.Subset(
            train_dataset, 
            train_idx
            )
        val_dataset_out = torch.utils.data.Subset(
            train_dataset, 
            val_idx,
            )

        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset_out, 
            shuffle=False,
            batch_size=model_config['train_params']['source_size'],
            )

        val_loader = torch.utils.data.DataLoader(
            dataset=val_dataset_out, 
            shuffle=False,
            batch_size=model_config['train_params']['source_size'],
            )

        train_loaders.append(train_loader)
        train_loaders.append(val_loader)

        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            shuffle=False,
            batch_size=model_config['train_params']['source_size'],
        )


        if args.verbose:
            print(' --------- Getting the model --------- ')
        
        ####### getting model #######
        model_class = get_model(model_config)

        # if semi-supervised training
        ae_train = model_config['train_params']['ae'] if 'ae' in model_config['train_params'] else False

        model_args = model_config['model_params']
        save_args = model_config['save_params']
        model_config['model_name'] += '-sd_{}-dst_{}'.format(args.seed, args.dataset_name)
        model_config['model_name'] = (model_config
            ['model_name']
            .replace('Conv3Net', 'C3N')
            .replace('-drstd', '')
            )
        # building models from parameters
        if ae_train:
            model = model_class(
                seed=args.seed, 
                device=device, 
                verbose=args.verbose, 
                model_name=model_config['model_name'],
                ae_source_fit=model_config['train_params']['ae_source_fit'],
                clf_source_fit=model_config['train_params']['clf_source_fit'],
                **save_args, 
                **model_args
                )
        else:
            model = model_class(
                seed=args.seed, 
                device=device, 
                verbose=args.verbose, 
                model_name=model_config['model_name'],
                source_fit=model_config['train_params']['source_fit'],
                **save_args, 
                **model_args
                )

        if args.verbose:
            print(' --------- Model --------- ')
            print(model)

        if args.verbose:
            print(' --------- Config --------- ')
            print(args, model_config)

        ####### running training #######
        if args.verbose:
            print(' --------- Training --------- ')


        # fitting the models
        if ae_train:
            if len(train_loaders)>2:
                model.fit(
                    train_labelled_loader=train_loaders[0], 
                    train_unlabelled_loader=train_loaders[1],
                    val_labelled_loader=train_loaders[2],
                    val_unlabelled_loader=train_loaders[3],
                    )
            elif len(train_loaders)==2:    
                model.fit(
                    train_labelled_loader=train_loaders[0], 
                    train_unlabelled_loader=train_loaders[1],
                    )
            else:
                raise TypeError('Wrong number of training loaders for semi-supverised learning.')
        else:
            if len(train_loaders)>1:
                model.fit(train_loader=train_loaders[0], val_loader=train_loaders[1])
            elif len(train_loaders)==1:    
                model.fit(train_loaders[0])
            else:
                raise TypeError('Wrong number of training loaders for traditional learning.')

        corrupt_sources = train_loaders[1].dataset.dataset.corrupt_sources
        corruption_level = train_loaders[1].dataset.dataset.corruption_level # dictionary of corruption level
        n_sources = train_loaders[1].dataset.dataset.n_sources
        corruption_level.update(
            {
                source: 0.0 
                for source in range(n_sources) 
                if not source in corruption_level
                }
            )

        val_target = []
        val_source = []
        for _, t, s in train_loaders[1]:
            val_target.append(t.numpy().reshape(-1))
            val_source.append(s.numpy().reshape(-1))

        val_target = np.concatenate(val_target, axis=0)
        val_source = np.concatenate(val_source, axis=0)


        test_target = []
        for _, t in test_loader:
            test_target.append(t.numpy().reshape(-1))

        test_target = np.concatenate(test_target, axis=0)

        results_temp = {
            'Corruption Type': corruption_type,
            'Seed': seed,
            'LAP N': model_config['model_params']['train_optimizer']['adam_lap']['lap_n'],
            'Depression Strength': model_config['model_params']['train_optimizer']['adam_lap']['depression_strength'],
            'Strictness': model_config['model_params']['train_optimizer']['adam_lap']['depression_function_kwargs']['strictness'],
            'Hold Off': model_config['model_params']['train_optimizer']['adam_lap']['depression_function_kwargs']['hold_off'],
            'Number of Corrupted Sources': model_config['train_params']['n_corrupt_sources'],
            'metrics': [],
            'Corruption level': str(
                [
                    value for value in corruption_level.values() 
                    if value != 0.0
                    ]
                )
            }

        # val predictions by source -------------------------------

        outputs_val = model.predict(train_loaders[1])
        probabilities_val, predictions_val = outputs_val.max(dim=1)

        val_source_idx = np.argwhere(val_source.reshape(-1,1) == np.arange(n_sources))

        for source in range(n_sources):
            idx = val_source_idx[val_source_idx[:,1] == source][:,0]
            
            val_accuracy_source, val_top_2_accuracy_source = accuracy_topk(
                outputs_val[idx], 
                torch.tensor(val_target[idx]), 
                topk=(1,2,),
                )
            results_temp['metrics'].extend([
                {
                'Metric': f'accuracy_source_{source}_corrupt_{source in corrupt_sources}_level_{int(corruption_level[source]*100)}',
                'Value': val_accuracy_source.item(),
                },
                {'Metric': f'top_2_accuracy_source_{source}_corrupt_{source in corrupt_sources}_level_{int(corruption_level[source]*100)}',
                'Value': val_top_2_accuracy_source.item(),
                }
                ])

        val_accuracy, val_top_2_accuracy = accuracy_topk(outputs_val, torch.tensor(val_target), topk=(1,2))

        results_temp['metrics'].extend([
            {
            'Metric': f'accuracy',
            'Value': val_accuracy.item(),
            },
            {'Metric': f'top_2_accuracy',
            'Value': val_top_2_accuracy.item(),
            }
            ])


        outputs_test = model.predict(test_loader)
        probabilities_test, predictions_test = outputs_test.max(dim=1)
        test_accuracy, test_top_2_accuracy = accuracy_topk(outputs_test, torch.tensor(test_target), topk=(1,2))

        results_temp['metrics'].extend([
            {
            'Metric': f'test_accuracy',
            'Value': test_accuracy.item(),
            },
            {'Metric': f'test_top_2_accuracy',
            'Value': test_top_2_accuracy.item(),
            }
            ])


        results_temp = pd.json_normalize(
            results_temp, 
            record_path='metrics',
            meta=[
                'Corruption Type',
                'Seed',
                'LAP N',
                'Depression Strength',
                'Strictness',
                'Hold Off',
                'Number of Corrupted Sources',
                'Corruption level',
                ]
            )

        if args.verbose:
            print('######################')
            print(results_temp)

        # saving results to file -------------------------------
        upper_dataset_name = f'{args.dataset_name}'.upper()
        output_path = f'{args.test_dir}{upper_dataset_name}_results_{seed}.csv'
        results_temp.to_csv(
            output_path, 
            mode= 'a', 
            header=not Path(output_path).is_file(),
            index=False,
            )

        model.to('cpu')
        del model
    
        pbar.update(1)
        pbar.refresh()

pbar.close()