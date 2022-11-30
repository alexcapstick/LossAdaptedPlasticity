import argparse
import random
import numpy as np
import pandas as pd
import torch
import yaml
from itertools import product
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from experiment_code.utils.utils import update_dict
from experiment_code.data_utils.dataloader_loaders import get_train_data
from experiment_code.models.model_code import get_model
from experiment_code.data_utils.dataloader_loaders import get_test_data
from experiment_code.testing_utils.testing_functions import accuracy_topk
import os
from pathlib import Path


####### possible arguments #######
parser = argparse.ArgumentParser(description='Train a chosen model with different algorithms')
parser.add_argument('--dataset-name', help='The dataset to use', type=str, default='mnist')
parser.add_argument('--model-name', help='The model Name', required=True)
parser.add_argument('--seed', help='random seed', nargs='+', type=int, default=None)
parser.add_argument('--data-dir', help='Directory for the data to be saved and loaded', 
    type=str, default='./data/')
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
parser.add_argument('--test-method', help='The testing method', type=str, default='traditional')
parser.add_argument('--test-dir', help='The directory to save the model test results',
    type=str, default='./outputs/synthetic_results/')

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

if args.seed is None:
    args.seed = [np.random.randint(0,1e6)]

args.seed_list = args.seed

if args.verbose:
    print(' --------- Running {} experiments  --------- '.format(len(args.seed_list)))

results = pd.DataFrame()

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
                                        }
                        }}}})
            else:
                raise NotImplementedError('Please point the hparam to the correct model_config param.')
            
            # adding hparams to model_name
            if param_name in hparam_short_name:
                model_config['model_name'] += '-{}_{}'.format(hparam_short_name[param_name], param)
            else:
                model_config['model_name'] += '-{}_{}'.format(param_name, param)

        if args.verbose:
            print(' --------- Extracting the data --------- ')

        ####### collating training data #######
        train_loaders = get_train_data(args, model_config)

        if args.verbose:
            print(' --------- Getting the model --------- ')
        
        ####### getting model #######
        model_class = get_model(model_config)

        # if semi-supervised training
        ae_train = model_config['train_params']['ae'] if 'ae' in model_config['train_params'] else False

        model_args = model_config['model_params']
        save_args = model_config['save_params']
        model_config['model_name'] += '-seed_{}-dataset_{}'.format(args.seed, args.dataset_name)

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
                **model_args)
        else:
            model = model_class(
                seed=args.seed, 
                device=device, 
                verbose=args.verbose, 
                model_name=model_config['model_name'],
                source_fit=model_config['train_params']['source_fit'],
                **save_args, 
                **model_args)

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
                model.fit(train_labelled_loader=train_loaders[0], 
                            train_unlabelled_loader=train_loaders[1],
                            val_labelled_loader=train_loaders[2],
                            val_unlabelled_loader=train_loaders[3],
                            )
            elif len(train_loaders)==2:    
                model.fit(train_labelled_loader=train_loaders[0], 
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

        #### testing

        # loading the test config
        test_config = yaml.load(open(args.config_file, 'r'), Loader=yaml.FullLoader)['testing-procedures'][args.test_method]
        if args.verbose: print('Testing config:', test_config)
        
        test_loader, test_targets = get_test_data(args, test_config=test_config)

        model_test_name  = model_config['model_name']
        output = model.predict(test_loader=test_loader)
        confidence, predictions = output.max(dim=1)
        if len(torch.unique(test_targets)) > 10:
            accuracy_top1, accuracy_top2, accuracy_top5 = accuracy_topk(output, test_targets, topk=(1,2,5))
            results_temp = {
                'Run': [model_test_name]*3,
                'Metric': ['Accuracy', 'Top 2 Accuracy', 'Top 5 Accuracy'],
                'Value': [accuracy_top1.item(), accuracy_top2.item(), accuracy_top5.item()],
                }
        else:
            accuracy_top1, accuracy_top2 = accuracy_topk(output, test_targets, topk=(1,2,))
            results_temp = {
                'Run': [model_test_name]*2,
                'Metric': ['Accuracy', 'Top 2 Accuracy'],
                'Value': [accuracy_top1.item(), accuracy_top2.item()],
                }

        if len(torch.unique(test_targets)) == 2:

            recall = recall_score(test_targets, predictions)
            precision = precision_score(test_targets, predictions)
            f1 = f1_score(test_targets, predictions)
            results_temp['Run'].extend([model_test_name]*3)
            results_temp['Metric'].extend(['Recall', 'Precision', 'F1'])
            results_temp['Value'].extend([recall, precision, f1])

        # collate and save results
        results = pd.concat([results, pd.DataFrame(results_temp)])

save_path = (
    args.test_dir 
    + args.model_name 
    + '-' 
    + args.dataset_name 
    + '-results.csv'
    )

results.to_csv(
    save_path,
    header=not Path(save_path).is_file(),
    mode='a',
    index=False
    )