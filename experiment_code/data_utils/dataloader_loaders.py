import torch
import numpy as np
from .dataset_loaders import *
from .corruption_manager import SourceCorrupter
import typing
import imblearn



####### samplers


class SourceBatchSampler(torch.utils.data.Sampler):
    def __init__(self, sources, seed = None, batch_size=20):
        '''
        A pytorch batch sampler that returns a batch of samples with 
        that same source.
        
        
        Arguments
        ---------
        
        - sources:  
            The sources of the data points. This should be
            the same length as the data set that is to be
            sampled.
        
        - seed: int (optional):
            Random seed for source order shuffling and 
            shuffling of points in each batch.
            Defaults to :code:`None`.
        
        - batch_size: int`, (optional):
            The size of each batch. Each batch
            will be smaller than or equal in 
            size to this value.
            Defaults to :code:`20`.
        
        
        '''

        sources = np.asarray(sources)
        sources_unique, sources_counts = np.unique(sources, return_counts=True)
        source_batches = np.ceil(sources_counts/batch_size).astype(int)
        rng = np.random.default_rng(seed)
        rng = np.random.default_rng(rng.integers(3))
        self.out = -1*np.ones((np.sum(source_batches), batch_size))
        source_order = rng.permutation(np.repeat(sources_unique, source_batches))

        for source in sources_unique:
            source_idx = np.argwhere(sources == source).reshape(-1)
            rng.shuffle(source_idx)
            out_temp = self.out[source_order==source].reshape(-1)
            out_temp[:len(source_idx)] = source_idx
            out_temp = out_temp.reshape(-1, batch_size)
            rng.shuffle(out_temp, axis=0)
            self.out[source_order==source] = out_temp
            rng = np.random.default_rng(rng.integers(3))
        
        self.out = [list(batch[batch != -1].astype(int)) for batch in self.out]

        return 


    def __iter__(self):
        return iter(self.out)
    
    def __len__(self):
        return len(self.out)




###### custom datasets

class MyData(torch.utils.data.Dataset):
    def __init__(self, *inputs):
        self.inputs = inputs
        
    def __getitem__(self,index):
        return [x[index] for x in self.inputs]
    
    def __len__(self):
        return len(self.inputs[0])





###### train dataloaders

def get_train_data(args, model_config):
    '''
    This function returns the training data associated 
    with the information in :code:`args` and :code:`model_config`.
    
    
    Arguments
    ---------
    
    - args:  
        This should have the attributes:
            - seed: The seed for random operations.
            - dataset_name: The dataset to use. The function
            that gets the data for this dataset should be called 
            :code:`get_[dataset_name]`.
            - data_dir: The directory that contains the data 
            folders.

    - model_config: dict: 
        This should have the key 'train_params', containing:
            - 'train_method': The type of training to be done.
            This is either :code:`'traditional source'` or 
            :code:`'traditional'`.
            - 'return_bool_source': Whether to return source
            names or boolean values dictating whether a source
            is corrupt or not. :code:`True` gives the latter.
            - 'n_sources': The number of sources
            for each dataset to be broken into.
            - 'source_size'` or :code:`'batch_size':
            The size of the batches used in traditional source and 
            traditional training respectively.
            - 'n_corrupt_sources': The number of corrupt
            sources used in traditional source training.
            - 'corruption_function'` and 
            :code:`'corruption_function_kwargs': Used in the corruption 
            when traditional source training. See :code:`SourceCorrupter`
            for more information.
            - 'corrupt_for_n_steps': The number of steps to
            corrupt for. See :code:`SourceCorrupter`
            for more information.
            - 'validation': The options for the validation 
            set:
                - 'do_val': Boolean value dictating whether
                to create a validation set.
                - 'train_split': The proportion of the data
                used for the training set.
                - 'corrupt': Whether to corrupt the validation
                data too.
    

    Returns
    --------
    
    - trainloaders: list` of :code:`torch.utils.data.DataLoader: 
        The list of data loaders, containing the training data. If 
        :code:`model_config['train_params']['validation']['do_val']=True`
        then this list also contains the validation loader.
    
    
    '''

    # get the data sets
    func_name = 'get_{}'.format(args.dataset_name)
    train_dataset, _, train_dataset_targets, _ = globals()[func_name](path=args.data_dir, return_targets=True)

    # produce the data loaders from the datasets
    if args.verbose:
        print(' --------- Producing the data loaders --------- ')


    trainloaders = []

    # if validation is specified
    try:
        do_val = model_config['train_params']['validation']['do_val']
    except KeyError:
        do_val = False

    if do_val:
        split_prop = model_config['train_params']['validation']['train_split']
        train_split = [int(split_prop*len(train_dataset)), 
                        len(train_dataset)-int(split_prop*len(train_dataset))]
        train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, 
                                                                    lengths=train_split, 
                                                                    generator=torch.Generator().manual_seed(args.seed))

    if model_config['train_params']['train_method'] == 'traditional':

        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, 
            shuffle=True, 
            batch_size=model_config['train_params']['batch_size'],
            )
        
        trainloaders.append(train_loader)

        if do_val:
            val_loader = torch.utils.data.DataLoader(dataset=val_dataset, 
                                                                batch_size=model_config['train_params']['batch_size'],
                                                        )
            trainloaders.append(val_loader)

    elif model_config['train_params']['train_method'] == 'traditional source':
        
        return_sources = (model_config['train_params']['return_sources'] 
                            if 'return_sources' in model_config['train_params'] else True)
        
        return_bool_source = (model_config['train_params']['return_bool_source'] 
                            if 'return_bool_source' in model_config['train_params'] else False)

        corrupt_for_n_steps = (model_config['train_params']['corrupt_for_n_steps'] 
                            if 'corrupt_for_n_steps' in model_config['train_params'] else None)

        # compiling train data
        train_dataset = SourceCorrupter(train_dataset, 
                                        n_sources=model_config['train_params']['n_sources'], 
                                        source_size=model_config['train_params']['source_size'], 
                                        n_corrupt_sources=model_config['train_params']['n_corrupt_sources'], 
                                        corruption_function=model_config['train_params']['corruption_function'],
                                        corruption_function_kwargs=model_config['train_params']['corruption_function_kwargs'],
                                        shuffle_sources=False, 
                                        shuffle_dataset=True, 
                                        return_sources=return_sources,
                                        return_bool_source=return_bool_source,
                                        seed = args.seed,
                                        corrupt_for_n_steps=corrupt_for_n_steps,
                                        )

        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                                    shuffle=False,
                                                    batch_size=model_config['train_params']['source_size'])
        trainloaders.append(train_loader)

        # if validation data too, compile that here
        if do_val:
            val_n_corrupt_sources = (model_config['train_params']['n_corrupt_sources'] 
                                        if model_config['train_params']['validation']['corrupt'] else 0)
            val_dataset = SourceCorrupter(val_dataset, 
                                    n_sources=model_config['train_params']['n_sources'], 
                                    source_size=model_config['train_params']['source_size'], 
                                    n_corrupt_sources=val_n_corrupt_sources, 
                                    corruption_function=model_config['train_params']['corruption_function'],
                                    corruption_function_kwargs=model_config['train_params']['corruption_function_kwargs'],
                                    shuffle_sources=False, 
                                    shuffle_dataset=True, 
                                    return_sources=return_sources,
                                    return_bool_source=return_bool_source,
                                    seed = args.seed,
                                    )
            val_loader = torch.utils.data.DataLoader(dataset=val_dataset, 
                                                        shuffle=False,
                                                        batch_size=model_config['train_params']['source_size'])
            trainloaders.append(val_loader)

    else:
        raise NotImplementedError("Please use 'semi supervised source', 'traditional' or 'traditional source'")

    return trainloaders










###### test dataloaders


def get_test_data(args, test_config):
    '''
    This function gets the test data based on 
    the arguments given.
    
    
    
    Arguments
    ---------
    
    - args:  
        This should have the attributes:
            - dataset_name: The dataset to use. The function
            that gets the data for this dataset should be called 
            :code:`get_[dataset_name]`.
            - data_dir: The directory that contains the data 
            folders.
    
    - test_config: dict: 
        This should be a dictionary containing the following:
            - 'test_method': The type of training to be done.
            This should be :code:`'traditional'`.
            - 'batch_size': This is the batch size 
            to be used when testing. If it is not given then 
            the length of the testing set is used.
    
    
    
    Returns
    --------
    
    - test_loader: torch.utils.data.DataLoader: 
        A dataloader containing the test data.
    
    - test_dataset_targets: torch.tensor: 
        The targets for the data in the dataloader.

    '''


    func_name = 'get_{}'.format(args.dataset_name)
    _, test_dataset, _, test_dataset_targets = globals()[func_name](path=args.data_dir, return_targets=True)
    if args.verbose:
        print(' --------- Producing the data loaders --------- ')

    if test_config['test_method'] == 'traditional':
        
        if 'batch_size' in test_config:
            batch_size = test_config['batch_size']
        else: 
            batch_size = len(test_dataset)

        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                    shuffle=False,
                                                    batch_size=batch_size)  

    else:
        raise NotImplementedError('Please use test_method=traditional')

    return test_loader, test_dataset_targets




class GroupBatchSampler(torch.utils.data.Sampler):
    def __init__(
        self, 
        group:typing.Union[np.ndarray, typing.List[typing.Any]], 
        seed:typing.Union[None, int]=None, 
        batch_size:int=20, 
        upsample:typing.Union[bool, typing.Dict[typing.Any, int]]=False,
        ):
        '''
        A pytorch batch sampler that returns a batch of samples with 
        that same group.

        Examples
        ---------

        The following will batch the training dataset
        into batches that contains single group, given 
        by the :code:`group` argument

        .. code-block::

            >>> dl = torch.utils.data.DataLoader(
            ...     train_dataset, 
            ...     batch_sampler=GroupBatchSampler(
            ...         group=train_group,
            ...         seed=seed,
            ...         batch_size=64,
            ...         )
            ...     )
        
        
        Arguments
        ---------
        
        - group: typing.Union[np.ndarray, typing.List[typing.Any]]:
            The group of the data points. This should be
            the same length as the data set that is to be
            sampled.
        
        - seed: int (optional):
            Random seed for group order shuffling and 
            shuffling of points in each batch.
            Defaults to :code:`None`.
        
        - batch_size: int, (optional):
            The size of each batch. Each batch
            will be smaller than or equal in 
            size to this value.
            Defaults to :code:`20`.
        
        - upsample: typing.Union[bool, typing.Dict[typing.Any, int]], (optional):
            Whether to upsample the smaller groups,
            so that all groups have the same size.
            Defaults to :code:`False`.
        
        
        '''

        rng = np.random.default_rng(seed)
        
        group = np.asarray(group)

        upsample_bool = upsample if type(upsample) == bool else True

        if upsample_bool:
            upsample_idx, \
                group = imblearn.over_sampling.RandomOverSampler(
                    sampling_strategy='all' if type(upsample) == bool else upsample,
                    random_state=rng.integers(1e9),
                    ).fit_resample(
                        np.arange(len(group)).reshape(-1,1), 
                        group
                        )
            upsample_idx = upsample_idx.reshape(-1)

        group_unique, group_counts = np.unique(group, return_counts=True)
        group_batches = (
            np.repeat(
                np.ceil(
                    np.max(group_counts)/batch_size
                    ).astype(int), 
                len(group_unique)) 
            if upsample 
            else np.ceil(group_counts/batch_size).astype(int)
            )
        rng = np.random.default_rng(
            rng.integers(low=0, high=1e9, size=(4,))
            )
        n_batches = np.sum(group_batches)
        self.out = -1*np.ones((n_batches, batch_size))
        group_order = rng.permutation(np.repeat(group_unique, group_batches))


        for g in group_unique:
            # get the index of the items from that group
            group_idx = np.argwhere(group == g).reshape(-1)
            # shuffle the group index
            rng.shuffle(group_idx)
            # get the section of the output that we will edit
            out_temp = self.out[group_order==g].reshape(-1)
            # replace the values with the index of the items
            out_temp[:len(group_idx)] = (
                upsample_idx[group_idx] if upsample 
                else group_idx
                )
            out_temp = out_temp.reshape(-1, batch_size)
            rng.shuffle(out_temp, axis=0)
            self.out[group_order==g] = out_temp
            rng = np.random.default_rng(
                rng.integers(low=0, high=1e9, size=(3,))
                )
        
        self.out = [list(batch[batch != -1].astype(int)) for batch in self.out]

        return 

    def __iter__(self):
        return iter(self.out)
    
    def __len__(self):
        return len(self.out)

