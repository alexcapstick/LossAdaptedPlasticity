import torch
import numpy as np
from collections import deque
import typing
import collections.abc as abc
from .corruption_functions import *
import copy


class SourceCorrupter(torch.utils.data.Dataset):
    def __init__(self, 
        dataset:torch.utils.data.Dataset, 
        n_sources:int=10,
        source_size:int=10,
        n_corrupt_sources:int=0,
        corruption_function:typing.Union[None, abc.Callable, str]=None,
        corruption_function_kwargs:dict={},
        seed:int=None,
        shuffle_dataset:bool=True,
        shuffle_sources:int=True,
        return_sources:bool=True,
        return_bool_source:bool=False,
        corrupt_for_n_steps:typing.Union[None, int]=None,
        corruption_level:typing.Union[None, list]=None,
        ):
        '''
        
        This is a wrapper for a dataset, that randomly bins the points into
        sources, and allows the corruption of data according
        to these sources.
        
        
        
        Arguments
        ---------
        
        - dataset: torch.utils.data.Dataset: 
            The dataset to be wrapped and corrupted according to source.
        
        - n_sources: int, optional:
            The number of sources to bin the dataset into. 
            Defaults to :code:`10`.
        
        - source_size: int, optional:
            The number of data points binned into each batch for each source.
            Datapoints in the original dataset that do not fill these batches
            will be present in the final batch, but the batch will be smaller. 
            For example, if the original
            dataset was of size 24 and :code:`source_size=5`, 
            then there would be 4 batches of size 5 and
            1 batch of size 4.
            Defaults to :code:`10`.
        
        - n_corrupt_sources: int, optional:
            The number of corrupt sources in the dataset.
            This number must be :code:`<=n_sources`. The corrupt
            sources are available through the attribute 
            :code:`.corrupt_sources`. :code:`0`, refers to no
            corrupt sources.
            Defaults to :code:`0`.

        - corruption_function: function` or :code:`string, optional:
            This is a function that can be used to corrupt the data points 
            from the sources in the attribute :code:`.corrupt_sources`.
            :code:`None` forces the dataset to apply no corruption. 
            If using :code:`'label_shuffle'` or :code:`'label_flip'`,
            the indices given to this dataset when iterating must 
            be sequential, starting from :code:`0`, meaning you 
            should not use the :code:`shuffle=True` argument in 
            :code:`torch.utils.data.DataLoader`.
            Possible :code:`str` values are: 'chunk_swap': swaps chunks of
            the data (see :code:`SourceChunkSwap`),
            :code:`'noise': adds or replaces data with noise (see :code:`NoiseCorrupter`),
            :code:`'label_random': Swaps each label with a random label (see LabelRandom), 
            :code:`'label_shuffle': Shuffles the labels of each batch 
            (add :code:`source_save=True` to :code:`corruption_function_kwargs` to ensure
            the shuffling is done the same for the same index of points),
            :code:`'label_flip': Chooses a label in each batch to be the label for all 
            data points in that batch (add :code:`source_save=True` to 
            :code:`corruption_function_kwargs` to ensure the flipping is 
            done the same for the same index of points). 
            If specifying a function, this must be callable with inputs :code:`x`,
            :code:`y`, :code:`source`, :code:`index` and work on a single datapoint
            at a time.
            Defaults to :code:`None`.
        
        - corruption_function_kwargs: dict` (optional):
            The arguments that will be passed to each of the :code:`corruption_function`,
            if specifying them by string.
            Defaults to :code:`{}`.
        
        - seed: int, optional:
            The random seed that is used to build the 
            shuffling for the order of the dataset and sources, as
            well as the sources that are to be corrupted.
            Defaults to :code:`None`.
        
        - shuffle_dataset: bool, optional:
            Whether to shuffle the order of the underlying dataset. 
            To avoid issues, this should be used instead of the 
            :code:`shuffle=True` argument in :code:`torch.utils.data.DataLoader`.
            Defaults to :code:`True`.
        
        - shuffle_sources: int, optional:
            Whether to shuffle the order of the sources,
            or if they should be returned in the order:
            :code:`[0, 1, .., n_sources, 0, 1, ..., n_sources, ...]`.
            Defaults to :code:`True`.
        
        - return_sources: bool, optional:
            Whether to return the source along with the 
            datapoint and target in the :code:`__getitem__()` method. 
            If :code:`True`, the :code:`__getitem__()` method will return
            :code:`X, y, source`, otherwise it will return :code:`X, y`.
            Defaults to :code:`True`.
        
        - return_bool_source: bool, optional:
            Whether to return a boolean value that dictates
            if a datapoint is corrupted or not, instead of the
            source name. This is used
            to test how well an oracle model would perform.
            Defaults to :code:`False`.

        - corrupt_for_n_steps: int, optional:
            This is the number of :code:`__getitem__` calls
            made to this class before the corruption function 
            is set to :code:`None`. This means that it is the
            number of data points retrieved before corruption
            is turned off.
            Defaults to :code:`None`.

        - corruption_level: typing.Union[None, list], optional:
            This is the level of corruption that will be found
            in each of the sources. It should be a list, as long as
            the number of corrupt sources. If :code:`None`, then 100% corruption 
            is applied to the corrupt sources.
            Defaults to :code:`None`.

        
        Raises
        ---------

        - TypeError: if n_corrupt_sources > n_sources`
        
        
        '''

        if n_corrupt_sources > n_sources:
            raise TypeError('The number of corrupt sources must be '\
                            'smaller than or equal to the number of sources. '\
                            'The number of sources is {} and you specified the number '\
                            'of corrupt sources as {}.'.format(n_sources, n_corrupt_sources))

        if seed is None:
            rng = np.random.default_rng(None)
            self.seed = rng.integers(low=1, high=1e9, size=1)[0]
        else:
            self.seed = seed
        
        rng = np.random.default_rng(seed)
        rng = np.random.default_rng(rng.integers(low=1, high=1e9, size=5))
        self.corruption_func_seed = rng.integers(low=1, high=1e9, size=1)
      
        self.dataset = dataset
        sources = np.arange(n_sources)
        self.corrupt_sources = rng.choice(sources, size=n_corrupt_sources, replace=False)

        self.dataset_length = len(dataset)
        self.source_length = len(dataset)//source_size + 1

        if shuffle_sources:
            self.source_order = rng.choice(sources, size=self.source_length, replace=True)
        else:
            self.source_order = np.tile(sources, (self.source_length//n_sources)+1)[:self.source_length]

        if shuffle_dataset:
            self.dataset_order = rng.permutation(self.dataset_length)
        else:
            self.dataset_order = np.arange(self.dataset_length)

        self.n_sources = n_sources
        self.source_size = source_size
        self.corruption_function = (self._get_corruption_function(corruption_function, corruption_function_kwargs) 
                                    if type(corruption_function) == str else corruption_function)
        self.return_sources = return_sources
        self.return_bool_source = return_bool_source

        self.rng = np.random.default_rng(rng.integers(low=1, high=1e9, size=1))

        self.corrupt_for_n_steps = corrupt_for_n_steps
        self.step = 0
        if corruption_level is None:
            self.corruption_level = None
        else:
            self.corruption_level = {
                source: corruption 
                for source, corruption 
                in zip(self.corrupt_sources, corruption_level)
                }
        self.print_index=False

        return
    
    def _get_corruption_function(self, corruption_function_name, corruption_function_kwargs):
        
        if corruption_function_name == 'chunk_swap':
            return SourceChunkSwap(seed = self.corruption_func_seed, **corruption_function_kwargs)
        
        elif corruption_function_name == 'noise':
            return NoiseCorrupter(seed = self.corruption_func_seed, **corruption_function_kwargs)
        
        elif corruption_function_name == 'label_random':
            return LabelRandom(seed = self.corruption_func_seed, **corruption_function_kwargs)
        
        elif corruption_function_name == 'label_shuffle':
            self.index_cached = deque([])
            self.source_save = corruption_function_kwargs['source_save']
            return self._label_shuffle
        
        elif corruption_function_name == 'label_flip':
            self.index_cached = deque([])
            self.source_save = corruption_function_kwargs['source_save']
            return self._label_flip
        
        else:
            raise NotImplementedError('Currently, this corruption function is not supported by string. '\
                                        'Try passing the function instead.')
        
        return


    def _label_shuffle(self, index, source_save=True, **kwargs):

        # if cache is empty, fill with inputs and targets
        # so that the targets can be shuffled
        if index == 0:
            self.index_cached = deque([])
        if len(self.index_cached) == 0 or index == 0:
            self.label_cached = []
            self.input_cached = deque([])
            for future_index in range(index, index+self.source_size):
                if future_index >= self.dataset_length:
                    break
                self.index_cached.append(future_index)
                datapoint_idx = self.dataset_order[future_index]
                x, y = self.dataset[datapoint_idx]
                self.label_cached.append(y)
                self.input_cached.append(x)
            # randomly shuffle labels for this batch
            if self.source_save:
                seed = index + self.seed
                self.rng = np.random.default_rng(seed)
            else:
                self.seed = self.rng.integers(low=1, high=1e9, size=1)[0]
                seed = self.seed
                self.rng = np.random.default_rng(seed)

            self.rng.shuffle(self.label_cached)
            self.rng = np.random.default_rng(self.rng.integers(low=1, high=10000, size=1))
            self.label_cached = deque(self.label_cached)
        x, y = self.input_cached.popleft(), self.label_cached.popleft()
        self.index_cached.popleft()

        return x, y


    def _label_flip(self, index, **kwargs):

        # if cache is empty, fill with inputs and targets
        # so that the targets can be shuffled
        if index == 0:
            self.index_cached = deque([])
        if len(self.index_cached) == 0:
            self.label_cached = []
            self.input_cached = deque([])
            for future_index in range(index, index+self.source_size):
                if future_index >= self.dataset_length:
                    break
                self.index_cached.append(future_index)
                datapoint_idx = self.dataset_order[future_index]
                x, y = self.dataset[datapoint_idx]
                self.label_cached.append(y)
                self.input_cached.append(x)
            # randomly shuffle labels for this batch
            if self.source_save:
                seed = index + self.seed
                self.rng = np.random.default_rng(seed)
            else:
                self.seed = self.rng.integers(low=1, high=1e9, size=1)[0]
                seed = self.seed
                self.rng = np.random.default_rng(seed)

            new_label = self.rng.choice(self.label_cached)
            self.label_cached = [int(new_label)]*len(self.label_cached)
            self.rng = np.random.default_rng(self.rng.integers(low=1, high=10000, size=1))
            self.label_cached = deque(self.label_cached)
        x, y = self.input_cached.popleft(), self.label_cached.popleft()
        self.index_cached.popleft()

        return x, y

    def __getitem__(self, index):
        index = int(index)
        self.step += 1
        source = self.source_order[index//self.source_size]
        datapoint_idx = self.dataset_order[index]
        x, y = self.dataset[datapoint_idx]
        if not self.corrupt_for_n_steps is None:
            if self.step > self.corrupt_for_n_steps:
                self.corruption_function = None
        if not self.corruption_function is None:
            if source in self.corrupt_sources:
                if not self.corruption_level is None:
                    x_true = copy.deepcopy(x)
                    y_true = copy.deepcopy(y)
                x, y = self.corruption_function(x=x, y=y, source=source, index=index)
                if not self.corruption_level is None:
                    seed = index + self.seed
                    self.rng = np.random.default_rng(seed)
                    corrupt = bool(
                        self.rng.choice(
                            2, 
                            p=[
                                1-self.corruption_level[source], 
                                self.corruption_level[source]
                                ]))
                    if not corrupt:
                        x, y = x_true, y_true

        if self.return_sources:
            if self.return_bool_source:
                return x, y, torch.tensor([source in self.corrupt_sources])
            return x, y, torch.tensor([source])
        else:
            return x, y
    
    def __len__(self):
        return self.dataset_length



