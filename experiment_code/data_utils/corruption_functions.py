import torch
import numpy as np
import typing



class NoiseCorrupter(object):
    def __init__(self, mean:float=0, std:float=1, same_across_rgb:bool=False, 
                    noise_only:bool=False, source_save:bool=True, seed:typing.Union[None,int]=None):
        '''
        A class for corrupting vector data by adding noise.

        
        Arguments
        ---------

        - mean: float (optional):
            Mean of the gaussian noise to add.
            Defaults to :code`0`.
        
        - std: float (optional):
            Standard deviation of the gaussian noise to add.
            Defaults to :code`1`.

        - same_across_rgb: bool (optional):
            This dictates whether the same shuffling will be used in
            each of the three dimensions of an image given.
            Defaults to :code`False`.

        - noise_only: bool (optional):
            This dictates whether the corrupted samples
            will have noise added or be replaces with noise.
            If :code`True`, data will be replaced with noise.
            Defaults to :code`False`.

        - source_save: bool (optional):
            Whether to use the same corruption for 
            the same index, or whether to 
            randomise the corruption every time an image is streamed.
            Defaults to :code`True`.
        
        - seed: None | int (optional):
            Whether to set the random operations in this
            class using a random seed.
            Defaults to :code`None`.
        
        '''

        self.mean = mean
        self.std = std

        if seed is None:
            rng = np.random.default_rng(None)
            self.seed = rng.integers(low=1, high=1e9, size=1)[0]
        else:
            rng = np.random.default_rng(seed)
            self.seed = rng.integers(low=1, high=1e9, size=1)[0]
        self.seed = int(self.seed)
        self.same_across_rgb = same_across_rgb
        self.noise_only = noise_only
        self.source_save = source_save

    
    def add_noise(self, x, source, index):

        if self.source_save:
            g = torch.Generator().manual_seed(index+self.seed)
            noise = torch.zeros(x.shape).normal_(mean=self.mean, std=self.std, generator=g)
        else:
            self.seed = int(np.random.default_rng(self.seed).integers(low=1, high=1e9, size=1)[0])
            g = torch.Generator().manual_seed(self.seed)
            noise = torch.normal(mean=torch.tensor(self.mean).float(), 
                                    std=torch.tensor(self.std).float(), 
                                    size=x.shape, generator=g)
        x_out = x + noise
        
        return x_out
    
    def add_noise_rgb(self, x, source, index):

        if self.source_save:
            g = torch.Generator().manual_seed(index+self.seed)
            noise = torch.zeros(x[0,:,:].shape).normal_(mean=self.mean, std=self.std, generator=g)
        else:
            self.seed = int(np.random.default_rng(self.seed).integers(low=1, high=1e9, size=1)[0])
            g = torch.Generator().manual_seed(self.seed)
            noise = torch.normal(mean=torch.tensor(self.mean).float(), 
                                    std=torch.tensor(self.std).float(), 
                                    size=x[0,:,:].shape, generator=g)
        
        x_out = x + noise

        return x_out
    
    def __call__(self, x:torch.tensor, y, source, index:int, **kwargs):
        '''
        
        Arguments
        ---------

        - x: torch.tensor:
            The vector to have its chunks permutated.
        
        - y: target
            This is the target of the input. This is ignored
            by the function.
        
        - source: hashable:
            This is the source of the input. This is ignored
            by the function.
        
        - index: int:
            This is the index of the data point that is 
            being corrupted. The index is used to make
            sure that the same data point is corrupted 
            in the same way each time it is called.
            This is only used if :code`source_save=True`.

        
        '''

        if self.noise_only:
            if self.source_save:
                g = torch.Generator().manual_seed(index+self.seed)
                x_out = torch.zeros(x.shape).normal_(mean=self.mean, std=self.std, generator=g)
            else:
                self.seed = int(np.random.default_rng(self.seed).integers(low=1, high=1e9, size=1)[0])
                g = torch.Generator().manual_seed(self.seed)
                x_out = torch.normal(mean=torch.tensor(self.mean).float(), 
                                        std=torch.tensor(self.std).float(), 
                                        size=x.shape, generator=g)
        else:
            if len(x.shape) <= 2:
                x_out =  self.add_noise(x, source, index)

            elif len(x.shape) == 3:
                if self.same_across_rgb:
                    x_out =  self.add_noise_rgb(x, source, index)
                else:
                    x_out =  self.add_noise(x, source, index)
            else:
                raise NotImplementedError('Please supply a 1D, 2D, or 3*2D x.')

        return x_out, y


class LabelRandom(object):
    def __init__(self, 
                    labels:int=10,
                    seed:int=None,
                    source_save:bool=False,
                    ):
        '''
        This randomly assigns a new label to a given input.
        
        Arguments
        ---------

        - labels: int or list (optional):
            If :code`int`, then all integers smaller
            than this value are possibly assigned labels.
            If :code`list`, then these labels are used
            for the randomly assigned labels.
        
        - seed: int, optional:
            The random seed to set the random labels. 
            Defaults to :code`None`.

        - source_save: bool, optional:
            This saves the random label mapping
            corruption by source, so that a given source
            maps the labels in the same way.
            Defaults to :code`False`.
        
        '''

        if seed is None:
            rng = np.random.default_rng(None)
            self.seed = rng.integers(low=1, high=10000, size=1)[0]
        else:
            rng = np.random.default_rng(seed)
            self.seed = rng.integers(low=1, high=1e9, size=1)[0]
    
        rng = np.random.default_rng(self.seed)
        self.rng = np.random.default_rng(rng.integers(low=1, high=10000, size=1))

        self.labels = labels
        self.source_save = source_save
        self.source_save_dict = {}

        return
    
    def __call__(self, x:torch.tensor, y, source, **kwargs):
        '''
        
        Arguments
        ---------

        - x: torch.tensor:
            The vector to have its chunks permutated.
        
        - y: target
            This is the target of the input.
        
        - source: hashable:
            This is ignored.
        
        '''

        new_seed = self.rng.integers(low=1, high=10000, size=1)
        self.rng = np.random.default_rng(new_seed)

        if self.source_save:
            if not source in self.source_save_dict:
                self.source_save_dict[source] = {}
            
            if not y in self.source_save_dict[source]:
                self.source_save_dict[source][y] = self.rng.choice(self.labels)
            
            y_out = self.source_save_dict[source][y]
            
        else:
            y_out = self.rng.choice(self.labels)

        return x, y_out


class SourceChunkSwap(object):
    def __init__(self, 
                    n_xpieces:int=10, 
                    source_save:bool=True, 
                    seed:int=None, 
                    same_across_rgb:bool=False,
                    ):
        '''
        A class for corrupting data by swapping chunks of the
        data with eachother.
        
        Arguments
        ---------

        - n_xpieces: int (optional):
            The number of chunks in the input to rearrange.
            For 2D shapes, this is the number of chunks in the
            x and y direction. This means that for a 2D shape,
            :code:`n_xpieces**2` chunks will be rearranged.
            Defaults to :code:`10`.
        
        - source_save: bool (optional):
            Whether to use the same corruption for 
            the same example, or whether to 
            randomise the corruption every time an image is streamed.
            Defaults to :code:`True`.
        
        - seed: int (optional):
            This value determines the random process for 
            the swapping of chunks in the corruption process.
            Defaults to :code:`None`.
        
        - same_across_rgb: bool (optional):
            This dictates whether the same shuffling will be used in
            each of the three dimensions of an image given.
            Defaults to :code:`False`.
        
        '''

        self.n_xpieces = n_xpieces
        self.source_save = source_save

        if seed is None:
            rng = np.random.default_rng(None)
            self.seed = rng.integers(low=1, high=1e9, size=1)[0]
        else:
            rng = np.random.default_rng(seed)
            self.seed = rng.integers(low=1, high=1e9, size=1)[0]
        
        self.rng = np.random.default_rng(self.seed)

        self.same_across_rgb = same_across_rgb

    def chunk_rearrange(self, data, chunk_sizes, new_order):
        '''
        Adapted from https://stackoverflow.com/a/62292488
        to require and work with :code:`torch.tensor`s.
        '''
        m = chunk_sizes[:,None] > torch.arange(chunk_sizes.max())
        d1 = torch.empty(m.shape, dtype=data.dtype)
        d1[m] = data
        return d1[new_order][m[new_order]]

    def chunk_rearrange_2d(self, data, chunk_sizes, new_order):
        swap_x = self.chunk_rearrange(torch.arange(data.shape[1]), chunk_sizes=chunk_sizes[1], new_order=new_order[1])
        swap_y = self.chunk_rearrange(torch.arange(data.shape[0]), chunk_sizes=chunk_sizes[0], new_order=new_order[0])
        return data[:, swap_y][swap_x,:]
    
    def call_1d(self, x, source, index):

        x_shape = x.shape[0]
        box_bounds = (torch.linspace(0, x_shape, self.n_xpieces+1, dtype=int)).reshape(-1,1)
        chunks = box_bounds[1:] - box_bounds[:-1]        

        if self.source_save:
            seed = index + self.seed
            self.rng = np.random.default_rng(seed)
        else:
            self.seed = self.rng.integers(low=1, high=1e9, size=1)[0]
            seed = self.seed
            self.rng = np.random.default_rng(seed)
            
        new_order = self.rng.permutation(len(chunks))
        x_out = self.chunk_rearrange(x, chunk_sizes=chunks, new_order=new_order)
        
        return x_out

    def call_2d(self, x, source, index):

        xy_shape = x.shape[0]
        xx_shape = x.shape[1]

        if self.n_xpieces > min(xy_shape, xx_shape):
            raise TypeError('Please make sure that the number of pieces is smaller than '\
                            'both sides of the input. Array was shape {} and the number '\
                                'of pieces was {}.'.format(x.shape, self.n_xpieces))
        
        ybox_bounds = (torch.linspace(0, xy_shape, self.n_xpieces+1, dtype=int)).reshape(-1,1)
        xbox_bounds = (torch.linspace(0, xx_shape, self.n_xpieces+1, dtype=int)).reshape(-1,1)
        
        ychunks = ybox_bounds[1:] - ybox_bounds[:-1]
        xchunks = xbox_bounds[1:] - xbox_bounds[:-1]

        chunks = torch.cat([ychunks.reshape(1,-1), xchunks.reshape(1,-1)], dim=0)

        if self.source_save:
            seed = index +  self.seed
            self.rng = np.random.default_rng(seed)
            self.rng = np.random.default_rng(self.rng.integers(low=1, high=1e9, size=2))
        else:
            self.seed = self.rng.integers(low=1, high=1e9, size=1)[0]
            seed = self.seed
            self.rng = np.random.default_rng(seed)
            self.rng = np.random.default_rng(self.rng.integers(low=1, high=1e9, size=2))

        new_order = []
        xnew_order = self.rng.permutation(len(xchunks), axis=1)
        ynew_order = self.rng.permutation(len(ychunks), axis=1)
        new_order.extend([xnew_order, ynew_order])
        x_out = self.chunk_rearrange_2d(x, chunk_sizes=chunks, new_order=new_order)
        
        return x_out
    
    def call_rgb(self, x, source, index):

        idx = torch.arange(len(x[0,:, :].reshape(-1))).reshape(x.shape[1], x.shape[2])
        if self.same_across_rgb:
            new_idx = self.call_2d(idx, source=source, index=index)
            # apply the new index to each channel
            x_out = x.reshape(x.shape[0], -1)[:,new_idx.reshape(-1)].reshape(x.shape)
        else:
            new_idx_list = [self.call_2d(idx, source=source, index=index) for _ in range(3)]
            # apply the new index to each channel
            x_out = x.reshape(x.shape[0], -1)
            for ii, new_idx in enumerate(new_idx_list):
                x_out[ii, :] = x_out[ii, new_idx.reshape(-1)]
            x_out = x_out.reshape(x.shape)

        return x_out
    
    def __call__(self, x:torch.tensor, y, source, index, **kwargs):
        '''
        
        Arguments
        ---------

        - x: torch.tensor:
            The vector to have its chunks permutated.
        
        - y: target
            This is the target of the input and is ignored.
        
        - source: hashable:
            The source of the input, which will be used to save
            the corruption for that source if 
            :code:`source_save=True`.
        
        '''
        if x.shape[0] == 1:
            x_out = x.reshape(-1)
            reshape_after = True
        else:
            x_out = x
            reshape_after = False

        if len(x_out.shape) == 2:
            x_out =  self.call_2d(x_out, source=source, index=index)
        elif len(x_out.shape) == 1:
            x_out =  self.call_1d(x_out, source=source, index=index)
        elif len(x_out.shape) == 3:
            x_out =  self.call_rgb(x_out, source=source, index=index)
        else:
            raise NotImplementedError('Please supply a 1D, 2D, or 3*2D x.')
        
        if reshape_after:
            x_out = x_out.reshape(1,-1)

        return x_out, y






class ECGCorruptor(torch.utils.data.Dataset):
    def __init__(
        self, 
        dataset:torch.utils.data.Dataset, 
        corrupt_sources:typing.Union[list, int, None]=None,
        noise_level:typing.Union[list, float, None]=None,
        seed:typing.Union[int, None]=None,
        axis:str='both',
        x_noise_std:float=0.1,
        ):
        '''
        ECG Data corruptor. You may pass a noise level, sources to corrupt,
        and the seed for determining the random events. This
        class allows you to corrupt either the :code:`'x'`, :code:`'y'`, 
        or :code:`'both'`. This class is built specifically for use with
        PTB_XL (found in :code:`aml.data.datasets`).

        This function will work as expected on all devices.
        
        
        Examples
        ---------
        
        .. code-block::
        
            >>> dataset = ECGCorruptor(
            ...     dataset=dataset_train
            ...     corrupt_sources=[0,1,2,3], 
            ...     noise_level=0.5, 
            ...     )

        
        
        Arguments
        ---------
        
        - dataset: torch.utils.data.Dataset:
            The dataset to corrupt. When iterated over,
            the dataset should return :code:`x`, :code:`y`, 
            and :code:`source`.

        - corrupt_sources: typing.Union[list, int, None], optional:
            The sources to corrupt in the dataset. This can be a 
            list of sources, an integer of the source, or :code:`None`
            for no sources to be corrupted.
            Defaults to :code:`None`.

        - noise_level: typing.Union[list, int, None], optional:
            This is the level of noise to apply to the dataset. 
            It can be a list of noise levels, a single noise level to
            use for all sources, or :code:`None` for no noise.
            Defaults to :code:`None`.

        - seed: typing.Union[int, None], optional:
            This is the seed that is used to determine random events.
            Defaults to :code:`None`.

        - axis: str, optional:
            This is the axis to apply the corruption to. This
            should be either :code:`'x'`, :code:`'y'`, 
            or :code:`'both'`.
            
            - :code:`'x'`: \
            Adds a Gaussian distribution to the \
            :code:`'x'` values with :code:`mean=0` and :code:`std=0.1`.
            
            - :code:`'y'`: \
            Swaps the binary label using the function :code:`1-y_true`.
            
            - :code:`'both'`: \
            Adds a Gaussian distribution to the \
            :code:`'x'` values with :code:`mean=0` and :code:`std=0.1` \
            and swaps the binary label using the function :code:`1-y_true`.

            Defaults to :code:`'both'`.

        - x_noise_std: float, optional:
            This is the standard deviation of the noise that 
            is added to :code:`x` when it is corrupted.
            Defaults to :code:`0.1`.
        
        
        '''

        assert axis in ['x', 'y', 'both'], \
            "Please ensure that the axis is from ['x', 'y', 'both']"
        
        self._axis = axis
        self._dataset = dataset
        self._x_noise_std = x_noise_std

        # setting the list of corrupt sources
        if corrupt_sources is None:
            self._corrupt_sources = []
        elif type(corrupt_sources) == int:
            self._corrupt_sources = [corrupt_sources]
        elif hasattr(corrupt_sources, '__iter__'):
            self._corrupt_sources = corrupt_sources
        else:
            raise TypeError(
                "Please ensure that corrupt_sources is an integer, iterable or None."
                )

        # setting the noise level
        if noise_level is None:
            self._noise_level = [0]*len(self._corrupt_sources)
        elif type(noise_level) == float:
            self._noise_level = [noise_level]*len(self._corrupt_sources)
        elif hasattr(noise_level, '__iter__'):
            if hasattr(noise_level, '__len__'):
                if hasattr(self._corrupt_sources, '__len__'):
                    assert len(noise_level) == len(self._corrupt_sources), \
                        "Please ensure that the noise level "\
                        "is the same length as the corrupt sources."
            self._noise_level = noise_level
        else:
            raise TypeError(
                "Please ensure that the noise level is a float, iterable or None"
                )
        self._noise_level = {
            cs: nl for cs, nl in zip(self._corrupt_sources, self._noise_level)
            }

        if seed is None:
            rng = np.random.default_rng(None)
            seed = rng.integers(low=1, high=1e9, size=1)[0]
        self.rng = np.random.default_rng(seed)

        self._corrupt_datapoints = {'x': {}, 'y':{}}

        return

    def _corrupt_x(self, index, x, y, s):
        if index in self._corrupt_datapoints['x']:
            x = self._corrupt_datapoints['x'][index]
        else:
            g_seed_mask, \
                g_seed_values, \
                class_seed = self.rng.integers(low=1, high=1e9, size=3)
            self.rng = np.random.default_rng(class_seed)
            g_values = torch.Generator(device=y.device).manual_seed(int(g_seed_values))
            g_mask = torch.Generator(device=y.device).manual_seed(int(g_seed_mask))
            mask = int(
                torch.rand(
                    size=(), generator=g_mask, device=x.device
                    ) > 1-self._noise_level[s]
                )
            values = torch.normal(
                mean=0, 
                std=self._x_noise_std, 
                generator=g_values, 
                size=x.size(), 
                device=x.device,
                )
            x = x + mask*values
            self._corrupt_datapoints['x'][index] = x
        return x, y, s
    
    def _corrupt_y(self, index, x, y, s):
        if index in self._corrupt_datapoints['y']:
            y = self._corrupt_datapoints['y'][index]
        else:
            g_seed_mask, \
                class_seed = self.rng.integers(low=1, high=1e9, size=2)
            self.rng = np.random.default_rng(class_seed)
            g_mask = torch.Generator().manual_seed(int(g_seed_mask))
            if torch.rand(size=(), generator=g_mask) > 1-self._noise_level[s]:
                y = torch.tensor(1, dtype=y.dtype, device=y.device)-y

            self._corrupt_datapoints['y'][index] = y

        return x, y, s

    @property
    def corrupt_sources(self):
        return self._corrupt_sources

    def __getitem__(self, index):
        x, y, s = self._dataset[index]
        if s in self._noise_level:
            if self._axis == 'x' or self._axis == 'both':
                x,y,s = self._corrupt_x(index, x, y, s)
            if self._axis == 'y' or self._axis == 'both':
                x,y,s = self._corrupt_y(index, x, y, s)
        return x,y,s
    
    def __len__(self,):
        return len(self._dataset)

    # defined since __getattr__ causes pickling problems
    def __getstate__(self):
        return vars(self)

    # defined since __getattr__ causes pickling problems
    def __setstate__(self, state):
        vars(self).update(state)

    def __getattr__(self, name):
        if hasattr(self._dataset, name):
            return getattr(self._dataset, name)
        else:
            raise AttributeError