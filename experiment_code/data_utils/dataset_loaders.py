import torch
import torchvision
from torchvision import transforms
import numpy as np
import typing
import typing
import ast
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MultiLabelBinarizer
import os
from torchvision.datasets.utils import download_and_extract_archive
import joblib
import tqdm
from ..utils.utils import tqdm_style


try:
    import wfdb
    wfdb_import_error = False
except ImportError:
    wfdb_import_error = True


###### transformations

class FlattenImage(torch.nn.Module):
    def __init__(self):
        '''
        Allows you to flatten an input to 
        1D. This is useful in pytorch
        transforms when loading data.
        
        '''
        super(FlattenImage, self).__init__()

    def forward(self, x):
        return x.reshape(-1)


def get_dataset_function(dataset_name):
    func_name = 'get_{}'.format(dataset_name)
    return globals()[func_name]



###### datasets

def get_mnist(path, return_targets=False):
    '''
    Function to get the mnist data from pytorch with
    some transformations first.

    The returned MNIST data will be flattened.

    Arguments
    ---------

    - path: str:
        The path that the data is located or will be saved.
        This should be a directory containing :code:``MNIST`.
    
    - return_targets: bool`, (optional):
        Whether to return the targets along with the 
        datasets.
        Defaults to :code:`False`.

    Returns
    ---------

        - train_mnist: torch.utils.data.Dataset`

        - test_mnist: torch.utils.data.Dataset`

        - If :code:`return_targets=True:
            - train_mnist_targets: torch.tensor`
            - test_mnist_targets: torch.tensor`

    '''
    transform_images = transforms.Compose([
                            transforms.PILToTensor(),
                            transforms.ConvertImageDtype(torch.float),
                            transforms.Normalize(mean=0, std=1),
                            FlattenImage(),
                            ])

    train_mnist = torchvision.datasets.MNIST(root=path, 
                                                download=True, 
                                                train=True,
                                                transform=transform_images)

    test_mnist = torchvision.datasets.MNIST(root=path, 
                                                    download=True, 
                                                    train=False,
                                                    transform=transform_images)
    if return_targets:
        train_mnist_targets = torch.tensor(np.asarray(train_mnist.targets).astype(int))
        test_mnist_targets = torch.tensor(np.asarray(test_mnist.targets).astype(int))

        return train_mnist, test_mnist, train_mnist_targets, test_mnist_targets

    return train_mnist, test_mnist



def get_fmnist(path, return_targets=False):
    '''
    Function to get the FMNIST data from pytorch with
    some transformations first.

    The returned FMNIST data will be flattened.

    Arguments
    ---------

    - path: str:
        The path that the data is located or will be saved.
        This should be a directory containing :code:``FashionMNIST`.
    
    - return_targets: bool`, (optional):
        Whether to return the targets along with the 
        datasets.
        Defaults to :code:`False`.

    Returns
    ---------

        - train_fmnist: torch.utils.data.Dataset`

        - test_fmnist: torch.utils.data.Dataset`

        - If :code:`return_targets=True:
            - train_fmnist_targets: torch.tensor`
            - test_fmnist_targets: torch.tensor`

    '''
    transform_images = transforms.Compose([
                            transforms.PILToTensor(),
                            transforms.ConvertImageDtype(torch.float),
                            transforms.Normalize(mean=0, std=1),
                            FlattenImage(),
                            ])

    train_fmnist = torchvision.datasets.FashionMNIST(root=path, 
                                                download=True, 
                                                train=True,
                                                transform=transform_images)

    test_fmnist = torchvision.datasets.FashionMNIST(root=path, 
                                                    download=True, 
                                                    train=False,
                                                    transform=transform_images)
    if return_targets:
        train_fmnist_targets = torch.tensor(np.asarray(train_fmnist.targets).astype(int))
        test_fmnist_targets = torch.tensor(np.asarray(test_fmnist.targets).astype(int))

        return train_fmnist, test_fmnist, train_fmnist_targets, test_fmnist_targets

    return train_fmnist, test_fmnist



def get_cifar10(path, return_targets=False):
    '''
    Function to get the CIFAR 10 data from pytorch with
    some transformations first.

    The returned CIFAR 10 data will be flattened.

    Arguments
    ---------

    - path: str:
        The path that the data is located or will be saved.
        This should be a directory containing :code:``cifar-10-batches-py`.
    
    - return_targets: bool`, (optional):
        Whether to return the targets along with the 
        datasets.
        Defaults to :code:`False`.

    Returns
    ---------

        - train_cifar: torch.utils.data.Dataset`

        - test_cifar: torch.utils.data.Dataset`

        - If :code:`return_targets=True:
            - train_cifar_targets: torch.tensor`
            - test_cifar_targets: torch.tensor`

    '''
    transform_images = transforms.Compose([
                            transforms.PILToTensor(),
                            transforms.ConvertImageDtype(torch.float),
                            #transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261)),
                            ])

    train_cifar = torchvision.datasets.CIFAR10(root=path, 
                                                download=True, 
                                                train=True,
                                                transform=transform_images)

    test_cifar = torchvision.datasets.CIFAR10(root=path, 
                                                    download=True, 
                                                    train=False,
                                                    transform=transform_images)
    if return_targets:
        train_cifar_targets = torch.tensor(np.asarray(train_cifar.targets).astype(int))
        test_cifar_targets = torch.tensor(np.asarray(test_cifar.targets).astype(int))

        return train_cifar, test_cifar, train_cifar_targets, test_cifar_targets

    return train_cifar, test_cifar


def get_cifar100(path, return_targets=False):
    '''
    Function to get the CIFAR 100 data from pytorch with
    some transformations first.

    The returned CIFAR 100 data will be flattened.

    Arguments
    ---------

    - path: str:
        The path that the data is located or will be saved.
        This should be a directory containing :code:``cifar-100-python`.
    
    - return_targets: bool`, (optional):
        Whether to return the targets along with the 
        datasets.
        Defaults to :code:`False`.

    Returns
    ---------

        - train_cifar: torch.utils.data.Dataset`

        - test_cifar: torch.utils.data.Dataset`

        - If :code:`return_targets=True:
            - train_cifar_targets: torch.tensor`
            - test_cifar_targets: torch.tensor`

    '''
    transform_images = transforms.Compose([
                            transforms.PILToTensor(),
                            transforms.ConvertImageDtype(torch.float),
                            #transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261)),
                            ])

    train_cifar = torchvision.datasets.CIFAR100(root=path, 
                                                download=True, 
                                                train=True,
                                                transform=transform_images)

    test_cifar = torchvision.datasets.CIFAR100(root=path, 
                                                    download=True, 
                                                    train=False,
                                                    transform=transform_images)

    if return_targets:
        train_cifar_targets = torch.tensor(np.asarray(train_cifar.targets).astype(int))
        test_cifar_targets = torch.tensor(np.asarray(test_cifar.targets).astype(int))

        return train_cifar, test_cifar, train_cifar_targets, test_cifar_targets

    return train_cifar, test_cifar








class WrapperDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset:torch.utils.data.Dataset,
        functions_index:typing.Union[typing.List[int], int, None]=None,
        functions:typing.Union[typing.Callable, typing.List[typing.Callable]]=lambda x: x,
        ):
        '''
        This allows you to wrap a dataset with a set of 
        functions that will be applied to each returned 
        data point. You can apply a single function to all 
        outputs of a data point, or a different function
        to each of the different outputs.
        
        
        
        Examples
        ---------

        The following would multiply all of the first returned
        values in the dataset by 2.
        
        .. code-block::
        
            >>> WrapperDataset(
            ...     dataset
            ...     functions_index=0,
            ...     functions=lambda x: x*2
            ...     )

        The following would multiply all of the returned
        values in the dataset by 2.
        
        .. code-block::
        
            >>> WrapperDataset(
            ...     dataset
            ...     functions_index=None,
            ...     functions=lambda x: x*2
            ...     )

        The following would multiply all of the first returned
        values in the dataset by 2, and the second by 2.
        
        .. code-block::
        
            >>> WrapperDataset(
            ...     dataset
            ...     functions_index=[0, 1],
            ...     functions=lambda x: x*2
            ...     )

        The following would multiply all of the first returned
        values in the dataset by 2, and the second by 3.
        
        .. code-block::
        
            >>> WrapperDataset(
            ...     dataset
            ...     functions_index=[0, 1],
            ...     functions=[lambda x: x*2, lambda x: x*3]
            ...     )
        
        
        Arguments
        ---------
        
        - dataset: torch.utils.data.Dataset: 
            The dataset to be wrapped.
        
        - functions_index: typing.Union[typing.List[int], int, None], optional:
            The index of the functions to be applied to. 

            - If :code:`None`, then if the :code:`functions` is callable, it \
            will be applied to all outputs of the data points, \
            or if the :code:`functions` is a list, it will be applied to the corresponding \
            output of the data point.

            - If :code:`list` then the corresponding index will have the \
            :code:`functions` applied to them. If :code:`functions` is a list, \
            then it will be applied to the corresponding indicies given in :code:`functions_index` \
            of the data point. If :code:`functions` is callable, it will be applied to all of the \
            indicies in :code:`functions_index`
        
            - If :code:`int`, then the :code:`functions` must be callable, and \
            will be applied to the output of this index.
            
            Defaults to :code:`None`.
        
        - functions: _type_, optional:
            This is the function, or list of functions to apply to the
            corresponding indices in :code:`functions_index`. Please
            see the documentation for the :code:`functions_index` argument
            to understand the behaviour of different input types. 
            Defaults to :code:`lambda x:x`.
        
        
        '''

        self._dataset = dataset
        if functions_index is None:
            if type(functions) == list:
                self.functions = {fi: f for fi, f in enumerate(functions)}
            elif callable(functions):
                self.functions=functions
            else:
                raise TypeError("If functions_index=None, please ensure "\
                    "that functions is a list or a callable object.")
        
        elif type(functions_index) == list:
            if type(functions) == list:
                assert len(functions_index) == len(functions), \
                    "Please ensure that the functions_index is the same length as functions."
                self.functions = {fi: f for fi, f in zip(functions_index, functions)}
            elif callable(functions):
                self.functions = {fi: functions for fi in functions_index}
            else:
                raise TypeError("If type(functions_index)==list, please ensure "\
                    "that functions is a list of the same length or a callable object.")

        elif type(functions_index) == int:
            if callable(functions):
                self.functions = {functions_index: functions}
            else:
                raise TypeError("If type(functions_index)==int, please ensure "\
                    "the functions is a callable object.")

        else:
            raise TypeError("Please ensure that functions_index is a list, int or None.")

        return

    def __getitem__(self, index):
        if type(self.functions) == dict:
            return [
                self.functions.get(nout, lambda x: x)(out) 
                for nout, out in enumerate(self._dataset[index])
                ]
        elif callable(self.functions):
            return [self.functions(out) for out in self._dataset[index]]
        else:
            raise TypeError("The functions could not be applied.")
    
    def __len__(self):
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






class MemoryDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset:torch.utils.data.Dataset,
        now:bool=True,
        verbose:bool=True,
        n_jobs:int=1,
        ):
        '''
        This dataset allows the user
        to wrap another dataset and 
        load all of the outputs into memory,
        so that they are accessed from RAM 
        instead of storage. All attributes of
        the original dataset will still be available, except
        for :code:`._dataset` and :code:`._data_dict` if they 
        were defined.
        It also allows the data to be saved in memory right
        away or after the data is accessed for the first time.
               
        
        Examples
        ---------
        
        .. code-block::
        
            >>> dataset = MemoryDataset(dataset, now=True)
        
        
        Arguments
        ---------
        
        - dataset: torch.utils.data.Dataset: 
            The dataset to wrap and add to memory.
        
        - now: bool, optional:
            Whether to save the data to memory
            right away, or the first time the 
            data is accessed. If :code:`True`, then
            this initialisation might take some time
            as it will need to load all of the data.
            Defaults to :code:`True`.
        
        - verbose: bool, optional:
            Whether to print progress
            as the data is being loaded into
            memory. This is ignored if :code:`now=False`.
            Defaults to :code:`True`.
        
        - n_jobs: int, optional:
            The number of parallel operations when loading 
            the data to memory.
            Defaults to :code:`1`.
        
        
        '''

        self._dataset = dataset
        self._data_dict = {}
        if now:

            pbar = tqdm.tqdm(
                total = len(dataset),
                desc='Loading into memory',
                disable=not verbose,
                smoothing=0,
                **tqdm_style
                )

            def add_to_dict(index):
                for ni, i in enumerate(index):
                    self._data_dict[i] = dataset[i]
                    pbar.update(1)
                    pbar.refresh()
                return None

            all_index = np.arange(len(dataset))
            index_list = [all_index[i::n_jobs] for i in range(n_jobs)]

            joblib.Parallel(
                n_jobs=n_jobs,
                backend='threading',
                )(
                    joblib.delayed(add_to_dict)(index)
                    for index in index_list
                    )
            
            pbar.close()

        return

    def __getitem__(self, index):

        if index in self._data_dict:
            return self._data_dict[index]
        else:
            output = self._dataset[index]
            self._data_dict[index] = output
            return output
    
    def __len__(self):
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










class PTB_XL(torch.utils.data.Dataset):
    def __init__(
        self,
        data_path:str='./',
        train:bool=True,
        sampling_rate:typing.Literal[100, 500]=100,
        source_name:typing.Literal['nurse', 'site', 'device']='nurse',
        return_sources:bool=True,
        binary:bool=False,
        subset=False,
        ):
        '''
        ECG Data, as described here: https://physionet.org/content/ptb-xl/1.0.2/.
        
        
        
        Examples
        ---------
        
        .. code-block::
        
            >>> dataset = PTB_XL(
            ...     data_path='../../data/', 
            ...     train=True, 
            ...     source_name='nurse', 
            ...     sampling_rate=500,
            ...     return_sources=False,
            ...     )

        
        
        Arguments
        ---------
        
        - data_path: str, optional:
            The path that the data is saved
            or will be saved. 
            Defaults to :code:`'./'`.
        
        - train: bool, optional:
            Whether to load the training or testing set. 
            Defaults to :code:`True`.
        
        - sampling_rate: typing.Literal[100, 500], optional:
            The sampling rate. This should be
            in :code:`[100, 500]`. 
            Defaults to :code:`100`.
        
        - source_name: typing.Literal['nurse', 'site', 'device'], optional:
            Which of the three attributes should be 
            interpretted as the data sources. This should
            be in  :code:`['nurse', 'site', 'device']`.
            This is ignored if :code:`return_sources=False`.
            Defaults to :code:`'nurse'`.
        
        - return_sources: bool, optional:
            Whether to return the sources alongside
            the data and targets. For example, with 
            :code:`return_sources=True`, for every index
            this dataset will return :code:`data, target, source`. 
            Defaults to :code:`True`.
        
        - binary: bool, optional:
            Whether to return classes based on whether the 
            ecg is normal or not, and so a binary classification
            problem.
            Defaults to :code:`False`.
        
        - subset: bool, optional:
            If :code:`True`, only the first 1000 items
            of the training and test set will be returned.
            Defaults to :code:`False`.
        
        
        '''

        if wfdb_import_error:
            raise ImportError('Please install wfdb before using this dataset.')

        assert sampling_rate in [100, 500], \
            "Please choose sampling_rate from [100, 500]"
        assert type(train) == bool, "Please use train = True or False"
        assert source_name in ['nurse', 'site', 'device'], \
            "Please choose source_name from ['nurse', 'site', 'device']"

        
        self.data_path = data_path
        self.download()
        self.data_path = os.path.join(
            self.data_path, 
            'ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.2/',
            )

        self.train=train
        self.sampling_rate = sampling_rate
        self.source_name = source_name
        self.return_sources = return_sources
        self.binary = binary
        self.meta_data = pd.read_csv(self.data_path+'ptbxl_database.csv')
        self.meta_data['scp_codes'] = (self.meta_data
            ['scp_codes']
            .apply(lambda x: ast.literal_eval(x))
            )
        self.aggregate_diagnostic() # create diagnostic columns
        self.meta_data = self.meta_data[~self.meta_data[self.source_name].isna()]

        if self.train:
            self.meta_data = self.meta_data.query("strat_fold != 10")
            if subset:
                self.meta_data = self.meta_data.iloc[:1000]
        else:
            self.meta_data = self.meta_data.query("strat_fold == 10")
            if subset:
                self.meta_data = self.meta_data.iloc[:1000]
        
        self.targets = self.meta_data[['NORM', 'CD', 'HYP', 'MI', 'STTC']].values
        self.sources = self.meta_data[self.source_name].values

        return

    def _check_exists(self):
        folder = os.path.join(
            self.data_path, 
            'ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.2',
            )
        return os.path.exists(folder)
        
    def download(self):
        
        if self._check_exists():
            print("Files already downloaded.")
            return

        download_and_extract_archive(
            url='https://physionet.org/static'\
                '/published-projects/ptb-xl/'\
                'ptb-xl-a-large-publicly-available'\
                '-electrocardiography-dataset-1.0.2.zip',
            download_root=self.data_path,
            extract_root=self.data_path,
            filename='ptbxl.zip',
            remove_finished=True
            )

        return

    @staticmethod
    def single_diagnostic(y_dict, agg_df):
        tmp = []
        for key in y_dict.keys():
            if key in agg_df.index:
                tmp.append(agg_df.loc[key].diagnostic_class)
        return list(set(tmp))

    def aggregate_diagnostic(self):
        agg_df = pd.read_csv(self.data_path +'scp_statements.csv', index_col=0)
        agg_df = agg_df[agg_df.diagnostic == 1]
        self.meta_data['diagnostic_superclass'] = (self.meta_data
            ['scp_codes']
            .apply(
                self.single_diagnostic, 
                agg_df=agg_df,
                )
            )
        mlb = MultiLabelBinarizer()
        self.meta_data = self.meta_data.join(
            pd.DataFrame(
                mlb.fit_transform(
                    self.meta_data.pop('diagnostic_superclass')
                    ),
                columns=mlb.classes_,
                index=self.meta_data.index,
                )
            )
        return

    def __getitem__(self, index):

        data = self.meta_data.iloc[index]

        if self.sampling_rate == 100:
            f = data['filename_lr']
            x = wfdb.rdsamp(self.data_path+f)
        elif self.sampling_rate == 500:
            f = data['filename_hr']
            x = wfdb.rdsamp(self.data_path+f)
        x = torch.tensor(x[0]).transpose(0,1).float()
        y = torch.tensor(
            data
            [['NORM', 'CD', 'HYP', 'MI', 'STTC']]
            .values
            .astype(np.int64)
            )
        if self.binary:
            y = y[0]
        source = data[self.source_name]

        if self.return_sources:
            return x, y, source
        else:
            return x, y
    
    def __len__(self):
        return len(self.meta_data)
    




# 
# data class -------------------------------
class CIFAR10_N(torch.utils.data.Dataset):
    def __init__(
        self, 
        data_path='./', 
        train=True,
        n_sources=10,
        n_corrupt_sources=4,
        noise_level=0.4,
        seed=None,
        ):

        assert noise_level*n_corrupt_sources/n_sources<=0.4, 'noise_level*n_corrupt_sources/n_sources<=0.4'

        if seed is None:
            rng = np.random.default_rng(None)
            self.seed = rng.integers(low=1, high=1e9, size=1)[0]
        else:
            self.seed = seed
        
        rng = np.random.default_rng(self.seed)

        self.data_path = data_path

        self.train = train
        self.cifar_train_dataset, self.cifar_test_dataset = get_cifar10(data_path)
        if train:
            self.download()
            rng = np.random.default_rng(rng.integers(low=1, high=1e9, size=1)[0])
            self.sources = rng.choice(n_sources, size=len(self.cifar_train_dataset))
            self.labels = torch.load(data_path + 'CIFAR-N/CIFAR-10_human.pt')
            worst_labels = np.asarray(self.labels['worse_label'])
            clean_labels = np.asarray(self.labels['clean_label'])
            wrong_label = worst_labels != clean_labels


            probabilities = np.ones(len(clean_labels))
            probabilities[wrong_label] = noise_level/np.sum(wrong_label)
            probabilities[~wrong_label] = (1-noise_level)/np.sum(~wrong_label)

            self.sources = -np.ones(len(clean_labels))
            rng = np.random.default_rng(rng.integers(low=1, high=1e9, size=1)[0])
            index_source_m = rng.choice(
                np.arange(len(clean_labels)), 
                p=probabilities, 
                replace=False, 
                size=(n_corrupt_sources, len(clean_labels)//n_sources),
                )

            self.sources[index_source_m] = np.arange(n_corrupt_sources).reshape(-1,1)
            rng = np.random.default_rng(rng.integers(low=1, high=1e9, size=1)[0])
            self.sources[self.sources==-1] = rng.choice(
                np.arange(n_corrupt_sources,n_sources), 
                replace=True, 
                size=len(self.sources[self.sources==-1]),
                )

            self.targets = -np.ones(len(clean_labels))
            self.targets[
                np.isin(
                    self.sources, 
                    np.arange(n_corrupt_sources)
                    )] = worst_labels[
                        np.isin(
                            self.sources, 
                            np.arange(n_corrupt_sources)
                            )
                            ]
            self.targets[
                np.isin(
                    self.sources, 
                    np.arange(n_corrupt_sources,n_sources)
                    )] = clean_labels[
                        np.isin(
                            self.sources, 
                            np.arange(n_corrupt_sources,n_sources)
                            )
                            ]
            self.targets = self.targets.astype(np.int64)

    def _check_exists(self):
        folder = os.path.join(
            self.data_path, 
            'CIFAR-N',
            )
        return os.path.exists(folder)
        
    def download(self):
        
        if self._check_exists():
            print("Files already downloaded.")
            return

        download_and_extract_archive(
            url='http://www.yliuu.com/'\
                'web-cifarN/files/CIFAR-N-1.zip',
            download_root=self.data_path,
            extract_root=self.data_path,
            filename='CIFAR-N-1.zip',
            remove_finished=True
            )

        return


    def __getitem__(self,index):
        if self.train:
            return (self.cifar_train_dataset[index][0], 
                self.targets[index],
                self.sources[index]
                )
        else:
            return self.cifar_test_dataset[index]
    
    def __len__(self):
        if self.train:
            return len(self.cifar_train_dataset)
        else:
            return len(self.cifar_test_dataset)

