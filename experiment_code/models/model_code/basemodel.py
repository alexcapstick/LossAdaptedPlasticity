import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from ...training_utils.fitting_classes import BasicModelFitter
from ...training_utils.optimizers import CombineOptimizers
from loss_adapted_plasticity.lap_wrapper import LAP
from ...testing_utils.testing_classes import ModelTesting
import typing


def get_optimizer_from_name(name):
    '''
    Get optimizer from name.
    
    
    
    Arguments
    ---------
    
    - name: str: 
        This can be any of :code:`'adam'`,
        :code:`'adadelta'`, :code:`'sgd'`,
        or any of the names of the optimisers 
        in :code:`torch.optim`.
    
    
    Raises
    ---------
    
        :code:`NotImplementedError: If optimizer name
        is not implemented.
    
    Returns
    --------
    
    - optimizer: torch.optim optimizer.
    
    
    '''
    if name == 'adam':
        return torch.optim.Adam
    elif name =='adadelta':
        return torch.optim.Adadelta
    elif name == 'sgd':
        return torch.optim.SGD
    else:
        return getattr(torch.optim, name)


def get_criterion(name):
    '''
    Get loss function from name.
    
    
    
    Arguments
    ---------
    
    - name: str: 
        This can be any of :code:`'CE'`,
        :code:`'MSE'`. It can also be 
        any of the names of the classes 
        associated with :code:`torch.nn`.
    
    
    Raises
    ---------
    
    - :code:`NotImplementedError: If loss name
        is not implemented.
    
    Returns
    --------
    
    - loss_function.
    
    
    '''
    if name == 'CE':
        return nn.CrossEntropyLoss() 
    elif name == 'MSE':
        return nn.MSELoss()
    else:
        return getattr(nn, name)



class BaseModel(nn.Module):
    '''
    Base model, for meta or non meta learning.
    This class contains functions that can help in building 
    and training models.
    
    '''
    def __init__(self, 
                    **kwargs,
                    ):
        super(BaseModel, self).__init__()

        self.meta_training = False
        self.validating = False
        self.testing = False
        self.in_epoch = False
        self.in_batch = False
        self.traditional_training = False



    def _resolution_calc(self, dim_in:int, kernel_size:int=3, 
                            stride:int=1, padding:int=0, dilation:int=1):
        '''
        Allows the calculation of resolutions after a convolutional layer.
        
        
        
        Arguments
        ---------
        
        - dim_in: int: 
            The dimension of an image before convolution is applied.
            If dim_in is a :code:`list` or :code:`tuple`, then two dimensions
            will be returned.
        
        - kernel_size: int`, optional:
            Defaults to :code:`3`.
        
        - stride: int`, optional:
            Defaults to :code:`1`.
        
        - padding: int`, optional:
            Defaults to :code:`0`.
        
        - dilation: int`, optional:
            Defaults to :code:`1`.
        
        
        Returns
        --------
        
        - dim_out: int: 
            The dimension size after the convolutional layer.
        
        
        '''
        if padding == 'valid':
            padding=0

        if type(dim_in) == list or type(dim_in) == tuple:
            out_h = dim_in[0]
            out_w = dim_in[1]
            out_h = (out_h + 2*padding - dilation * (kernel_size-1) - 1)/stride + 1
            out_w = (out_w + 2*padding - dilation * (kernel_size-1) - 1)/stride + 1

            return (out_h, out_w)
        
        return int(np.floor((dim_in + 2*padding -  (kernel_size - 1) - 1)/stride + 1))
    
    def _get_conv_params(self, layer):
        '''
        Given a pytorch Conv2d layer, this function can
        return a dictionary of the kernel size, stride
        and padding.
        
        
        
        Arguments
        ---------
        
        - layer: torch.nn.Conv2d: 
            Pytorch convolutional layer.       
        
        
        Returns
        --------
        
        - params: dict: 
            Dictionary containing the parameters of 
            the convolutional layer.
        
        
        '''
        kernel_size = layer.kernel_size[0] if type(layer.kernel_size) == tuple else layer.kernel_size
        stride = layer.stride[0] if type(layer.stride) == tuple else layer.stride
        padding = layer.padding[0] if type(layer.padding) == tuple else layer.padding
        return {'kernel_size': kernel_size, 'stride': stride, 'padding': padding}
    


    # the following allow for specific options during training.

    def epoch_start(self, obj=None, **kwargs):
        self.in_epoch = True
        return
    
    def batch_start(self, obj=None, **kwargs):
        self.in_batch = True
        return
    
    def batch_end(self, obj=None, **kwargs):
        self.in_batch = False
        return
    
    def val_start(self, obj=None, **kwargs):
        self.validating = True
        return
    
    def val_end(self, obj=None, **kwargs):
        self.validating = False
        return      

    def epoch_end(self, obj=None, **kwargs):
        self.in_epoch = False
        return

    def test_start(self, obj=None, **kwargs):
        self.testing = True
        return
    
    def traditional_train_start(self, obj=None, **kwargs):
        self.traditional_training = True
        return

    def traditional_train_end(self, obj=None, **kwargs):
        self.traditional_training = False
        return

    def test_end(self, obj=None, **kwargs):
        self.testing = False
        return

























class BaseLearner(object):
    def __init__(self, **kwargs):
        '''
        This is the base class for model training and 
        predicting and contains methods
        for getting optimizers, parameters and criterions.
        
        '''

        super(BaseLearner, self).__init__(**kwargs)

        return

    def _get_params_from_names(self, names):
        '''
        Returns the layer parameters from the layers name.
        This is used to specify optimizer layers based on the
        name of the layers.
        
        
        
        Arguments
        ---------
        
        - names: str: 
            Layer name. If :code:`'all'`, all layers will be returned.
        
        
        Raises
        ---------
        
        - :code:`TypeError: If layer name is not an attribute of the model.
        
        Returns
        --------
        
        - layer_params`.
        
        
        '''
        params = []
        for name in names:
            if hasattr(self, name):
                params += list(getattr(self, name).parameters())
            elif name == 'all':
                params += list(self.parameters())
            else:
                raise TypeError('There is no such parameter name: {}'.format(name))
        return params

    def _get_optimizer(self, opt_dict):
        '''
        Returns an optimizer, initiated with keywords.

        
        Arguments
        ---------
        
        - opt_dict: dict: 
            The optimizer name as keys, and dictionaries of 
            keywords as values. An example is::

                {'adam_lap': {
                                'params':['all'], 
                                'lr':0.01, 
                                'lap_n': 20,
                                'depression_function': 'min_max_mean',
                                'depression_function_kwargs': {}
                                },
                }

            The values may also be a list of optimizer keywords
            that will be used as different parameter groups in the
            optimizer.
        
        Raises
        ---------
        
        - NotImplementedError: 
            If the values are not dictionaries or a list.
        
        Returns
        --------
        
        - Single :code:`optimizer` or list of :code:`optimizer`s,
        depending on the number of optimizers given in the 
        :code:`opt_dict`.
        
        
        '''
        optimizer_list = []
        for optimizer_name, optimizer_kwargs in opt_dict.items():
            if '_lap' in optimizer_name:
                lap_wrapper=True
                optimizer_class = get_optimizer_from_name(
                    optimizer_name.replace('_lap', '')
                    )
            else:
                optimizer_class = get_optimizer_from_name(optimizer_name)
            if type(optimizer_kwargs) == dict:
                params = self._get_params_from_names(optimizer_kwargs['params'])
                if lap_wrapper:
                    optimizer = LAP(
                        optimizer_class, 
                        params=params,
                        **{k:v for k,v in opt_dict[optimizer_name].items() if k != 'params'}
                        )
                else:
                    optimizer = optimizer_class(params=params, 
                         **{k:v for k,v in opt_dict[optimizer_name].items() if k != 'params'})
            elif type(optimizer_kwargs) == list:
                param_groups = []
                for group in optimizer_kwargs:
                    group['params'] = self._get_params_from_names(group['params'])
                    param_groups.append(group)
                if lap_wrapper:
                    raise NotImplemented("LAP learning is not implemented with param groups")
                else:
                    optimizer = optimizer_class(param_groups)
            else:
                raise NotImplementedError('Either pass a dictionary or list to the optimizer keywords')
            
            optimizer_list.append(optimizer)

        if len(optimizer_list) == 1:
            return optimizer_list[0]
        else:
            return CombineOptimizers(*optimizer_list)
    
    def _get_criterion(self, name):
        '''
        Function that allows you to get the loss function
        by name.       
        
        
        Arguments
        ---------
        
        - name: str: 
            The name of the loss function.
        

        Returns
        --------
        
        - criterion`.
        
        '''
        return get_criterion(name)

























class BaseLearningModel(BaseLearner):
    def __init__(self, 
                    n_epochs=100,
                    train_optimizer={
                        'adam': {
                            'params': ['all'],
                            'lr': 0.01,
                            },
                        },
                    train_criterion='CE',
                    verbose=True,
                    device='auto',
                    model_path:typing.Union[None, str]=None, 
                    result_path:typing.Union[None, str]=None, 
                    model_name:str='',
                    source_fit:bool=False,
                    seed:typing.Union[None, int]=None,
                    **kwargs,
                    ):
        '''
        This is the base class for training and testing
        models.
        It allows a model to have the attributes :code:`.fit`
        and :code:`.predict`.
        

        Arguments
        ---------
        
        - n_epochs: int`, optional:
            The number of epochs to train the model for. 
            Defaults to :code:`100`.
        
        - train_optimizer: dict`, optional:
            The dictionary containing the optimizer and params to 
            train with. 
            If using a dictionary:
            Supply the optimizer name as keys, and dictionaries of 
            keywords as values. An example is::

                {'adam_lap': {
                    'params':['all'], 
                    'lr':0.01, 
                    'lap_n': 20,
                    'depression_function': 'min_max_mean',
                    'depression_function_kwargs': {}
                    },

            The values may also be a list of optimizer keywords
            that will be used as different parameter groups in the
            optimizer.
            Defaults to :code:`{'adam': {'params': ['all'], 'lr': 0.01, }, }`.
        
        - train_criterion: str`, optional:
            The name of the loss function to use. 
            This can be any of :code:`'CE'` or :code:`'MSE'`.
            Defaults to :code:`'CE'`.
        
        - verbose: bool`, optional:
            Whether to print information as the model
            is training or predicting. 
            Defaults to :code:`True`.
        
        - device: str`, optional:
            The device to perform the training and 
            testing on. If :code:`auto`, the training
            will perform on :code:`'cuda'` if available. 
            Defaults to :code:`'auto'`.
        
        - model_path: typing.Union[None, str]`, optional:
            The path to save the model to, after training is complete.
            The model and state dict will be saved in this path, using
            the model_name given. 
            Defaults to :code:`None`.
        
        - result_path: typing.Union[None, str]`, optional:
            The path to save the loss values and the graph showing
            the loss whilst training. 
            Defaults to :code:`None`.
        
        - model_name: str`, optional:
            The model name, used when saving the model and the 
            results. 
            Defaults to :code:`''`.
        
        - source_fit: bool`, optional:
            Whether the model should be fitted with the sources.
            This is used for LAP training. 
            Defaults to :code:`False`.
        
        - seed: typing.Union[None, int]`, optional:
            The random seed to be used for random operations. 
            Defaults to :code:`None`.
            
        '''

        if seed is None:
            seed = np.random.randint(0,1e9)

        torch.manual_seed(seed)

        super(BaseLearningModel, self).__init__(**kwargs)

        self.train_optimizer = train_optimizer
        self.train_criterion = train_criterion
        self.verbose = verbose
        self.n_epochs = n_epochs
        self.device = ('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device
        self.model_path = model_path
        self.result_path = result_path
        self.model_name = model_name
        self.fitted = False
        self.fitting_class = None
        self.source_fit=source_fit

        self.built_training_method = False

        return

    def _build_training_methods(self):
        # train optimizer
        if type(self.train_optimizer) == dict:
            self.train_optimizer = self._get_optimizer(self.train_optimizer)
        else:
            pass

        if type(self.train_criterion) == str:
            self.train_criterion = self._get_criterion(self.train_criterion)
        else:
            pass
        
        self.built_training_method = True

        return


    def fit(self,
            train_loader:torch.utils.data.DataLoader, 
            metrics_track:typing.List[str]=[],
            val_loader:typing.Union[None, torch.utils.data.DataLoader]=None,
            train_scheduler=None,
            **kwargs,
            ): 
        '''
        Method for fitting a model. This wraps a model fitting
        class, BasicModelFitter, and its corresponding source fitting
        version.
        
        
        
        Arguments
        ---------
        
        - train_loader: torch.utils.data.DataLoader: 
            The training data loader to be used to train the model.
        
        - metrics_track: list`, (optional):
            List of metrics to track in the training.
            To see the list of supported metrics, see 
            the documentation for BasicModelFitter. 
            Defaults to :code:`[]`.
        
        - val_loader: torch.utils.data.DataLoader`, (optional):
            The data loader for the validation data. 
            Defaults to :code:`None`.
        
        - train_scheduler: torch.optim.lr_scheduler`, (optional):
            Learning rate scheduler for training. 
            Defaults to :code:`None`.
        
        
        
        Returns
        --------
        
        - self`
        
        
        '''

        self._build_training_methods()
        self.to(self.device)
        if self.fitting_class is None:
            writer = SummaryWriter(comment='-'+self.model_name)

            self.fitting_class = BasicModelFitter(
                model=self, 
                device=self.device, 
                verbose=self.verbose, 
                model_name=self.model_name,
                result_path=self.result_path,
                model_path=self.model_path,
                metrics_track=metrics_track,
                writer=writer,
                )

        self = self.fitting_class.fit(
            train_loader=train_loader,
            n_epochs=self.n_epochs,
            criterion=self.train_criterion,
            optimizer=self.train_optimizer,
            val_loader=val_loader,
            train_scheduler=train_scheduler,
            source_fit=self.source_fit,
            **kwargs,
            )

        self.to('cpu')
        self.fitted = True
        return self
    
    def predict(self,
                test_loader:torch.utils.data.DataLoader, 
                targets_too:bool=True,
                ):
        '''
        Method for making predictions on a test loader.
        
        Arguments
        ---------
        
        - test_loader: torch.utils.data.DataLoader: 
            A data loader containing the test data.
        
        - targets_too: bool`, optional:
            Dictates whether the test loader contains the targets too. 
            Defaults to :code:`True`.
        
        
        Returns
        --------
        
        - output: torch.tensor: 
            The resutls from the predictions
        
        
        '''
        
        self.mt = ModelTesting(
            model=self,
            device=self.device,
            verbose=self.verbose,
            )
        
        output = self.mt.predict(test_loader=test_loader, targets_too=targets_too)

        return output






