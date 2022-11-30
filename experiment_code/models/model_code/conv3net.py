from .basemodel import BaseLearningModel, BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import typing


class Conv3Net(BaseModel):
    def __init__(self,
                    input_dim=32, 
                    in_channels=3,
                    channels=32,
                    n_out=10,
                    **kwargs,
                ):
        '''
        A Convolution model class.
        
        
        
        Arguments
        ---------
        
        - input_dim: int`, optional:
            The size of the input in one direction. This
            assumes that the input is a square image. 
            Defaults to :code:`32`.
        
        - in_channels: int`, optional:
            The number of channels in the input.
            Defaults to :code:`3`.

        - channels: int`, optional:
            The number of channels in the first conv layer.
            The second and third conv layers will have 
            double the number of channels.
            Defaults to :code:`32`.
        
        - n_out: int`, optional:
            The size of the output. :code:`2` should be
            used for binary classification. 
            Defaults to :code:`10`.

        '''

        super(Conv3Net, self).__init__()

        self.input_dim = input_dim
        self.channels = channels
        self.n_out = n_out

        # =============== Cov Network ===============
        self.net = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels, self.channels, 3, padding='valid')),
            ('relu1', nn.ReLU()),
            ('mp1', nn.MaxPool2d(2, 2)),
            ('conv2', nn.Conv2d(self.channels, self.channels*2, 3, padding='valid')),
            ('relu2', nn.ReLU()),
            ('mp2', nn.MaxPool2d(2, 2)),
            ('conv3', nn.Conv2d(self.channels*2, self.channels*2, 3, padding='valid')),
            ('relu3', nn.ReLU()),
            ('flatten', nn.Flatten()),
            ]))

        # =============== Linear ===============
        self.pm_fc = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(self.size_of_dim_out(self.input_dim)**2*(self.channels*2), 64)),
            ('relu1', nn.ReLU()),
            ]))

        # =============== Classifier ===============
        self.pm_clf = nn.Linear(64, n_out)
        #self.softmax = nn.LogSoftmax(dim=1)
        #self.softmax = nn.Softmax(dim=1)

        return

    def size_of_dim_out(self, dim_in):

        out = self._resolution_calc(dim_in=dim_in, **self._get_conv_params(self.net.conv1))
        out = self._resolution_calc(dim_in=out, **self._get_conv_params(self.net.mp1))
        out = self._resolution_calc(dim_in=out, **self._get_conv_params(self.net.conv2))
        out = self._resolution_calc(dim_in=out, **self._get_conv_params(self.net.mp2))
        out = self._resolution_calc(dim_in=out, **self._get_conv_params(self.net.conv3))

        return out


    def forward(self, X):
        out = self.pm_clf(
                self.pm_fc(
                    self.net(X)))
        return out







class Conv3NetLearning(BaseLearningModel, Conv3Net):
    def __init__(self, 
                    input_dim:int=32, 
                    in_channels:int=3,
                    channels:int=64,
                    n_out:int=10,
                    device:str='auto',
                    n_epochs:int=100,
                    train_optimizer:dict={'adam_lap': {'params':['all'], 
                                                'lr':0.01, 
                                                'lap_n': 20,
                                                'depression_function': 'discrete_ranking_std',
                                                'depression_function_kwargs': {}, 
                                                'depression_strength': 0, 
                                                },
                                    },
                    train_criterion:str='CE',
                    verbose:bool=True,
                    model_path:typing.Union[None, str]=None, 
                    result_path:typing.Union[None, str]=None, 
                    model_name:str='',
                    seed:typing.Union[None, int]=None,
                    source_fit:bool=True,
                    **kwargs,
                    ):
        '''
        This class wraps the MLP model to allow for fitting
        and predicting.
        
        
        Arguments
        ---------
        
        - input_dim: int`, optional:
            The size of the input in one direction. This
            assumes that the input is a square image. 
            Defaults to :code:`32`.
        
        - in_channels: int`, optional:
            The number of channels in the input.
            Defaults to :code:`3`.

        - channels: int`, optional:
            The number of channels in the first conv layer.
            The second and third conv layers will have 
            double the number of channels.
            Defaults to :code:`32`.
        
        - n_out: int`, optional:
            The size of the output. :code:`2` should be
            used for binary classification. 
            Defaults to :code:`10`.

        - n_epochs: int`, optional:
            The number of epochs to train the model for. 
            Defaults to :code:`100`.
        
        - train_optimizer: dict`, optional:
            The dictionary containing the optimizer and params to 
            train with. 
            If using a dictionary:
            Supply the optimizer name as keys, and dictionaries of 
            keywords as values. An example is:
            :code:`
            {'adam_lap': {
                            'params':['all'], 
                            'lr':0.01, 
                            'lap_n': 20,
                            'depression_function': 'min_max_mean',
                            'depression_function_kwargs': {},
                            'depression_strength': 0, 
                            },
            }
            :code:`
            The values may also be a list of optimizer keywords
            that will be used as different parameter groups in the
            optimizer.
            Defaults to :code:`
            {'adam_lap': {
                            'params':['all'], 
                            'lr':0.01, 
                            'lap_n': 20,
                            'depression_function': 'min_max_mean',
                            'depression_function_kwargs': {},
                            'depression_strength': 0, 
                            },
            }
            :code:`.
        
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
        super(Conv3NetLearning, self).__init__(
                    input_dim=input_dim, 
                    in_channels=in_channels,
                    channels=channels,
                    n_out=n_out,
                    n_epochs=n_epochs,
                    train_optimizer=train_optimizer, 
                    train_criterion=train_criterion,
                    verbose=verbose,
                    device=device,
                    model_path=model_path, 
                    result_path=result_path, 
                    model_name=model_name,
                    seed=seed,
                    source_fit=source_fit,
                    **kwargs,
                    )

        self.device = ('cuda' if torch.cuda.is_available() else 'cpu') if device =='auto' else device
        self.model_name = model_name
        self.result_path = result_path
        self.model_path = model_path
        self.source_fit=source_fit

    def fit(self,
            train_loader, 
            metrics_track=['accuracy'],
            val_loader=None,
            **kwargs,
            ):
        super(Conv3NetLearning, self).fit(
                            train_loader=train_loader, 
                            metrics_track=metrics_track,
                            val_loader=val_loader,
                            **kwargs,
                            )
        return self






