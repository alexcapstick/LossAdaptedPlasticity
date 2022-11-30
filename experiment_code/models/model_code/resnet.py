import torch.nn as nn
from .basemodel import BaseLearningModel
from .basemodel import BaseModel
import torch
import typing
import numpy as np



class ResBlock(nn.Module):
    def __init__(self,
        input_dim:int, 
        input_channels:int,
        out_channels:int,
        out_dim:int,
        kernel_size:int=3,
        dropout_rate:float=0.2,
        ):
        super(ResBlock, self).__init__()
        self.input_dim = input_dim
        self.kernel_size = kernel_size
        self.input_channels = input_channels
        self.out_channels = out_channels
        self.out_dim = out_dim
        
        self.x1 = nn.Sequential(
            nn.Conv1d(
                input_channels, 
                out_channels,
                kernel_size=kernel_size,
                bias=False,
                padding='same',
                ),
            nn.BatchNorm1d(
                num_features=out_channels,
                affine=False,
                ),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
            )

        self.x2 = nn.Sequential(
            nn.Conv1d(
                in_channels=out_channels, 
                out_channels=out_channels,
                kernel_size=kernel_size,
                bias=False,
                stride=input_dim//out_dim,
                )
            )
        
        self.y1 = nn.Sequential(
            nn.Conv1d(
                in_channels=input_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding='same',
                bias=False,
                )
            )
        
        self.xy1 = nn.Sequential(
            nn.BatchNorm1d(
                num_features=out_channels,
                affine=False,
                ),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
            )

        return

    def _skip_connection(self, y):
        downsample = self.input_dim//self.out_dim
        if downsample > 1:

            same_pad = np.ceil(
                0.5*((y.size(-1)//self.out_dim)*(self.out_dim-1) - y.size(-1) + downsample)
                )
            if same_pad < 0:
                same_pad = 0
            y = nn.functional.pad(y, (int(same_pad), int(same_pad)), "constant", 0)
            y = nn.MaxPool1d(
                kernel_size=downsample,
                stride=downsample,
                )(y)
        
        elif downsample == 1:
            pass
        else:
            raise ValueError("Size of input should always decrease.")
        y = self.y1(y)
        
        return y

    def forward(self, inputs):
        x, y = inputs

        # y
        y = self._skip_connection(y)

        # x
        x = self.x1(x)
        same_pad = np.ceil(
            0.5*((x.size(-1)//self.out_dim)*(self.out_dim-1) - x.size(-1) + self.kernel_size)
            )
        if same_pad < 0:
            same_pad = 0
        x = nn.functional.pad(x, (int(same_pad), int(same_pad)), "constant", 0)
        x = self.x2(x)

        # xy
        xy = x + y
        y = x
        xy = self.xy1(xy)

        return [xy, y]


class ResNet(BaseModel):
    def __init__(
        self,
        input_dim:int=4096,
        input_channels:int=64,
        n_output:int=10,
        kernel_size:int=16,
        dropout_rate:float=0.2,
        ):
        '''
        Model with 4 :code:`ResBlock`s, in which
        the number of channels increases linearly
        and the output dimensions decreases
        exponentially. This model will
        require the input dimension to be of at least 
        256 in size. This model is designed for sequences,
        and not images. The expected input is of the type::

            [n_batches, n_filters, sequence_length]


        Examples
        ---------
        
        .. code-block::
        
            >>> model = ResNet(
                    input_dim=4096,
                    input_channels=64,
                    kernel_size=16,
                    n_output=5,
                    dropout_rate=0.2,
                    )
            >>> model(
                    torch.rand(1,64,4096)
                    )
            tensor([[0.3307, 0.4782, 0.5759, 0.5214, 0.6116]], grad_fn=<SigmoidBackward0>)
        
        
        Arguments
        ---------
        
        - input_dim: int, optional:
            The input dimension of the input. This
            is the size of the final dimension, and 
            the sequence length.
            Defaults to :code:`4096`.
        
        - input_channels: int, optional:
            The number of channels in the input.
            This is the second dimension. It is the
            number of features for each sequence element.
            Defaults to :code:`64`.
        
        - n_output: int, optional:
            The number of output classes in 
            the prediction. 
            Defaults to :code:`10`.
        
        - kernel_size: int, optional:
            The size of the kernel filters
            that will act over the sequence. 
            Defaults to :code:`16`.       
        
        - dropout_rate: float, optional:
            The dropout rate of the ResNet
            blocks. This should be a value
            between :code:`0` and  :code:`1`.
            Defaults to :code:`0.2`.     
        
        '''
        super(ResNet, self).__init__()

        self.x1 = nn.Sequential(
            nn.Conv1d(
                in_channels=input_channels,
                out_channels=input_channels,
                kernel_size=kernel_size,
                stride=1,
                padding='same',
                bias=False,
                ),
            nn.BatchNorm1d(
                num_features=input_channels,
                affine=False,
                ),
            nn.ReLU(),
        )

        self.x2 = nn.Sequential(
            ResBlock(
                input_dim=input_dim, # 4096
                input_channels=input_channels, # 64
                out_channels=2*input_channels//1, # 128
                kernel_size=kernel_size, # 16
                out_dim=input_dim//4, # 1024,
                dropout_rate=dropout_rate,
                ),

            ResBlock(
                input_dim=input_dim//4, # 1024
                input_channels=2*input_channels//1, # 128
                out_channels=3*input_channels//1, # 192
                kernel_size=kernel_size, # 16
                out_dim=input_dim//16, # 256
                dropout_rate=dropout_rate,
                ),

            ResBlock(
                input_dim=input_dim//16, # 256
                input_channels=3*input_channels//1, # 192
                out_channels=4*input_channels//1, # 256
                kernel_size=kernel_size, # 16
                out_dim=input_dim//64, # 64
                dropout_rate=dropout_rate,
                ),

            ResBlock(
                input_dim=input_dim//64, # 64
                input_channels=4*input_channels//1, # 256
                out_channels=5*input_channels//1, # 320
                kernel_size=kernel_size, # 16
                out_dim=input_dim//256, # 16
                dropout_rate=dropout_rate,
                ),
            )

        self.x3 = nn.Flatten()
        self.x4 = nn.Sequential(        
            nn.Linear(
                (input_dim//256) * (5*input_channels//1),
                n_output,
                )
            )

    def forward(self, x):
        
        x = self.x1(x)
        x, _ = self.x2([x,x])
        x = self.x3(x)
        x = self.x4(x)

        return x


class ResNetLearning(BaseLearningModel, ResNet):
    def __init__(
        self,
        input_dim:int=4096,
        input_channels:int=64,
        n_output:int=10,
        kernel_size:int=16,
        dropout_rate:float=0.2,
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
        This class wraps the ResNet model to allow for fitting
        and predicting.
        

        Arguments
        ---------
        
        - input_dim: int, optional:
            The input dimension of the input. This
            is the size of the final dimension, and 
            the sequence length.
            Defaults to :code:`4096`.
        
        - input_channels: int, optional:
            The number of channels in the input.
            This is the second dimension. It is the
            number of features for each sequence element.
            Defaults to :code:`64`.
        
        - n_output: int, optional:
            The number of output classes in 
            the prediction. 
            Defaults to :code:`10`.
        
        - kernel_size: int, optional:
            The size of the kernel filters
            that will act over the sequence. 
            Defaults to :code:`16`.       
        
        - dropout_rate: float, optional:
            The dropout rate of the ResNet
            blocks. This should be a value
            between :code:`0` and  :code:`1`.
            Defaults to :code:`0.2`.     

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
                            'depression_function_kwargs': {}
                            },
            }
            :code:`
            The values may also be a list of optimizer keywords
            that will be used as different parameter groups in the
            optimizer.
            Defaults to::

                {
                    'adam_lap': {
                        'params':['all'], 
                        'lr':0.01, 
                        'lap_n': 20,
                        'depression_function': 'min_max_mean',
                        'depression_function_kwargs': {},
                        'depression_strength': 0, 
                        },
                    }
        
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
        super(ResNetLearning, self).__init__(
            input_dim=input_dim,
            input_channels=input_channels,
            n_output=n_output,
            kernel_size=kernel_size,
            dropout_rate=dropout_rate,
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

        self.device = ('cuda' if torch.cuda.is_available() else 'cpu') \
            if device == 'auto' else device
        self.model_name = model_name
        self.result_path = result_path
        self.model_path = model_path
        self.source_fit=source_fit

    def fit(self,
        train_loader, 
        metrics_track=['accuracy'],
        val_loader=None,
        **kwargs
        ):
        '''
        Method for fitting the model.
        
        Arguments
        ---------
        
        - train_loader: torch.utils.data.DataLoader: 
            The training data loader to be used to train the model.
        
        - metrics_track: list`, optional:
            List of metrics to track in the training.
            To see the list of supported metrics, see 
            the documentation for BasicModelFitter. 
            Defaults to :code:`[]`.
        
        - val_loader: torch.utils.data.DataLoader`, optional:
            The data loader for the validation data. 
            Defaults to :code:`None`.
        
        
        Returns
        --------
        
        - self`

        '''
        super(ResNetLearning, self).fit(
            train_loader=train_loader, 
            metrics_track=metrics_track,
            val_loader=val_loader,
            **kwargs,
            )
        return self
