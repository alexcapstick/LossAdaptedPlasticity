from .basemodel import BaseLearningModel, BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import typing


class MLP(BaseModel):
    def __init__(self,
                    in_features=100, 
                    out_features=100,
                    hidden_layer_features=(100,),
                    dropout=0.2,
                    **kwargs,
                ):
        '''
        A Multilayer Perceptron class.
        
        
        
        Arguments
        ---------
        
        - in_features: int`, optional:
            The size of the input. 
            Defaults to :code:`100`.
        
        - out_features: int`, optional:
            The size of the output. :code:`2` should be
            used for binary classification. 
            Defaults to :code:`100`.
        
        - hidden_layer_features: tuple`, optional:
            The sizes of the hidden layers. An example would be
            :code:`[100,100,100]`, which will create three hidden
            layers before the output layer. 
            Defaults to :code:`(100,)`.
        
        - dropout: float`, optional:
            The dropout value to use in each
            of the hidden layers. 
            Defaults to :code:`0.2`.

        '''

        super(MLP, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout

        self.hidden_layer_features = hidden_layer_features

        in_out_list = [in_features] + list(self.hidden_layer_features) + [out_features] 
        
        in_list = in_out_list[:-1][:-1]
        out_list = in_out_list[:-1][1:]


        # =============== Linear ===============
        self.layers = nn.ModuleList([nn.Sequential(
                                            nn.Linear(in_value, out_value), 
                                            nn.Dropout(self.dropout),
                                            nn.ReLU(),
                                            )
                                            for in_value, out_value in zip(in_list, out_list)])

        # =============== Classifier ===============
        self.clf = nn.Linear(in_out_list[-2], in_out_list[-1])
        self.softmax = nn.Softmax(dim=1)


    def reset_classifier(self, class_to_reset=None):
        '''
        This function resets the weights incoming to a class on the
        predictive layer.
        '''
        device = self.clf.weight.device
        
        if class_to_reset is None:
            torch.nn.init.kaiming_normal_(self.clf.weight)
            self.clf.bias.data = torch.ones(([self.out_features]), device=device)
        else:
            torch.nn.init.kaiming_normal_(self.clf.weight[class_to_reset].unsqueeze(0))
            self.clf.bias.data[class_to_reset] = torch.ones(([1]), device=device)

        return


    def forward(self, X):

        out = X
        for layer in self.layers:
            out = layer(out)
        out = self.clf(out)
        out = self.softmax(out)

        return out







class MLPLearning(BaseLearningModel, MLP):
    def __init__(self, 
                    in_features:int=100, 
                    out_features:int=100,
                    hidden_layer_features:typing.Union[tuple, list]=(100,),
                    dropout:float=0.2,
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
        
        - in_features: int`, optional:
            The size of the input. 
            Defaults to :code:`100`.
        
        - out_features: int`, optional:
            The size of the output. :code:`2` should be
            used for binary classification. 
            Defaults to :code:`100`.
        
        - hidden_layer_features: tuple`, optional:
            The sizes of the hidden layers. An example would be
            :code:`[100, 100, 100]`, which will create three hidden
            layers before the output layer. 
            Defaults to :code:`(100,)`.
        
        - dropout: float`, optional:
            The dropout value to use in each
            of the hidden layers. 
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
        super(MLPLearning, self).__init__(
                    in_features=in_features, 
                    out_features=out_features, 
                    hidden_layer_features=hidden_layer_features, 
                    dropout=dropout, 
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
        super(MLPLearning, self).fit(
                            train_loader=train_loader, 
                            metrics_track=metrics_track,
                            val_loader=val_loader,
                            **kwargs,
                            )
        return self






