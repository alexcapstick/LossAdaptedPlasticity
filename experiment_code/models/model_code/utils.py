from .mlp import *
from .conv3net import *
from .resnet import *


def get_model(model_config:dict):
    '''
    Allows you to get a model class by its name.
    
    
    
    Arguments
    ---------
    
    - model_config: dict: 
        This should contain the key
        :code:`'model_name'`.
    
    
    Raises
    ---------
    
        :code:`NotImplementedError`: If the model name
        is not yet implemented as a string.
    
    Returns
    --------
    
    - The model class.
    
    
    '''

    name = model_config['model_name']

    if 'MLP' in name:
        model_class = MLPLearning
    
    elif 'Conv3Net' in name:
        model_class = Conv3NetLearning
    
    elif 'ResNet' in name:
        model_class = ResNetLearning
    
    else:
        raise NotImplementedError('The model type {} is not yet '\
                                    'implemented using a string.'.format(name))

    return model_class