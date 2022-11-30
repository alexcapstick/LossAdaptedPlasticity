import torch
import time
from ..utils.utils import PBar
import sys



class ModelTesting:
    def __init__(self,
                    model,
                    device:str='auto',
                    verbose:bool=True,
                    ):
        '''

        This class allows for the testing of models.
        
        
        Arguments 
        ---------

        - model: pytorch model:
            The model to be tested.
        
        - device: str (optional):
            The device for the model and data to be 
            loaded to and used during testing.
            Defaults to :code:`'auto'`.

        - verbose: bool (optional):
            Whether to print information about progress during testing.
            Defaults to :code:`True`.

        '''
        
        self.model = model
        self.device = ('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device
        self.verbose = verbose

        return
    

    @torch.no_grad()
    def predict(self, 
                test_loader:torch.utils.data.DataLoader, 
                targets_too:bool=False):
        '''
        This function allows for inference on a dataloader of 
        test data. The predictions will be returned as a tensor.

        Arguments
        ---------

        - test_loader: torch.utils.data.DataLoader:
            This is the test data loader that contains the test data.
            If this data loader contains the inputs as well as 
            the outputs, then make sure to set the argument
            :code:`targets_too=True`. If the targets are included
            as well, make sure that each iteration of the dataloader
            returns (inputs, targets).

        - targets_too: bool, (optional):
            This dictates whether the dataloader contains the targets
            as well as the inputs.
        
        Returns
        ---------

        - predictions: torch.tensor:
            The outputs of the model for each of the inputs
            given in the :code:`test_loader`.

        '''
        self.model.to(self.device) # move model to device
        # print error bar if verbose is True
        predict_bar = PBar(show_length = 20, n_iterations=len(test_loader))
        predict_bar['Batch'] = '-'
        predict_bar['Took'] = '-s'
        printing_statement = 'Predicting:  {}'.format(predict_bar.give())
        if self.verbose:
            sys.stdout.write('\r')
            sys.stdout.write(printing_statement)
            sys.stdout.flush()
        self.model.eval() # move model to eval mode
        output = []
        start_predict_time = time.time() # start time of inference
        cumulative_batch_time = 0 # value for us to calculate the average inference time
        for nb, inputs in enumerate(test_loader):
            start_batch_time = time.time()
            if targets_too:
                inputs = inputs[0]
            else:
                pass
            inputs = inputs.to(self.device)
            output.append(self.model(inputs))
            end_batch_time = time.time()
            predict_bar.update(1)

            batch_time = end_batch_time-start_batch_time
            cumulative_batch_time += batch_time

            predict_bar['Batch'] = '{}'.format(nb+1)
            predict_bar['Took'] = '{:.2f}'.format(cumulative_batch_time/(nb+1))
            printing_statement = 'Predicting:  {}'.format(predict_bar.give())

            if self.verbose and ((nb+1)%10 == 0):
                sys.stdout.write('\r')
                sys.stdout.write(printing_statement)
                sys.stdout.flush()

        output = torch.cat(output).detach().cpu()
        end_predict_time = time.time()
        self.model.train()

        predict_bar['Predict Took'] = '{:.1e}s'.format(end_predict_time-start_predict_time)
        printing_statement = 'Predicting:  {}'.format(predict_bar.give())

        if self.verbose:
            sys.stdout.write('\r')
            sys.stdout.write(printing_statement)
            sys.stdout.flush()
            sys.stdout.write('\n')

        return output