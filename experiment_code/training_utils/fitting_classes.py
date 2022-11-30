import numpy as np
import torch
import sys
from ..utils.utils import PBar
import time
import seaborn as sns
sns.set('talk')
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import typing



def loss_plot(loss, val_loss = None, n_epochs = None):
    '''
    For an array of loss values, this function will plot
    a simple loss plot of the results.

    Arguments
    ---------

    - loss: list` or :code:`np.array:
        The loss values to plot. Each value
        should represent the loss on that step.
    
    - val_loss: list` or :code:`np.array:
        The validation loss values to plot. Each value
        should represent the validation loss on that step.
        If :code:`None`, no validation loss line will be drawn.
        Defaults to :code:`None`.

    - n_epochs: int:
        The total number of steps that the model
        will be trained for. This is used to set the 
        bounds on the figure. If :code:`None`, then
        the limits will be based on the data given.
        Defaults to :code:`None`.

    Returns
    ---------

    - fig: matplotlib.pyplot.figure:
        The figure containing the axes, which contains
        the plot.
    
    - ax: matplotlib.pyplot.axes:
        The axes containing the plot.


    '''

    # set the plotting area
    fig, ax = plt.subplots(1,1,figsize = (15,5))
    # plot the data
    ax.plot(np.arange(len(loss))+1,loss, label = 'Training Loss')
    if not val_loss is None:
        ax.plot(np.arange(len(val_loss))+1, val_loss, label = 'Validation Loss')
    # label the plots
    ax.set_title('Loss per Sample on Each Step')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Step')
    ax.legend()

    # set limits if the n_epochs is known
    if not n_epochs is None:
        ax.set_xlim(0,n_epochs)

    return fig, ax






















class BasicModelFitter:
    def __init__(self, 
                    model, 
                    device:str='auto', 
                    verbose:bool=False, 
                    model_path:typing.Union[None, str]=None, 
                    result_path:typing.Union[None, str]=None, 
                    model_name:str='', 
                    metrics_track:list=[],
                    writer:torch.utils.tensorboard.SummaryWriter=None,
                    ):
        '''
        This class can be used to find a model and perform inference.


        Arguments
        ---------
        - model: pytorch model
            This is the pytorch model that can be fit
            and have inference done using.

        - device: str (optional):
            This is the device name that the model will be trained on. 
            Most common arguments here will be :code:`'cpu'` 
            or :code:`'cuda'`. :code:`'auto'` will 
            pick  :code:`'cuda'` if available, otherwise
            the training will be performed on :code:`'cpu'`.
            Defaults to :code:`'auto'`.
        
        - verbose: bool (optional):
            Allows the user to specify whether progress
            should be printed as the model is training.
            Defaults to :code:`False`.

        - model_path: str` or :code:`None (optional):
            Path to the directory in which the models will be saved
            after training. If :code:`None`, no models are saved.
            If specifying a path, make sure that this path exists.
            Defaults to :code:`None`.
        
        - model_name: str:
            The name of the model. This is used when saving
            results and the model.
            Defaults to :code:`''`
        
        - metrics_track: list of str:
            List of strings containing the names of the 
            metrics to be tracked. Acceptable values are in
            :code:`['accuracy']`. 
            Loss is tracked by default.
            
            - :code:'accuracy'` reports the mean accuracy over 
                an epoch AFTER the model has been trained on the examples.
                :code:`'accuracy'` is accessible via the attributes 
                :code:`.train_accuracy` and :code:`.val_accuracy`.

            Defaults to :code:`[]`.
        
        - writer: torch.utils.tensorboard.SummaryWriter:
            This is the tensorboard writer that is used to track
            metrics as the model is training. If a writer is not
            passed as an argument then one is assigned with
            the current date and time, and :code:`model_name` as its title.

        '''


        self.model = model
        self.train_loss = {}
        self.val_loss = {}
        self.track_accuracy = True if 'accuracy' in metrics_track else False
        self.train_accuracy = {}
        self.val_accuracy = {}
        self.n_trains = -1
        self.device = ('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device
        self.verbose = verbose
        self.model_save = False if model_path is None else True
        self.result_save = False if result_path is None else True
        
        # make sure paths are of the correct format
        if self.model_save:
            if len(model_path) == 0:
                model_path = './'
            elif model_path[-1] != '/':
                model_path += '/'
        if self.result_save:
            if len(result_path) == 0:
                result_path = './'
            elif result_path[-1] != '/':
                result_path += '/'

        self.model_path = model_path
        self.model_name = model_name
        self.result_path = result_path

        # setting tensorboard writer
        if writer is None:
            self.writer = SummaryWriter(comment='-'+model_name)
        else:
            self.writer = writer

        return


    def _fit_traditional_batch(self, data):

        if hasattr(self.model, 'batch_start'):
            self.model.batch_start(self)
        
        self.optimizer.zero_grad()

        inputs, labels = data
        inputs, labels = inputs.to(self.device), labels.to(self.device)
        start_forward_backward_time = time.time()
        # ======= forward ======= 
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)
        # ======= backward =======
        start_optim_time = time.time()
        loss.backward()
        self.optimizer.step()

        # ======= measuring metrics =======
        end_optim_time = time.time()
        end_forward_backward_time = time.time()

        self.writer.add_scalar('Training Loss', loss, self.step)
        self.writer.add_scalar('Optimiser Time', 
                                end_optim_time-start_optim_time, self.step)
        self.writer.add_scalar('Forward and Backward Time', 
                                end_forward_backward_time-start_forward_backward_time, self.step)

        # assuming the loss function produces mean loss over all instances
        self.epoch_loss += loss.item()*len(inputs)
        self.instances += len(inputs)

        if self.track_accuracy:
            prediction = outputs.argmax(dim=1)
            correct = torch.sum(prediction == labels).item()
            self.epoch_training_correct += correct
            self.writer.add_scalar('Training Accuracy', correct/len(inputs), self.step)
        self.step += 1
        
        if hasattr(self.model, 'batch_end'):
            self.model.batch_end(self)

        return


    def _fit_source_traditional_batch(self, data,):
        '''
        This method assumes that each batch is made of a single
        source.
        '''

        if hasattr(self.model, 'batch_start'):
            self.model.batch_start(self)
        
        self.optimizer.zero_grad()

        inputs, labels, source = data

        if len(torch.unique(source)) > 1:
            raise NotImplementedError('Please make sure each data batch contains a single source.')

        inputs, labels = inputs.to(self.device), labels.to(self.device)
        source = source[0].item()

        if not source in self.source_step_dict:
            self.source_step_dict[source] = 0
        source_step = self.source_step_dict[source]

        start_forward_backward_time = time.time()
        # ======= forward ======= 
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)
        # ======= backward =======
        start_optim_time = time.time()
        loss.backward()
        self.optimizer.step(loss=loss, source=source, writer=self.writer)

        # ======= measuring metrics =======
        end_optim_time = time.time()
        end_forward_backward_time = time.time()

        self.writer.add_scalar('Training Loss', loss, self.step)
        self.writer.add_scalars('Training Source Loss', {'{}'.format(source): loss}, source_step)
        self.writer.add_scalar('Optimiser Time', end_optim_time-start_optim_time, self.step)
        self.writer.add_scalar('Forward and Backward Time', end_forward_backward_time-start_forward_backward_time, self.step)
        
        # assuming the loss function produces mean loss over all instances
        self.epoch_loss += loss.item()*len(inputs)
        self.instances += len(inputs)

        if self.track_accuracy:
            prediction = outputs.argmax(dim=1)
            correct = torch.sum(prediction == labels).item()
            self.epoch_training_correct += correct
            self.writer.add_scalar('Training Accuracy', correct/len(inputs), self.step)
            self.writer.add_scalars('Training Source Accuracy', 
                                    {'{}'.format(source): correct/len(inputs)}, source_step)

        if hasattr(self.model, 'batch_end'):
            self.model.batch_end(self)

        self.step += 1
        self.source_step_dict[source] += 1

        return

    def _validation(self, val_loader):
        val_loss = 0
        with torch.no_grad():
            epoch_validation_correct = 0
            instances = 0
            if hasattr(self.model, 'val_start'):
                self.model.val_start(self)

            for nb, data in enumerate(val_loader):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                # perform inference
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item()*len(inputs)
                instances += len(inputs)
                
                if self.track_accuracy:
                    prediction = self.model(inputs).argmax(dim=1)
                    correct = torch.sum(prediction == labels).item()
                    epoch_validation_correct += correct

        if hasattr(self.model, 'val_end'):
            self.model.val_end(self)

        epoch_val_loss = val_loss/instances
        self.epoch_bar['Val Loss']= '{:.2e}'.format(epoch_val_loss)
        self.writer.add_scalar('Validation Loss', epoch_val_loss, self.step)
        self.val_loss_temp.append(epoch_val_loss)

        if self.track_accuracy: 
            self.val_accuracy_temp.append(epoch_validation_correct/instances) 
            self.writer.add_scalar('Validation Accuracy', epoch_validation_correct/instances, self.step)
            self.epoch_bar['Val Acc'] = '{:.1f}%'.format(epoch_validation_correct/instances*100)

        return 

    def _source_validation(self, val_loader):
        val_loss = 0
        with torch.no_grad():
            epoch_validation_correct = 0
            instances = 0
            if hasattr(self.model, 'val_start'):
                self.model.val_start(self)

            for nb, data in enumerate(val_loader):
                inputs, labels, source = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                # perform inference
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item()*len(inputs)
                instances += len(inputs)
                
                if self.track_accuracy:
                    prediction = self.model(inputs).argmax(dim=1)
                    correct = torch.sum(prediction == labels).item()
                    epoch_validation_correct += correct

        if hasattr(self.model, 'val_end'):
            self.model.val_end(self)

        epoch_val_loss = val_loss/instances
        self.epoch_bar['Val Loss']= '{:.2e}'.format(epoch_val_loss)
        self.writer.add_scalar('Validation Loss', epoch_val_loss, self.step)
        self.val_loss_temp.append(epoch_val_loss)

        if self.track_accuracy: 
            self.val_accuracy_temp.append(epoch_validation_correct/instances) 
            self.writer.add_scalar('Validation Accuracy', epoch_validation_correct/instances, self.step)
            self.epoch_bar['Val Acc'] = '{:.1f}%'.format(epoch_validation_correct/instances*100)

        return 


    def _fit_epoch(self, train_loader, val_loader=None):
        
        if hasattr(self.model, 'epoch_start'):
                self.model.epoch_start(self)
        epoch_start_time = time.time()
        self.epoch_loss = 0
        self.epoch_training_correct = 0
        self.instances = 0

        # train over batches
        for nb, data in enumerate(train_loader):
            if self.source_fit:
                self._fit_source_traditional_batch(data)
            else:
                self._fit_traditional_batch(data)

        if not self.train_scheduler is None:
            self.train_scheduler.step()

        self.epoch_loss = self.epoch_loss/self.instances
        self.train_loss_temp.append(self.epoch_loss)
        
        self.epoch_bar['Loss'] = '{:.2e}'.format(self.epoch_loss)
        if self.track_accuracy: 
            self.train_accuracy_temp.append(self.epoch_training_correct/self.instances) 
            self.epoch_bar['Acc'] = '{:.1f}%'.format(self.epoch_training_correct/self.instances*100)
        
        # ======= validation =======
        self.model.eval()
        if self.val_too:
            if self.source_fit:
                self._source_validation(val_loader=val_loader)
            else:
                self._validation(val_loader=val_loader)
        self.model.train()

        epoch_end_time = time.time()
        self.epoch_bar['Took'] = '{:.1e}s'.format(epoch_end_time-epoch_start_time)

        if hasattr(self.model, 'epoch_end'):
            self.model.epoch_end(self)
        
        # if results saving is true, save the graph.
        if self.result_save:
            if self.val_too:
                fig, ax = loss_plot(self.train_loss_temp, self.val_loss_temp, n_epochs=self.n_epochs)
            else:
                fig, ax = loss_plot(self.train_loss_temp, n_epochs=self.n_epochs)
            fig.savefig(self.result_path + 'loss_plot-{}.pdf'.format(self.model_name), bbox_inches='tight')
            plt.close()

        return


    def fit(self, 
            train_loader, 
            n_epochs, 
            criterion, 
            optimizer, 
            val_loader=None,
            train_scheduler=None,
            source_fit=False,
            ):
        '''
        This fits the model.

        Arguments
        ---------
            
        - train_loader: torch.utils.data.DataLoader:
            Data loader for the training data. Each iteration 
            should contain the inputs and the targets.

        - n_epochs: int:
            This is the number of epochs to run the training for.
        
        - criterion: pytorch loss function:
            This is the loss function that will be used in the training.
        
        - optimizer: pytorch optimiser:
            This is the optimisation method used in the training.
        
        - val_loader: torch.utils.data.DataLoader (optional):
            Data loader for the validation data. Each iteration 
            should contain the inputs and the targets.
            If :code:`None` then no validation tests will be performed
            as the model is training.
            Defaults to :code:`None`.
        
        - train_scheduler: torch.optim.lr_scheduler (optional):
            Learning rate scheduler for training. 
            Defaults to :code:`None`.
        
        - source_fit: bool (optional):
            This argument tells the class whether sources are available in 
            the train and validation loaders and passes them to the optimizer
            during training.
            Defaults to :code:`False`.


        Returns
        ---------

        - model: pytorch model
            This returns the pytorch model after being 
            fitted using the arguments given.


        '''
        # attributes required for training
        self.n_trains += 1
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_scheduler = train_scheduler
        self.n_epochs = n_epochs
        self.val_too = not val_loader is None # true if val_loader is not None
        self.train_loss_temp = []
        self.val_loss_temp = []
        self.train_accuracy_temp = []
        self.val_accuracy_temp = []
        self.step = 0
        self.source_fit = source_fit
        if self.source_fit:
            self.source_step_dict = {}
        
        # setting model for training and putting on device
        self.model.train()
        self.model.to(self.device)
        if hasattr(self.model, 'traditional_train_start'):
            self.model.traditional_train_start(self)

        # printing if verbose is true
        self.epoch_bar = PBar(show_length = 10, n_iterations=self.n_epochs)
        self.epoch_bar['Epoch'] = '-'
        self.epoch_bar['Took'] = '-s'
        self.epoch_bar['Loss'] = '-'
        if self.val_too: 
            self.epoch_bar['Val Loss'] = '-'
        printing_statement = 'Training:    {}'.format(self.epoch_bar.give())
        if self.verbose:
            sys.stdout.write('\r')
            sys.stdout.write(printing_statement)
            sys.stdout.flush()
        
        train_start_time = time.time()

        # ======= training =======
        for epoch in range(self.n_epochs):
            self._fit_epoch(train_loader=train_loader, val_loader=val_loader)
            # updating epoch information
            self.epoch_bar.update(1)
            self.epoch_bar['Epoch'] = '{}'.format(epoch+1)
            printing_statement = 'Training:    {}'.format(self.epoch_bar.give())
            # printing information if verbose is true
            if self.verbose:
                sys.stdout.write('\r')
                sys.stdout.write(printing_statement)
                sys.stdout.flush()

        if hasattr(self.model, 'traditional_train_end'):
            self.model.traditional_train_end(self)

        # saving tracked metrics in the attributes that can be accessed after training
        self.train_loss[self.n_trains] = np.asarray(self.train_loss_temp)
        self.val_loss[self.n_trains] = np.asarray(self.val_loss_temp)
        self.train_accuracy[self.n_trains] = np.asarray(self.train_accuracy_temp)
        self.val_accuracy[self.n_trains] = np.asarray(self.val_accuracy_temp)

        train_end_time = time.time()

        # saving the model to the model path
        if self.model_save:
            save_name = ('{}-epoch_{}'.format(self.model_name, epoch+1)
                            + '-all_trained' )
            torch.save(self.model.state_dict(), self.model_path + save_name + '-state_dict' + '.pth')

        # printing the final information if verbose is true
        self.epoch_bar['Train Took'] = '{:.1e}s'.format(train_end_time-train_start_time)
        printing_statement = 'Training:    {}'.format(self.epoch_bar.give())
        if self.verbose:
            sys.stdout.write('\r')
            sys.stdout.write(printing_statement)
            sys.stdout.flush()
            sys.stdout.write('\n')
        self.writer.close()

        return self.model


