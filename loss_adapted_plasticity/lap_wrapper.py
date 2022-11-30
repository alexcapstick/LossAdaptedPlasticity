import math
import numpy as np
import logging
import typing
import torch
from torch.utils.tensorboard import SummaryWriter


def weighted_avg_and_std(values, weights) -> typing.Tuple[np.array, np.array]:
    '''
    Return the weighted average and standard deviation.

    Arguments
    ---------

    - values: np.array:
        The array containing the values to
        calculate the mean and std on.
    
    - weights: np.array:
        The weights used in the mean and std.


    Returns
    ---------

    - out: typing.Tuple[np.array, np.array]:
        The weighted mean and std.

    '''
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values-average)**2, weights=weights)
    return (average, math.sqrt(variance))









class DiscreteRankingSTD(object):
    def __init__(self,
                    function=lambda x: x,
                    discrete_amount:float=0.005,
                    hold_off:int=0, 
                    strictness:float=1.0,
                    ):
        '''
        This class calculates the level of depression 
        to apply to gradient updates from a batch of 
        source data.
        
        
        
        Arguments
        ---------
        
        - function: _type_, optional:
            This argument allows you to apply a function
            to the value before it is returned. 
            Defaults to :code:`lambda x: x`.
        
        - discrete_amount: float, optional:
            The step size used when calculating the depression. 
            Defaults to :code:`0.005`.
        
        - hold_off: int, optional:
            The number of calls to this function before
            depression will start. Until depression starts,
            this function will return 0 on each call. 
            Defaults to :code:`0`.
        
        - strictness: float, optional:
            The number of standard deviations away from the 
            mean loss a mean source loss has to be 
            before depression is applied. 
            Defaults to :code:`1.0`.
        
        
        '''
        self.function = function
        self.discrete_amount = discrete_amount
        self.source_xn = np.asarray([])
        self.hold_off = hold_off
        self.strictness = strictness
        self.step = 0
        return

    def __call__(self, 
                    loss_array:np.ndarray, 
                    source_idx:int, 
                    *args, 
                    **kwargs)->float:
        '''
        
        Arguments
        ---------
        
        - loss_array: np.ndarray: 
            The loss values for the last n batches of each source.
            Where n is the history size.
            This should be of shape (n_sources, n_batches_prev_tracked).
        
        - source_idx: int: 
            The index in the loss array of the source 
            being updated.
        

        Returns
        --------
        
        - out: float:
            The depression value, d in the depression calculation:
            dep = 1-tanh(m*d)**2.
            This means, the larger the value, the more depression 
            will be applied during training.
        
        '''
        # increasing step and checking if the hold off time has passed.
        self.step += 1
        if self.step < self.hold_off:
            return 0

        logging.debug('Source Index {}'.format(source_idx))

        # keeps track of the current depression applied to each source
        # these will be used as weights in the standard deviation and 
        # mean calculations
        if len(loss_array) > len(self.source_xn):
            self.source_xn = np.hstack([self.source_xn, np.zeros(len(loss_array) - len(self.source_xn))])

        # mask is True where loss array source is not equal to the current source
        mask = np.ones(loss_array.shape[0], dtype=bool)
        mask[source_idx] = False

        # if the range in loss values is close to 0, return no depression
        if np.all(np.isclose(np.ptp(loss_array[mask]), 0)):
            return 0

        # mean loss of current source
        mean_source_loss = np.mean(loss_array[~mask])

        # weighted mean and standard deviation of the sources other
        # than the current source.
        weights = np.ones_like(loss_array)/((self.source_xn + 1)[:,np.newaxis])
        (mean_not_source_loss, 
        std_not_source_loss) = weighted_avg_and_std(loss_array[mask], 
                                                    weights=weights[mask])

        # calculates whether to trust a source more or less
        logging.debug('{} < {}'.format(mean_source_loss, mean_not_source_loss + self.strictness*std_not_source_loss))
        if mean_source_loss < mean_not_source_loss + self.strictness*std_not_source_loss:
            movement = -1
        else:
            movement = 1
        logging.debug('movement {}'.format(movement))
        logging.debug('source_xn {}'.format(self.source_xn[source_idx]))
        # moving the current trust level depending on the movement calculated above
        self.source_xn[source_idx] += movement
        if self.source_xn[source_idx] < 0:
            self.source_xn[source_idx] = 0
        
        # calculating the depression value
        depression = self.function(self.discrete_amount*self.source_xn[source_idx])

        return depression










class LAP(object):
    def __init__(
        self,
        optimizer:torch.optim.Optimizer,
        lap_n:int=10,
        depression_strength:float=1.0,
        depression_function='discrete_ranking_std', 
        depression_function_kwargs:dict={},
        source_is_bool:bool=False,
        **opt_kwargs,
        ):
        '''
        Depression won't be applied until at least :code:`lap_n` loss values
        have been collected for at least two sources. This could be 
        longer if a :code:`hold_off` parameter is used in the depression function.

        This class will wrap any optimiser and perform lap gradient depression
        before the values are passed to the underlying optimiser.


        Examples
        ---------
        
        The following wraps the Adam optimiser with the LAP functionality.
        
        .. code-block::
            
            >>> optimizer = LAP(
            ...     torch.optim.Adam, params=model.parameters(), lr=0.01,
            ...     )
        
        Ensure that when using this optimiser, during the :code:`.step`
        method, you use the arguments :code:`loss` and :code:`source`. 
        For example::

            >>> loss = loss.backward()
            >>> optimizer.step(loss, source)


        Arguments
        ---------

        - optimizer: torch.optim.Optimizer:
            The optimizer to wrap with the LAP algorithm.
        
        - lap_n: int, optional:
            The number of previous loss values for each source
            to be used in the loss adapted plasticity
            calculations.
            Defaults to :code:`10`.
        
        - depression_strength: float:
            This float determines the strength of the depression
            applied to the gradients. It is the value of m in 
            dep = 1-tanh(m*d)**2.
            Defaults to :code:`1`.
        
        - depression_function: function or string, optional:
            This is the function used to calculate the depression
            based on the loss array (with sources containing full 
            loss history) and the source of the current batch. 
            Ensure that the first two arguments of this function are
            loss_array and source_idx.
            If string, please ensure it is 'discrete_ranking_std'
            Defaults to :code:`'discrete_ranking_std'`.
        
        - depression_function_kwargs: dict, optional:
            Keyword arguments that will be used in depression_function
            when initiating it, if it is specified by a string.
            Defaults to :code:`{}`.
        
        - source_is_bool: bool, optional:
            This tells the optimizer that the sources will be named True
            when the source is corrupted and False if the source is not.
            If the incoming source is corrupted, then the optimizer will not
            make a step.
            Defaults to :code:`False`.
        
        '''

        if (not 0 <= lap_n) and (type(lap_n) == int):
            raise ValueError("Invalid parameter for lap_n: {}. "\
                                "Please use an integer larger than 0".format(lap_n))
        if not 0.0 <= depression_strength:
            raise ValueError("Invalid depression stregnth: {}".format(depression_strength))

        self.optimizer = optimizer(**opt_kwargs)

        # storing settings and creating the loss array
        self.lap_n = lap_n
        self.loss_array = -1*np.ones((1,self.lap_n))
        self.source_dict = {}
        self.n_sources = 0
        self.depression_strength = depression_strength
        self.depression_function_kwargs = depression_function_kwargs
        self.depression_function = (
            depression_function 
            if not type(depression_function) == str 
            else self._get_depression_function(depression_function)
            )
        self.source_step_dict = {}
        self.source_is_bool = source_is_bool

        return

    def _has_complete_history(self):
        # returns source indices in which there is a complete history of losses
        return np.argwhere(np.sum(self.loss_array != -1, axis=1) == self.lap_n).reshape(-1)

    def _get_depression_function(self, name):
        '''
        Function to get the drepression function by name.
        '''
        if name == 'discrete_ranking_std':
            return DiscreteRankingSTD(**self.depression_function_kwargs)

        else:
            raise NotImplementedError('{} is not a known depression function. Please '\
                                        'pass the function instead of the name.'.format(name))

    @torch.no_grad()
    def step(
        self, 
        loss:float, 
        source:typing.Hashable, 
        override_dep:typing.Union[bool,None]=None, 
        writer:typing.Union[SummaryWriter, None]=None, 
        **kwargs,
        ):
        '''
        Performs a single optimization step.

        Arguments
        ---------

        - loss: float:
            This is the loss value that is used in the depression calculations.
        
        - source: hashable:
            This is the source name that is used to
            store the loss values for the different sources.
        
        - override_dep: bool or None:
            If None, then whether to apply depression will be decided
            based on the logic of this class. If True, then depression will 
            be applied. This might cause unexpected results if there is no depression value
            calculated based on whether there is enough data available in the 
            .loss_array. In this case, not depression is applied.
            If False, then depression will not be applied.
            This is mostly useful as an option to turn off LAP.
            Defaults to :code:`None`.
        
        - writer: torch.utils.tensorboard.SummaryWriter:
            A tensorboard writer can be passed into this function to track metrics.
            Defaults to :code:`None`.

        '''

        logging.debug('source, {}'.format(source))
        logging.debug('loss, {}'.format(loss))

        # if reliability of source is given, update only when
        # data is reliable
        if self.source_is_bool:
            if source:
                return None
            else:
                if not override_dep in [True, False]:
                    override_dep = False

        # building the loss array
        if not source in self.source_dict:
            # if new source, add row to the loss array
            self.source_dict[source] = self.n_sources
            self.n_sources += 1
            source_idx = self.source_dict[source]
            self.loss_array = np.concatenate([self.loss_array, -1*np.ones((1, self.lap_n))], axis=0)
            self.loss_array[source_idx, -1] = loss
        else:
            # if already tracked source, move history along and add new loss value
            source_idx = self.source_dict[source]
            losses = self.loss_array[source_idx]
            losses[:-1] = losses[1:]
            losses[-1] = loss
            logging.debug('losses, {}'.format(losses))
            logging.debug('loss array, {}'.format(self.loss_array))
        
        # saves the number of times each source has been seen for summary writer
        if not source in self.source_step_dict:
            self.source_step_dict[source] = 0
        self.source_step_dict[source] += 1

        # finds sources that have a complete history of losses
        history_idx = self._has_complete_history()

        # if current source has full history and at least one other source does
        # then perform depression calculations
        if (len(history_idx)>1) and (source_idx in history_idx):
            depressing = True
        else:
            depressing = False

        # calculate the depression value
        if depressing:
            depression = self.depression_function(
                loss_array=self.loss_array[history_idx], 
                source_idx=np.argwhere(history_idx == source_idx).reshape(-1)[0]
                )
        logging.debug('depressing, {}'.format(depressing))
        
        # depression boolean override from argument
        # if override is True and there is no depression value calculated
        # the then depression value is set to 0 (no depression)
        if not override_dep is None:
            if override_dep in [True, False]:
                if not depressing:
                    depression = 0.0
                depressing = override_dep
            else:
                raise TypeError('override_dep must be of boolean value, or None. Please see docs.')

        for group in self.optimizer.param_groups:
            params_with_grad = []

            # calculate the actual depression to be multiplied by the gradients
            if depressing:
                logging.debug('Depression, {}'.format(depression))
                actual_depression = 1-torch.pow(
                                            torch.tanh(
                                                torch.tensor(depression*self.depression_strength)),
                                            2).item()
            else:
                actual_depression = 1
            
            # saves the depression value to the writer
            if not writer is None:
                writer.add_scalars('Actual Depression Value', 
                                    {'{}'.format(source): actual_depression}, 
                                    self.source_step_dict[source])

            logging.debug('Actual Depression, {}'.format(actual_depression))

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    if p.grad.is_sparse:
                        raise RuntimeError('This has not been designed for sparse '\
                            'gradients and may not return expected results')
                    
                    # ======= applying depression ======= 
                    p.grad.mul_(actual_depression)
                    # =================================== 
            
            self.optimizer.step(**kwargs)

    # defined since __getattr__ causes pickling problems
    def __getstate__(self):
        return vars(self)

    # defined since __getattr__ causes pickling problems
    def __setstate__(self, state):
        vars(self).update(state)

    def __getattr__(self, name):
        if hasattr(self.optimizer, name):
            return getattr(self.optimizer, name)
        else:
            raise AttributeError

