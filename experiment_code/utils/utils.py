import json
import numpy as np
import collections.abc

# package wide styling for progress bars
tqdm_style = {
    #'ascii':" ▖▘▝▗▚▞▉", 
    'ascii':"▏▎▍▋▊▉", 
    #'colour':'black',
    'dynamic_ncols': True,
    }

def format_mean_std(values):
    '''
    Given a numpy array of values, this function returns
    a string containing the mean and std of those numbers 
    in the format: :code:`'$MEAN \pm STD$'`.
    
    Arguments
    ---------
    
    - values: np.array: 
        Numpy array of values.
    
    Returns
    --------
    
    - out: str: 
        A string containing the mean and std of those numbers 
        in the format :code:`'$MEAN \pm STD$'`.
    
    
    '''
    return str(np.mean(values))[:6] + " +/- " + str(np.std(values))[:6]


def update_dict(d:dict, u:dict):
    '''
    https://stackoverflow.com/a/3233356

    This function updates the dictionary :code:`d`,
    with the information in dictionary :code:`u`.
    
    
    Arguments
    ---------
    
    - d: dict: 
        The dictionary to update.
    
    - u: dict: 
        The dictionary to update :code:`d` with.

    Returns
    --------
    
    - out: dict: 
        Updated dictionary
    
    '''
    
    # recursively search keys until values are found
    # in u, then update d
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update_dict(d.get(k, {}), v) # recursive call
        else:
            d[k] = v
    return d




class ArgFake:
    def __init__(self, arguments:dict):
        '''
        Fake argparse arguments
        
        Arguments
        ---------
        
        - arguments: dict: 
            This is a dictionary containing the arguments
            to be available as attributes.
        
        '''
        self.arguments = arguments
        for key in self.arguments:
            setattr(self, key, self.arguments[key])

    def __str__(self):
        return str(self.arguments)


# used in json dump to encode a numpy array
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj,np.int64):
            return int(obj)
        return json.JSONEncoder.default(self, obj)




def exponential_average(data, alpha_value, *args,  axis=None, **kwargs):
    '''
    Calculates the exponential average. This is the average
    of the points :code:`(x0, (a**1)*x1, (a**2)*x2, ..., (a**n)*xn)`.
    
    It has the same axis rules as :code:`numpy.mean`.
    
    
    Arguments
    ---------
    
    - data: array: 
        The data to calculate the exponential average of. This should
        be an array. With the last index of :code:`axis` being the 
        most recent value.
    
    - alpha_value: float: 
        This should be a float :code:`0 <= alpha_value <= 1`. 
        A value of :code:`1` returns the standard mean, since there 
        is no decay in the average. A value of 0 returns
        the first value in :code:`data`, since the decay is maximal.
    
    - axis: int:
        The axis to calculate the exponential average along.
        :code:`0` are columns and :code:`1` are rows. :code:`None` 
        will calculate the exponential average of all values 
        in the array.
        Defaults to :code:`None`.
    
    
    
    Returns
    --------
    
    - out: float` or :code:`array: 
        The exponential average of the data, given :code:`axis`.
    
    '''
    if alpha_value == 1.0:
        return np.mean(data, axis=axis)
    else:
        if len(data.shape)<2:
            axis=0
        else:
            if axis is None:
                data = data.copy().reshape(-1)
                axis=0
        scale = (1-alpha_value)/(1-alpha_value**(data.shape[axis]))
        alpha_vec = alpha_value**(np.arange(data.shape[axis])[::-1])
        ouput = scale*np.sum(data*alpha_vec, axis=axis)
        return ouput



def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    import math
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values-average)**2, weights=weights)
    return (average, math.sqrt(variance))


    


class PBar(dict):

    '''
  
    This is a class for a simple progress bar.

    You may also assign values to this class like::
        
        >>> pbar = PBar(100,5, variable_length=11)
        >>> pbar['Epoch'] = 10


    When the progress bar is returned, it will return::

        [#>---] - Epoch: 10           


    When adding this information, make sure that the
    sum of the characters in the name of the variable, 
    and the value do not exceed :code:`variable_length`.

    '''

    def __init__(self, 
                    show_length:int, 
                    n_iterations:int, 
                    done_symbol:str='#', 
                    next_symbol:str='>', 
                    todo_symbol:str='-',
                    variable_length:int=20):

        '''
        Arguments
        ---------

        - show_length: int:
            This is the length of the progress bar to show.

        - n_iterations: int:
            This is the number of iterations the progress bar should expect.

        - done_symbol: str:
            This is the symbol shown for the finished section of the progress bar.
            Defaults to :code:`'#'`.
            
        - next_symbol: str:
            This is the symbol that will show after the :code:`done_symbol` as the 
            progress bar is printing.
            Defaults to :code:`'>'`.

        - todo_symbol: str:
            This is the symbol shown for the unfinished section of the progress bar.
            Defaults to :code:`'-'`.
        
        - variable_length: int:
            This is the length of each of the printed variables. Spacing will be placed
            either side to ensure each pronted variable has this many characters.
            Defaults to :code:`20`.

        '''

        self.show_length = show_length
        self.n_iterations = n_iterations
        self.done_symbol = done_symbol
        self.todo_symbol = todo_symbol
        self.next_symbol = next_symbol
        self.variable_length = variable_length
        self.things_to_show = {}
        self.progress = 0

        return

    def update(self, n = 1):
        '''
        This is the update function for the progress bar

        Arguments
        ---------

        - n: int:
            The number of steps to update the progress bar by.
            Defaults to :code:`1`

        '''

        self.progress += n

        return


    def give(self):

        '''
        Returns
        ---------

        - out: str:
            This returns a string containing the progress bar 
            and information.

        '''

        total_bar_length = self.show_length
        current_progress = self.progress if self.progress <= self.n_iterations else self.n_iterations
        n_iterations = self.n_iterations
        hashes_length = int((current_progress)/n_iterations*total_bar_length)
        hashes = self.done_symbol*hashes_length
        dashes = self.todo_symbol*(total_bar_length-hashes_length)
        if len(dashes) >= 1:
            dashes = self.next_symbol + dashes[1:]
        
        extra_info = ''
        if len(self.keys()) > 0:
            for key in self.keys():
                extra_info_to_add = '{}: {}'.format(key, self[key])
                spacing = self.variable_length - len(extra_info_to_add)
                spacing_after = ' '*(spacing//2)
                spacing_before = ' '*(spacing//2 + spacing%2)
                extra_info += '{}{}: {}{}'.format(spacing_before, key, self[key], spacing_after)
        
        extra_info += 10*' '

        out = '[{}{}] {}'.format(hashes, dashes, extra_info)

        return out

    def show(self):
        '''
        This prints the progress bar.
        
        '''

        p_bar_to_print = self.give()

        print(p_bar_to_print)

        return