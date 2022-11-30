import joblib
import typing
from .parallel_progress import ProgressParallel


def _p_apply_construct_inputs(
    **kwargs,
    ) -> typing.Tuple[typing.List[typing.Dict[str,typing.Any]], typing.Dict[str,typing.Any]]:
    '''
    This function allows you to pass keyword
    arguments that are prefixed with :code:`list__`
    and not, and return a list of dictionaries that
    can be used as keyword arguments in :code:`p_apply`.

    The example explains this better.
    
    
    
    Examples
    ---------
    
    .. code-block::
    
        >>> list_kwargs, reused_kwargs = _p_apply_construct_inputs(
        ...     x=[1,2,3,4,5,6],
        ...     list__y=[[1,2,3], [1,2], [1,]],
        ...     )
        >>> list_kwargs
        [{'y': [1, 2, 3]}, {'y': [1, 2]}, {'y': [1]}]
        >>> reused_kwargs
        {'x': [1, 2, 3, 4, 5, 6]}
    
    
    Raises
    ---------
    
        TypeError: If the :code:`list__` arguments are
        of different lengths.
    
    Returns
    --------
    
    - out: typing.Tuple[typing.List[typing.Dict[str,typing.Any]], typing.Dict[str,typing.Any]]: 
        A tuple with a list of dictionaries and a dictionary.
    
    
    '''
    
    list_kwargs = {}
    reused_kwargs = {}
    
    list_len = None
    for key, value in kwargs.items():
        if 'list__' in key:
            list_kwargs[key.replace('list__', '')] = value
            if list_len is None:
                list_len = len(value)
            if len(value) != list_len:
                raise TypeError("Ensure all of the list__ prefixed "\
                    "arguments have the same length."
                    )
        else:
            reused_kwargs[key] = value

    list_kwargs = [
        dict(zip(list_kwargs.keys(), values)) 
        for values in zip(*list_kwargs.values())
        ]

    if len(list_kwargs) == 0:
        list_kwargs = [{}]

    return list_kwargs, reused_kwargs




def p_apply(
    func:typing.Callable, 
    n_jobs:int=1, 
    backend:str='threading', 
    verbose:bool=True, 
    **kwargs,
    ) -> typing.List[typing.Any]:
    '''
    This class allows you to parallelise any function
    over some inputs.
    
    You may use the prefix :code:`list__` to any
    argument for each element to be parallelised,
    and not prefix an argument for it to be 
    consistent between parallel computations.

    This is more easily seen through example.
    
    
    Examples
    ---------
    
    In the following example, the function is 
    parallelised over the :code:`y` argument, since
    this is prefixed with :code:`list__`. This
    means that :code:`x` is added to each of the
    :code:`y`s.

    .. code-block::
    
        >>> p_apply(
        ...     lambda x,y: x+y,
        ...     x=np.array([0,1,2]),
        ...     list__y=np.array([0,1,2]),
        ...     )
        Parallel function: 3it [00:00, 2000.78it/s]
        [array([0, 1, 2]), array([1, 2, 3]), array([2, 3, 4])]

    In the next example, the function is 
    parallelised over none of the arguments, since
    none are prefixed with :code:`list__`.
    This means that :code:`x` is added to :code:`y`.

    .. code-block::

        >>> p_apply(
        ...     lambda x,y: x+y,
        ...     x=np.array([0,1,2]),
        ...     y=np.array([0,1,2]),
        ...     )
        Parallel function: 1it [00:00, ?it/s]
        [array([0, 2, 4])]
    
    
    Arguments
    ---------
    
    - func: typing.Callable: 
        The function to be used.
    
    - n_jobs: int, optional:
        The number of jobs to run in parallel. Be mindful that
        the functions and related computations might be expensive
        for the CPU, GPU, or RAM.
        Defaults to :code:`1`.
    
    - backend: str, optional:
        The backend used for the parallel compute. This should
        be an acceptable value for :code:`joblib.Parallel`.
        Defaults to :code:`threading`.
    
    - verbose: bool, optional:
        Whether to print progress. 
        Defaults to :code:`True`.
    
    Returns
    --------
    
    - out: typing.List[typing.Any]: 
        A list of the outputs. The list
        iterates over the parallel computes.
    
    
    '''
    
    list_kwargs, reused_kwargs = _p_apply_construct_inputs(**kwargs)

    return ProgressParallel(
        n_jobs=n_jobs,
        backend=backend,
        verbose=verbose,
        desc=f'Parallel {func.__name__}',
        )(
            joblib.delayed(func)(
                **lk,
                **reused_kwargs,
                ) for lk in list_kwargs
            )