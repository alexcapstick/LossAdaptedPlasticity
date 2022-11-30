import typing
from joblib import Parallel
import tqdm
from .utils import tqdm_style


class ProgressParallel(Parallel):
    def __init__(
        self, 
        tqdm_bar:typing.Union[tqdm.tqdm, None]=None, 
        verbose:bool=True,
        desc:str='In Parallel',
        total:int=None,
        tqdm_style:typing.Dict[str,typing.Any]=tqdm_style,
        *args, 
        **kwargs,
        ):
        '''
        This is a wrapper for the joblib Parallel
        class that allows for a progress bar to be passed into
        the :code:`__init__` function so that the progress 
        can be viewed.

        Recall that using :code:`backend='threading'`
        allows for shared access to variables!
        
        
        
        Examples
        ---------
        
        .. code-block:: 
        
            >>> pbar = tqdm.tqdm(total=5)
            >>> result = ProgressParallel(
            ...     tqdm_bar=pbar,
            ...     n_jobs=10,
            ...     )(
            ...         joblib.delayed(f_parallel)(i)
            ...         for i in range(5)
            ...     )
        
        Alternatively, you do not need to pass a :code:`tqdm` bar:

        .. code-block:: 
        
            >>> result = ProgressParallel(
            ...     n_jobs=10,
            ...     total=20,
            ...     desc='In Parallel',
            ...     )(
            ...         joblib.delayed(f_parallel)(i)
            ...         for i in range(20)
            ...     )
        
        
        Arguments
        ---------
        
        - tqdm_bar: typing.Union[tqdm.tqdm, None]: 
            The tqdm bar that will be used in the
            progress updates.
            Every time progress is displayed, 
            :code:`tqdm_bar.update(n)` will be called,
            where :code:`n` is the number of updates made.
            If :code:`None`, then a bar is created 
            inside this class.
            Defaults to :code:`None`.
        
        - verbose: bool: 
            If :code:`tqdm_bar=None`, then this
            argument allows the user to stop the 
            progress bar from printing at all.
            Defaults to :code:`True`.
        
        - desc: str: 
            If :code:`tqdm_bar=None`, then this
            argument allows the user to add 
            a description to the progress bar.
            Defaults to :code:`'In Parallel'`.
        
        - total: str: 
            If :code:`tqdm_bar=None`, then this
            argument allows the user to add 
            a total to the progress bar, rather
            than let the bar automatically update it
            as it finds new tasks. If :code:`None`, then
            the total might update multiple times as the 
            parallel process queues jobs.
            Defaults to :code:`None`.
        
        - tqdm_style: typing.Dict[str,typing.Any]: 
            A dictionary passed to the tqdm object
            which can be used to pass kwargs.
            :code:`desc`, :code:`total`, and  :code:`disable`
            (verbose) cannot be passed here. Please 
            use the arguments above.
            Defaults to :code:`tqdm_style`.

        
        '''

        super().__init__(verbose=False, *args, **kwargs)

        if tqdm_bar is None:
            self.tqdm_bar = tqdm.tqdm(
                desc=desc, 
                total=total, 
                disable=not verbose, 
                smoothing=0,
                **tqdm_style,
                )
            self.total=total
            self.bar_this_instance=True
        else:
            self.tqdm_bar = tqdm_bar
            self.bar_this_instance=False
        self.previously_completed = 0
        self._verbose = verbose
    
    def __call__(self, *args, **kwargs):
        return Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        if not self._verbose:
            return
        if self.bar_this_instance:
            # Original job iterator becomes None once it has been fully
            # consumed : at this point we know the total number of jobs and we are
            # able to display an estimation of the remaining time based on already
            # completed jobs. Otherwise, we simply display the number of completed
            # tasks.
            if self.total is None:
                if self._original_iterator is None:
                    # We are finished dispatching
                    if self.n_jobs == 1:
                        self.tqdm_bar.total = None
                        self.total = None
                    else:
                        self.tqdm_bar.total = self.n_dispatched_tasks
                        self.total = self.n_dispatched_tasks

        difference = self.n_completed_tasks - self.previously_completed
        self.tqdm_bar.update(difference)
        self.tqdm_bar.refresh()
        self.previously_completed += difference
        
        if self.bar_this_instance:
            if self.previously_completed == self.total:
                self.tqdm_bar.close()

        return