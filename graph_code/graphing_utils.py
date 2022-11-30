import matplotlib.pyplot as plt
import seaborn as sns
import contextlib
import matplotlib
from cycler import cycler
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import typing
import joblib
import tqdm
import functools
import pandas as pd
from joblib.externals.loky import get_reusable_executor
import os
from joblib import Parallel


tqdm_style = {
                #'ascii':" ▖▘▝▗▚▞▉", 
                'ascii':"▏▎▍▋▊▉", 
                #'colour':'black',
                'dynamic_ncols': True,
                }


def clean_tb(df):
    df = (df
    .drop(['run', 'level_0', 'level_1'], axis=1)
    .rename({'level_2': 'Run', 'level_3': 'Source', 'value': 'Value', 'step': 'Step'}, axis=1)
    .assign(
        Source = lambda x: x['Source'].str.split('_').str[-1].astype(float).astype(int)
        )
    )
    return expand_run_names(df)



def expand_run_names(df, runs_col = 'Run'):
    
    # producing maps
    unique_runs = df[runs_col].unique()
    corruption_map = {run: run.split('-c_')[1].split('-')[0] for run in unique_runs if '-c_' in run}
    n_sources_map = {run: run.split('-ns_')[1].split('-')[0] for run in unique_runs if '-ns_' in run}
    n_corrupt_sources_map = {run: run.split('-ncs_')[1].split('-')[0] for run in unique_runs if '-ncs_' in run}
    source_size_map = {run: run.split('-ssize_')[1].split('-')[0] for run in unique_runs if '-ssize_' in run}
    depression_strength_map = {run: run.split('-ds_')[1].split('-')[0] for run in unique_runs if '-ds_' in run}
    seed_map = {run: run.split('-seed_')[1].split('-')[0] for run in unique_runs if '-seed_' in run}
    epoch_map = {run: run.split('-ne_')[1].split('-')[0] for run in unique_runs if '-ne_' in run}
    lap_n = {run: run.split('-lap_n_')[1].split('-')[0] for run in unique_runs if '-lap_n_' in run}
    stns_map = {run: run.split('-stns_')[1].split('-')[0] for run in unique_runs if '-stns_' in run}
    
    # mapping dataframe
    df['Corruption'] = df['Run'].map(corruption_map)
    df['Number of Sources'] = df['Run'].map(n_sources_map)
    df['Number of Corrupt Sources'] = df['Run'].map(n_corrupt_sources_map)
    df['Source Size'] = df['Run'].map(source_size_map)
    df['Depression Strength'] = df['Run'].map(depression_strength_map)
    df['Seed'] = df['Run'].map(seed_map)
    df['Number of Epochs'] = df['Run'].map(epoch_map)
    df['LAP N'] = df['Run'].map(lap_n)
    df['Strictness'] = df['Run'].map(stns_map)

    return df


def run_model_map(run, lap_n_dict):
    if '_srb' in run:
        model_name = 'Corruption Oracle'
        return model_name
    elif '-ds_0.0' in run:
        model_name = 'Standard Model'
        return model_name
    elif '-ds_1.0' in run:
        for lap_n, model_name in lap_n_dict.items():
            if lap_n in run:
                return model_name
        return pd.NA
    else:
        return pd.NA


def run_corrupt_map(run):

    corrupt_map = {
        'ARFL-no_c': 'Original\nData',
        'ARFL-c_cs': 'Chunk\nShuffle',
        'ARFL-c_rl': 'Random\nLabel',
        'ARFL-c_lbs': 'Batch\nLabel\nShuffle',
        'ARFL-c_lbf': 'Batch\nLabel\nFlip',
        'ARFL-c_ns': 'Added\nNoise',
        'ARFL-c_no': 'Replace\nWith\nNoise',
        }

    for model_name, corrupt_name in corrupt_map.items():
        if model_name in run:
            return corrupt_name
    return pd.NA




def boxplot(*args, **kwargs):
    '''
    This is a wrapper for the seaborn boxplot
    function that includes some default formatting.

    By default, all lines will have width :code:`2`, be black
    and the boxplot width is :code:`0.75`.
    '''
    boxprops = dict(
        linestyle='-',
        linewidth=2.0,
        edgecolor='black',
        )
    if 'boxprops' in kwargs:
        boxprops.update(kwargs['boxprops'])
    kwargs['boxprops'] = boxprops

    capprops = dict(
        linestyle='-',
        linewidth=2.0,
        color='black',
        )
    if 'capprops' in kwargs:
        capprops.update(kwargs['capprops'])
    kwargs['capprops'] = capprops

    medianprops = dict(
        linestyle='-',
        linewidth=2.0,
        color='black',
        )
    if 'medianprops' in kwargs:
        medianprops.update(kwargs['medianprops'])
    kwargs['medianprops'] = medianprops

    whiskerprops = dict(
        linestyle='-',
        linewidth=2.0,
        color='black',
        )
    if 'whiskerprops' in kwargs:
        whiskerprops.update(kwargs['whiskerprops'])
    kwargs['whiskerprops'] = whiskerprops

    if 'width' not in kwargs:
        kwargs['width'] = 0.75

    return sns.boxplot(
        *args, 
        **kwargs,
        )


# colours
tol_muted = [
    '#332288', 
    '#88CCEE', 
    '#44AA99', 
    '#117733', 
    '#999933', 
    '#DDCC77', 
    '#CC6677', 
    '#882255',
    '#AA4499'
    ]

ibm = [
    "#648fff",
    "#fe6100",
    "#dc267f", 
    "#785ef0",
    "#ffb000",
    ]


# colour map
def set_colour_map(colours:list=tol_muted):
    '''
    Sets the default colour map for all plots.
    
    
    
    Examples
    ---------
    
    The following sets the colourmap to :code:`tol_muted`:

    .. code-block::
    
        >>> set_colour_map(colours=avt.tol_muted)
    
    
    Arguments
    ---------
    
    - colours: list, optional:
        Format that is accepted by 
        :code:`cycler.cycler`. 
        Defaults to :code:`tol_muted`.
    
    '''
    custom_params = {"axes.prop_cycle": cycler(color=colours)}
    matplotlib.rcParams.update(**custom_params)

# context functions
@contextlib.contextmanager
def temp_colour_map(colours=tol_muted):
    '''
    Temporarily sets the default colour map for all plots.
    

    Examples
    ---------
    
    The following sets the colourmap to :code:`tol_muted` for
    the plotting done within the context:

    .. code-block::
    
        >>> with set_colour_map(colours=avt.tol_muted):
        ...     plt.plot(x,y)
    
    
    Arguments
    ---------
    
    - colours: list, optional:
        Format that is accepted by 
        :code:`cycler.cycler`. 
        Defaults to :code:`tol_muted`.
    
    '''
    set_colour_map(colours=colours)


@contextlib.contextmanager
def paper_theme(colours=ibm):
    with matplotlib.rc_context():
        plt.style.use('seaborn-poster')
        custom_params = {
            
            "axes.spines.right": False, 
            "axes.spines.top": False, 
            "axes.edgecolor" : 'black',
            'axes.linewidth': 2,
            'axes.grid': True,
            'axes.axisbelow': True,
            "axes.prop_cycle": cycler(color=colours),

            'grid.alpha': 0.5,
            'grid.color': '#b0b0b0',
            'grid.linestyle': '--',
            'grid.linewidth': 2,

            "font.family": "Times New Roman",
            
            'xtick.major.width': 2,
            'ytick.major.width': 2,
            
            'boxplot.whiskerprops.linestyle': '-',
            'boxplot.whiskerprops.linewidth': 2,
            'boxplot.whiskerprops.color': 'black',
            
            'boxplot.boxprops.linestyle': '-',
            'boxplot.boxprops.linewidth': 2,
            'boxplot.boxprops.color': 'black',

            'boxplot.meanprops.markeredgecolor': 'black',

            'boxplot.capprops.color': 'black',
            'boxplot.capprops.linestyle': '-',
            'boxplot.capprops.linewidth': 1.0,
            
            'legend.title_fontsize': 16,
            'legend.fontsize': 16,

            }
        

        matplotlib.rcParams.update(**custom_params)

        yield


def save_fig(fig:plt.figure, file_name:str, **kwargs) -> None:
    '''
    This function saves a pdf, png, and svg of the figure,
    with :code:`bbox_inches='tight'` and :code:`dpi=300`.
    

    Arguments
    ---------

    - fig: plt.figure:
        The figure to save.
    
    - file_name: str:
        The file name, including path, to save the figure at.
        This should not include the extension, which will 
        be added when each file is saved.

    '''
    fig.savefig(f'{file_name}.pdf', bbox_inches='tight', **kwargs)
    fig.savefig(f'{file_name}.png', dpi=300, bbox_inches='tight', **kwargs)
    fig.savefig(f'{file_name}.svg', bbox_inches='tight', **kwargs)










def dirtree(
    path:str, 
    level:typing.Union[None, int]=None, 
    files_only:bool=False,
    file_path:bool=False
    ) -> dict:
    '''
    This function will produce a dictionary of 
    the file structure. All keys with value of 
    :code:`None` are files, and if 
    :code:`files_only=True` all values that are 
    part of a list are files.
    
    
    
    Examples
    ---------
    
    .. code-block:: 
    
        >>> dirtree('./')
    
    
    Arguments
    ---------
    
    - path: str: 
        The path to search over.
    
    - level: typing.Union[None, int], optional:
        The number of levels to recursively search.
        :code:`level=0` is the files in the directory of the path,
        and :code:`level=1` would be all of the files in the directories
        of the directory of the path, etc. 
        :code:`None` searches recursively until there are no 
        more directories in the leaves.
        Note that :code:`level=-1` will not return the 
        tree from the last level, but instead act as if
        :code:`level=None`.
        Defaults to :code:`None`.
    
    - files_only: bool, optional:
        Whether to only show the files, or the folders too.
        :code:`True` will only return the files.
        Defaults to :code:`False`.
    
    - file_path: bool, optional:
        If :code:`True`, then the returned
        names will contain the full paths.
        Defaults to :code:`False`.
    
    Returns
    ---------
    
    - directory_dict: dict:
        The dictionary containing the file structure.
    
    '''
    
    def recursive_build(path, level):
        if level is None:
            level = -1
        if os.path.isdir(path):
            if level != 0:
                d = {}
                for name in os.listdir(path):
                    if file_path:
                        d[os.path.join(path, name)] = recursive_build(os.path.join(path, name), level=level-1)
                    else:
                        d[name] = recursive_build(os.path.join(path, name), level=level-1)
            else:
                if files_only:
                    if file_path:
                        d = [os.path.join(path, name) for name in os.listdir(path) if not os.path.isdir(os.path.join(path, name))]
                    else:
                        d = [name for name in os.listdir(path) if not os.path.isdir(os.path.join(path, name))]
                else:
                    if file_path:
                        d = [os.path.join(path, name) for name in os.listdir(path)]
                    else:
                        d = [name for name in os.listdir(path)]
        else:
            d=None
        return d
    out = {path: recursive_build(path, level)}
    return out








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
            Defaults to :code:`aml_tqdm_style` (see :code:`aml.tqdm_style`).

        
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











class TensorboardLoad:
    def __init__(
        self, 
        path:str, 
        level:typing.Union[int, None]=None,
        verbose:bool=True,
        n_jobs=1,
        ):
        '''
        This class allows you to load tensorboard files
        from a directory. 
        
        
        Arguments
        ---------
        
        - path: str: 
            The path of the directory containing the files.
        
        - level: typing.Union[int, None], optional:
            The maximum number of levels to dive into
            when loading the files. If :code:`None` then
            all levels are loaded. Note that :code:`level=-1`
            will behave as if :code:`level=None`, and will return
            all of the files, and not just the deepest level.
            Defaults to :code:`None`.
        
        - verbose: bool, optional:
            Whether to print progress when
            loading the files. 
            Defaults to :code:`True`.
        
        - n_jobs: int, optional:
            The number of parallel operations when loading 
            the data.
            Defaults to :code:`1`.
        
        '''

        self.level= level if not level is None else -1
        self.path = path
        self.verbose=verbose
        self.n_jobs = n_jobs

        self.file_directory = dirtree(path=path, level=level, files_only=True, file_path=True,)

        queue = [[self.file_directory, -1]]
        self.level_dict = {}
        while len(queue) > 0:
            next_item, next_item_level= queue.pop()
            if not next_item_level in self.level_dict:
                self.level_dict[next_item_level] = []

            if type(next_item) == dict:
                for key in next_item.keys():
                    if next_item[key] is None:
                        self.level_dict[next_item_level].append(key)
                    else:
                        queue.append([next_item[key], next_item_level+1])
            else:
                self.level_dict[next_item_level].extend(next_item)
        
        self.level_dict.pop(-1)

        return

    type_func_dict = {
            'scalars': 'Scalars',
            'graph': 'Graphs',
            'meta_graph': 'MetaGraph',
            'run_metadata': 'RunMetadata',
            'histograms': 'Histograms',
            'distributions': 'CompressedHistograms',
            'images': 'Images',
            'audio': 'Audio',
            'tensors': 'Tensors'
            }

    @staticmethod
    def _type_load_data(file, type_name, type_func, tags, query_expression):

        acc = EventAccumulator(file)
        acc.Reload()

        run_tags = acc.Tags()[type_name] if tags is None else tags

        results = []

        for t in run_tags:

            try:          
                events =  getattr(acc, type_func)(tag=t)
            except KeyError:
                return pd.DataFrame()

            rows = [{'value': e.value, 'step': e.step} for e in events]

            results_temp = {
                'run': file,
                'type': type_name,
                'tag': t, 
                'rows': rows, 
                }
            names = file.replace("\\", '__--__').replace("/",'__--__').split('__--__')
            names = {f'level_{nl}': name for nl, name in enumerate(names[:-1])}
            results_temp.update(names)
            
            results_temp = (pd.json_normalize(
                results_temp, 
                record_path='rows', 
                meta=['run', 'type', 'tag'] + list(names.keys())
                )
                [['run'] + list(names.keys()) + ['type', 'tag', 'step', 'value']]
                )
            results.append(results_temp)
            
        if len(results) > 0:
            results = pd.concat(results)
            if not query_expression is None:
                results = results.query(query_expression)
        else:
            results = pd.DataFrame()
        return results

    def _type_loader(
        self, 
        type_name:str,
        tags:typing.Union[typing.List[str], str, None]=None, 
        query_expression:typing.Union[str, None]=None,
        ) -> typing.Dict[int, pd.DataFrame]:
        if type(tags) == str:
            tags = [tags]
        elif type(tags) == list:
            pass
        elif tags is None:
            pass
        else:
            raise TypeError("Please ensure that tags is a str, list of str, or None.")

        results = {}

        # loading the event accumulators
        n_files = sum(map(len, self.level_dict.values()))
        tqdm_progress = tqdm.tqdm(
            total=n_files, 
            desc='Loading Files', 
            disable=not self.verbose,
            **tqdm_style,
            )

        for level, files in self.level_dict.items():

            parallel_func = functools.partial(
                self._type_load_data,
                type_name=type_name,
                type_func=self.type_func_dict[type_name],
                tags=tags,
                query_expression=query_expression,
                )

            parallel_comps = [
                joblib.delayed(parallel_func)( 
                    file=file,
                    )
                for file in files
                ]

            level_results = ProgressParallel(
                tqdm_bar=tqdm_progress, 
                n_jobs=self.n_jobs,
                )(parallel_comps)

            # delete parallel processes
            get_reusable_executor().shutdown(wait=True)

            if len(level_results) > 0:
                level_results = pd.concat(level_results)
                results[level] = level_results.reset_index(drop=True)
            else:
                results[level] = pd.DataFrame()
        
        tqdm_progress.close()
        
        return results
    
    @staticmethod
    def _load_file_tags(file,):

        acc = EventAccumulator(file)
        acc.Reload()

        run_tags = acc.Tags()

        return run_tags

    @property
    def tags(self,):
        '''
        This returns the available tags that can be used 
        to filter the results when loading files.
        '''

        # loading the event accumulators
        n_files = sum(map(len, self.level_dict.values()))
        tqdm_progress = tqdm.tqdm(
            total=n_files, 
            desc='Loading Files', 
            disable=not self.verbose,
            **tqdm_style,
            )

        for level, files in self.level_dict.items():

            parallel_comps = [
                joblib.delayed(self._load_file_tags)( 
                    file=file,
                    )
                for file in files
                ]

            tag_dict_list = ProgressParallel(
                tqdm_bar=tqdm_progress, 
                n_jobs=self.n_jobs,
                )(parallel_comps)

            # delete parallel processes
            get_reusable_executor().shutdown(wait=True)

        result = {}
        for each_dict in tag_dict_list:
            for key, values in each_dict.items():
                if key not in result:
                    result[key] = set([])
                if hasattr(values, '__iter__'):
                    for v in values:
                        result[key].add(v)
                else:
                    result[key].add(v)
        result = {key: list(values) for key, values in result.items()}

        return result

    def scalars(
        self, 
        tags:typing.Union[typing.List[str], str, None]=None, 
        query_expression:typing.Union[str, None]=None,
        ) -> typing.Dict[int, pd.DataFrame]:
        '''
        This function collects all of the scalars
        written to the tensorboard, with the given
        tag and querying expression.
        
        Examples
        ---------
        
        The following would load the accuracy
        values from all files, in which the accuracy
        is more than or equal to 0.5.

        .. code-block:: 
        
            >>> tbload = TensorboardLoad('./')
            >>> tbload.scalars(
            ...     'Accuracy', 
            ...     query_expression="value >= 0.5",
            ...     )
        
        
        Arguments
        ---------
        
        - tags: typing.Union[typing.List[str], str, None], optional:
            The tag of the results that are required. If :code:`None`
            then all tags are returned. 
            Defaults to :code:`None`.
        
        - query_expression: typing.Union[str, None], optional:
            An expression that will be passed to the pandas dataframe
            that allows the user to filter out un-wanted rows
            before that dataframe is concatenated with the results. 
            Defaults to :code:`None`.
        

        Raises
        ---------
        
            TypeError: If the tag is not a string, a list of strings or None.
        
        Returns
        --------
        
        - out: typing.Dict[int, pd.DataFrame]: 
            Pandas dataframe containing the results.
        
        
        '''

        return self._type_loader(
            type_name='scalars', 
            tags=tags, 
            query_expression=query_expression,
            )

    def histograms(
        self, 
        tags:typing.Union[typing.List[str], str, None]=None, 
        query_expression:typing.Union[str, None]=None,
        ) -> typing.Dict[int, pd.DataFrame]:
        '''
        This function collects all of the histograms
        written to the tensorboard, with the given
        tag and querying expression.
        
        Examples
        ---------
        
        The following would load the accuracy
        histograms from all files.

        .. code-block:: 
        
            >>> tbload = TensorboardLoad('./')
            >>> tbload.histograms(
            ...     'Accuracy', 
            ...     )
        
        
        Arguments
        ---------
        
        - tags: typing.Union[typing.List[str], str, None], optional:
            The tag of the results that are required. If :code:`None`
            then all tags are returned. 
            Defaults to :code:`None`.
        
        - query_expression: typing.Union[str, None], optional:
            An expression that will be passed to the pandas dataframe
            that allows the user to filter out un-wanted rows
            before that dataframe is concatenated with the results. 
            Defaults to :code:`None`.
        

        Raises
        ---------
        
            TypeError: If the tag is not a string, a list of strings or None.
        
        Returns
        --------
        
        - out: typing.Dict[int, pd.DataFrame]: 
            Pandas dataframe containing the results.
        
        
        '''

        return self._type_loader(
            type_name='histograms', 
            tags=tags, 
            query_expression=query_expression,
            )

    def distributions(
        self, 
        tags:typing.Union[typing.List[str], str, None]=None, 
        query_expression:typing.Union[str, None]=None,
        ) -> typing.Dict[int, pd.DataFrame]:
        '''
        This function collects all of the distributions
        written to the tensorboard, with the given
        tag and querying expression.
        
        Examples
        ---------
        
        The following would load the accuracy
        distributions from all files.

        .. code-block:: 
        
            >>> tbload = TensorboardLoad('./')
            >>> tbload.distributions(
            ...     'Accuracy', 
            ...     )
        
        
        Arguments
        ---------
        
        - tags: typing.Union[typing.List[str], str, None], optional:
            The tag of the results that are required. If :code:`None`
            then all tags are returned. 
            Defaults to :code:`None`.
        
        - query_expression: typing.Union[str, None], optional:
            An expression that will be passed to the pandas dataframe
            that allows the user to filter out un-wanted rows
            before that dataframe is concatenated with the results. 
            Defaults to :code:`None`.
        

        Raises
        ---------
        
            TypeError: If the tag is not a string, a list of strings or None.
        
        Returns
        --------
        
        - out: typing.Dict[int, pd.DataFrame]: 
            Pandas dataframe containing the results.
        
        
        '''

        return self._type_loader(
            type_name='distributions', 
            tags=tags, 
            query_expression=query_expression,
            )

    def images(
        self, 
        tags:typing.Union[typing.List[str], str, None]=None, 
        query_expression:typing.Union[str, None]=None,
        ) -> typing.Dict[int, pd.DataFrame]:
        '''
        This function collects all of the images
        written to the tensorboard, with the given
        tag and querying expression.
        
        Examples
        ---------
        
        The following would load the accuracy
        images from all files.

        .. code-block:: 
        
            >>> tbload = TensorboardLoad('./')
            >>> tbload.images(
            ...     'Accuracy', 
            ...     )
        
        
        Arguments
        ---------
        
        - tags: typing.Union[typing.List[str], str, None], optional:
            The tag of the results that are required. If :code:`None`
            then all tags are returned. 
            Defaults to :code:`None`.
        
        - query_expression: typing.Union[str, None], optional:
            An expression that will be passed to the pandas dataframe
            that allows the user to filter out un-wanted rows
            before that dataframe is concatenated with the results. 
            Defaults to :code:`None`.
        

        Raises
        ---------
        
            TypeError: If the tag is not a string, a list of strings or None.
        
        Returns
        --------
        
        - out: typing.Dict[int, pd.DataFrame]: 
            Pandas dataframe containing the results.
        
        
        '''

        return self._type_loader(
            type_name='images', 
            tags=tags, 
            query_expression=query_expression,
            )

    def audio(
        self, 
        tags:typing.Union[typing.List[str], str, None]=None, 
        query_expression:typing.Union[str, None]=None,
        ) -> typing.Dict[int, pd.DataFrame]:
        '''
        This function collects all of the audio
        written to the tensorboard, with the given
        tag and querying expression.
        
        Examples
        ---------
        
        The following would load the accuracy
        audio from all files.

        .. code-block:: 
        
            >>> tbload = TensorboardLoad('./')
            >>> tbload.audio(
            ...     'Accuracy', 
            ...     )
        
        
        Arguments
        ---------
        
        - tags: typing.Union[typing.List[str], str, None], optional:
            The tag of the results that are required. If :code:`None`
            then all tags are returned. 
            Defaults to :code:`None`.
        
        - query_expression: typing.Union[str, None], optional:
            An expression that will be passed to the pandas dataframe
            that allows the user to filter out un-wanted rows
            before that dataframe is concatenated with the results. 
            Defaults to :code:`None`.
        

        Raises
        ---------
        
            TypeError: If the tag is not a string, a list of strings or None.
        
        Returns
        --------
        
        - out: typing.Dict[int, pd.DataFrame]: 
            Pandas dataframe containing the results.
        
        
        '''

        return self._type_loader(
            type_name='audio', 
            tags=tags, 
            query_expression=query_expression,
            )

    def tensors(
        self, 
        tags:typing.Union[typing.List[str], str, None]=None, 
        query_expression:typing.Union[str, None]=None,
        ) -> typing.Dict[int, pd.DataFrame]:
        '''
        This function collects all of the tensors
        written to the tensorboard, with the given
        tag and querying expression.
        
        Examples
        ---------
        
        The following would load the accuracy
        tensors from all files.

        .. code-block:: 
        
            >>> tbload = TensorboardLoad('./')
            >>> tbload.tensors(
            ...     'Accuracy', 
            ...     )
        
        
        Arguments
        ---------
        
        - tags: typing.Union[typing.List[str], str, None], optional:
            The tag of the results that are required. If :code:`None`
            then all tags are returned. 
            Defaults to :code:`None`.
        
        - query_expression: typing.Union[str, None], optional:
            An expression that will be passed to the pandas dataframe
            that allows the user to filter out un-wanted rows
            before that dataframe is concatenated with the results. 
            Defaults to :code:`None`.
        

        Raises
        ---------
        
            TypeError: If the tag is not a string, a list of strings or None.
        
        Returns
        --------
        
        - out: typing.Dict[int, pd.DataFrame]: 
            Pandas dataframe containing the results.
        
        
        '''

        return self._type_loader(
            type_name='tensors', 
            tags=tags, 
            query_expression=query_expression,
            )
