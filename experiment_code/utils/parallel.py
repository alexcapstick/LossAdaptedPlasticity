import typing
import numpy as np
import joblib

from .parallel_progress import ProgressParallel



class ParallelModelling:
    def __init__(
        self, 
        models:typing.Any,
        n_jobs:int=1,
        backend:str='threading',
        **kwargs,
        ):
        '''
        This class allows you to train, predict, or 
        anything else on models in parallel.
        
        In fact, this class is general enough to accept
        any object to perform parallel methods on.
        
        
        
        Examples
        ---------
        
        .. code-block::

            >>> n_runs = 20
            >>> pf = ParallelModelling(
            ...     models=[
            ...         ResNetLearning(
            ...             seed=seed, 
            ...             model_name=f'RN-seed_{seed}',
            ...             **model_args
            ...             )
            ...         for seed in range(n_runs)
            ...         ],
            ...     n_jobs=2
            ...     )
            >>> pf.fit(train_loader=train_dl, val_loader=val_dl)
            >>> predictions = pf.predict(train_loader=train_dl, val_loader=val_dl)

        Similarly, if different models need to be trained with different
        train loaders, you may also use the :code:`list__` prefix. If doing
        this, please ensure that the argument lists are the same size as
        the number of models being trained. An example would be:

        .. code-block::

            >>> pf.fit(list__train_loader=train_dl_list, list__val_loader=val_dl_list)
        
        The parallel computations are done using the :code:`backend='threading'` 
        by default, which means that memory is shared across the processes.
        This will ensure that datasets can be shared across the computations.
        If this is not desired, please use :code:`backend='loky'`.

        The returned values for the methods will be a list
        of the returned values for the method on each
        of the models. You may also call attributes of the models, which 
        will be returned in the same way. 
        
        Note, that if an attribute is callable, then 
        it will be returned as a parallel function, which 
        means that if an attribute is callable and has its own attributes,
        it will not be accessible in this way. An example of this special
        case is when trying to access the weights of pytorch models
        in parallel. :code:`pf.layer1.weight` will raise an error, as 
        :code:`pf.layer1` is callable. To access the weights, please
        instead use: :code:`[model.layer1.weight for model in pf.models]`.


        Arguments
        ---------
        
        - models: typing.Any: 
            The models to be wrapped and accessed in parallel.
            Please ensure that they all have the attribute 
            that you are hoping to call.
        
        - n_jobs: int, optional:
            The number of jobs to run in parallel. Be mindful that
            the models and related computations might be expensive
            for the CPU, GPU, or RAM.
            Defaults to :code:`1`.
        
        - backend: str, optional:
            The backend used for the parallel compute. This should
            be an acceptable value for :code:`joblib.Parallel`.
            Defaults to :code:`threading`.
        
        - kwargs: 
            All other keyword arguments are passed to
            :code:`joblib.Parallel`.



        Attributes
        ----------

        - models:
            The models supplied in the :code:`__init__` function.
        
        - parallel_args: dict:
            The arguments passed to the :code:`joblib.Parallel`
            :code:`__init__` function. This will include the number
            of jobs and the backend being used. Arguments not shown 
            here, but used in :code:`joblib.Parallel`, will be set
            to the default values.


        '''
        self.models = models
        self.parallel_args = {'backend': backend, 'n_jobs': n_jobs}
        self.parallel_args.update(**kwargs)
        return

    def _construct_inputs(
        self,
        **kwargs,
        ):
        
        list_kwargs = {}
        reused_kwargs = {}
        
        for key, value in kwargs.items():
            if 'list__' in key:
                if len(value) != len(self.models):
                    raise TypeError('Please ensure that list__ kwargs '\
                        'have the same length as the number of models.')
                list_kwargs[key.replace('list__', '')] = value
            else:
                reused_kwargs[key] = value

        list_kwargs = [
            dict(zip(list_kwargs.keys(), values)) 
            for values in zip(*list_kwargs.values())
            ]
        if len(list_kwargs) == 0:
            list_kwargs = [{}]*len(self.models)

        return list_kwargs, reused_kwargs

    # defined since __getattr__ causes pickling problems
    def __getstate__(self):
        return vars(self)

    # defined since __getattr__ causes pickling problems
    def __setstate__(self, state):
        vars(self).update(state)

    def __getattr__(self, name):
        if np.all([hasattr(model, name) for model in self.models]):
            if np.all([callable(getattr(model, name)) for model in self.models]):
                def parallel_attribute(**kwargs):
                    list_kwargs, reused_kwargs = self._construct_inputs(**kwargs)
                    output = ProgressParallel(
                        desc=f'Parallel {name}',
                        total=len(self.models),
                        **self.parallel_args
                        )(
                            joblib.delayed(getattr(model, name))(
                                **model_kwargs,
                                **reused_kwargs,
                                )
                            for (
                                model, 
                                model_kwargs, 
                                ) in zip(self.models, list_kwargs)
                            )
                    return output
                return parallel_attribute
            else:
                output = ProgressParallel(
                    desc=f'Parallel {name}',
                    total=len(self.models),
                    **self.parallel_args
                    )(
                        joblib.delayed(getattr)(model, name)
                        for model in self.models
                        )
                return output
        else:
            raise AttributeError("Please ensure all models have this attribute.")