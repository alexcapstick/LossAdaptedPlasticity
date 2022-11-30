import torch
import numpy as np
from sklearn.metrics import precision_recall_curve, auc


def auc_precision_recall_curve(
    y_true:np.ndarray, 
    y_proba:np.ndarray, 
    pos_label=None, 
    sample_weight=None,
    ) -> float:
    '''
    A function that calculates the area under
    the precision-recall curve
    between two arrays. This is modelled 
    on the Scikit-Learn :code:`recall_score`, 
    :code:`precision_score`, :code:`accuracy_score`,
    and :code:`f1_score`.

    Examples
    ---------

    .. code-block::

        >>> import numpy as np
        >>> auc_precision_recall_curve(
        ...     y_true=np.array([0,1,0,1,0]),
        ...     y_proba=np.array([0,0,0,1,0]),
        ...     )
        0.85


    Arguments
    ---------

    - y_true: np.ndarray:
        The array of true values.

    - y_proba: np.ndarray:
        The array of predicted score values. If :code:`y_pred` 
        has shape :code:`(N,2)`, and :code:`y_true` has two unique
        values, then the probability of a positive class will 
        be assumed to be :code:`y_proba[:,1]`.

    - pos_label: typing.Union[str, int], optional:
        The class to report if :code:`average='binary'` and the data is binary.
        If the data are multiclass or multilabel, this will be ignored;
        setting :code:`labels=[pos_label]` and 
        :code:`average != 'binary'` will report
        scores for that label only.
        Defaults to :code:`1`.

    - sample_weight: typing.Union[np.ndarray, None], optional:
        Sample weights.
        Defualts to :code:`None`.
    
    Returns
    ---------

    - auc: float:
        The area under the precision-recall curve.
    
    '''

    if len(y_proba.shape) == 2:
        if len(np.unique(y_true)) == 2:
            y_proba = y_proba[:,1]

    y, x, _ = precision_recall_curve(
        y_true, 
        y_proba, 
        pos_label=pos_label, 
        sample_weight=sample_weight,
        )
    
    return auc(x,y)



# the following is from https://discuss.pytorch.org/t/top-k-error-calculation/48815/3
# with edits to the documentation and commenting.
def accuracy_topk(output:torch.tensor, target:torch.tensor, topk:tuple=(1,)):
    """
    https://discuss.pytorch.org/t/top-k-error-calculation/48815/3

    Computes the accuracy over the k top predictions for the specified values of k.

    ref:
    - https://pytorch.org/docs/stable/generated/torch.topk.html
    - https://discuss.pytorch.org/t/imagenet-example-accuracy-calculation/7840
    - https://gist.github.com/weiaicunzai/2a5ae6eac6712c70bde0630f3e76b77b
    - https://discuss.pytorch.org/t/top-k-error-calculation/48815/2
    - https://stackoverflow.com/questions/59474987/how-to-get-top-k-accuracy-in-semantic-segmentation-using-pytorch


    Arguments
    ---------

    - output: torch.tensor:
        The prediction of the model.
    
    - target: torch.tensor:
        The targets that each prediction corresponds to.
    
    - topk: tuple (optional):
        This is a tuple of values that represent the k values
        for which the accuracy should be calculated with.


    Returns
    ---------

    - topk_accuracies: list:
        This returns a list of the top k accuracies for
        the k values specified.

    """
    with torch.no_grad():
        maxk = max(topk)  # max number labels we will consider in the right choices for out model
        batch_size = target.size(0)
        # get top maxk indicies that correspond to the most likely probability scores
        _, y_pred = output.topk(k=maxk, dim=1) 
        y_pred = y_pred.t()  # [B, maxk] -> [maxk, B]

        # expand the target tensor to the same shape as y_pred
        target_reshaped = target.view(1, -1).expand_as(y_pred)  # [B] -> [B, 1] -> [maxk, B]
        # compare the target to each of the top k predictions made by the model
        correct = (y_pred == target_reshaped)  # [maxk, B] for each example we know which topk prediction matched truth

        # get topk accuracy
        list_topk_accs = [] 
        for k in topk:
            # find which of the top k predictions were correct
            ind_which_topk_matched_truth = correct[:k]  # [maxk, B] -> [k, B]
            # calculate the number of correct predictions
            flattened_indicator_which_topk_matched_truth = ind_which_topk_matched_truth.reshape(-1).float()  # [k, B] -> [kB]
            tot_correct_topk = flattened_indicator_which_topk_matched_truth.float().sum(dim=0, keepdim=True)  # [kB] -> [1]
            topk_acc = tot_correct_topk / batch_size  # topk accuracy for entire batch
            list_topk_accs.append(topk_acc)
        return list_topk_accs  # list of topk accuracies for batch [topk1, topk2, ... etc]