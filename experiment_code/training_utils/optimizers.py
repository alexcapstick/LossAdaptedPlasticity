
class CombineOptimizers(object):
    def __init__(self, *optimizers):
        '''
        Allows the user to add multiple optimizers and update parameters with
        a single :code:`.step()` call.
        This code has been taken from:
        https://discuss.pytorch.org/t/two-optimizers-for-one-model/11085/7
        
        
        Arguments
        ---------

        - optimizers: torch.optim:
            The optimizers to combine into one class. You 
            may pass as many as you like.

        '''

        self.optimizers = optimizers

    def zero_grad(self):
        for op in self.optimizers:
            op.zero_grad()

    def step(self, *args, **kwargs):
        for op in self.optimizers:
            op.step(*args, **kwargs)

