# If You're Just Looking for the Graphs!

These can be found in the notebooks:

- `graph_synthetic.ipynb`
- `graph_ecg.ipynb`
- `graph_cifarn_different_noise.ipynb`

Read on for more information on how to run the experiments yourself, and how you can use LAP:


# Code for Loss Adapted Plasticity

This directory contains the code required to run the experiments to produce the results presented in the paper "Loss Adapted Plasticity in Deep Neural Networks to Learn from Data with Unreliable Sources"

To get started and download all dependencies, run:

    pip3 install -r requirements.txt 

## LAP Optimisers

The key code in this directory relates to the LAP optimizer wrapper (written in pytorch), which can be used in conjunction with any torch optimizer such as Adam and SGD, with the only difference being that they need to be passed the loss and the source when performing the step.

In current pytorch code, the optimisation process looks as follows:

```python
inputs, labels = data_batch
# ======= forward ======= 
outputs = model(inputs)
loss = criterion(outputs, labels)
# ======= backward =======
loss.backward()
optimizer.step()
```

When using a LAP optimiser, this needs to be changed to (with the key difference being the optimiser step `optimizer.step(loss, source)`):

```python
inputs, labels = data_batch
# ======= forward ======= 
outputs = model(inputs)
loss = criterion(outputs, labels)
# ======= backward =======
loss.backward()
optimizer.step(loss, source) # key difference
```

The code for these optimizers can be found in `./loss_adapted_plasticity/lap_wrapper.py`, as the class `LAP`. They can be imported from this directory using the code:

```python
from loss_adapted_plasticity import LAP
```

And can be used as follows:

```python
import torch
optimizer = LAP(torch.optim.Adam, lr=0.01)
```

## Training and Testing on Synthetic Data

To train the models on synthetic data, please use the `test_hparam_synthetic.py` or `test_hparam_different_noise_synthetic.py` files with the desired arguments for that given experiment. 


The corruption and their corresponding short name is given in the list:

- No Corruption : `no_c`
- Chunk Shuffle : `c_cs`
- Random Label : `c_rl`
- Batch Label Shuffle : `c_lbs`
- Batch Label Flip : `c_lbf`
- Added Noise : `c_ns`
- Replace With Noise : `c_no`


The optional hyper-parameters used when training using LAP with these scripts are:

- Depression Strength: `--depression-strength`
    - This is the strength of the depression applied and is defined as $d_\zeta$ in the paper.
- Strictness Parameter: `--strictness`
    - This is the strictness parameter defined as $\lambda$ in the paper.
- Loss History Length:  `--lap-n`
    - This is the number of losses to store for each source when making the depression calculations.
- Hold Off Value: `--hold-off`
    - This is the number of calls to the depression function before depression will be applied. It allows you to control when depression is applied.



### Training on CIFAR-10

For training the LAP trained model, please run:

    python test_hparam_synthetic.py --model-name Conv3Net-[Corruption Short Name]-drstd --seed 2 4 8 16 32 64 128 256 512 1024 --dataset-name cifar10 --n-sources 10 --n-corrupt-sources 4 --source-size 128 --depression-strength 1.0 --strictness 0.8 --lap-n 25 --n-epochs 25 -v

For standard model training, please use (the key difference being `--depression-strength 0.0`):

    python test_hparam_synthetic.py --model-name Conv3Net-[Corruption Short Name]-drstd --seed 2 4 8 16 32 64 128 256 512 1024 --dataset-name cifar10 --n-sources 10 --n-corrupt-sources 4 --source-size 128 --depression-strength 0.0 --n-epochs 25 -v

For training the oracle model, please run: 

    python test_hparam_synthetic.py --model-name Conv3Net-[Corruption Short Name]_srb-drstd --seed 2 4 8 16 32 64 128 256 512 1024 --dataset-name cifar10 --n-sources 10 --n-corrupt-sources 4 --source-size 128 --depression-strength 0.0 --n-epochs 25 -v

The oracle model is not implemented for the "No Corruption" case, since here there is no need for an oracle method.


### Training on CIFAR-100

To train the LAP trained model, please run:

    python test_hparam_synthetic.py --model-name Conv3Net_100-[Corruption Short Name]-drstd --seed 2 4 8 16 32 64 128 256 512 1024 --dataset-name cifar100 --n-sources 10 --n-corrupt-sources 2 --source-size 128 --depression-strength 1.0 --strictness 0.8 --lap-n 25 --hold-off 250 --n-epochs 40 -v

Similarly, for standard training:

    python test_hparam_synthetic.py --model-name Conv3Net_100-[Corruption Short Name]-drstd --seed 2 4 8 16 32 64 128 256 512 1024 --dataset-name cifar100 --n-sources 10 --n-corrupt-sources 2 --source-size 128 --depression-strength 0.0 --n-epochs 40 -v

And for the oracle method:
    
    python test_hparam_synthetic.py --model-name Conv3Net_100-[Corruption Short Name]_srb-drstd --seed 2 4 8 16 32 64 128 256 512 1024 --dataset-name cifar100 --n-sources 10 --n-corrupt-sources 2 --source-size 128 --depression-strength 0.0 --n-epochs 40 -v

The oracle model is not implemented for the "No Corruption" case, since here there is no need for an oracle method.

### Training on F-MNIST

To train the LAP trained model, please run:

    python test_hparam_synthetic.py --model-name MLP-[Corruption Short Name]-drstd --seed 2 4 8 16 32 64 128 256 512 1024 --dataset-name fmnist --n-sources 10 --n-corrupt-sources 6 --source-size 200 --depression-strength 1.0 --strictness 0.8 --lap-n 50 --n-epochs 40 -v

Similarly, for standard training:

    python test_hparam_synthetic.py --model-name MLP-[Corruption Short Name]-drstd --seed 2 4 8 16 32 64 128 256 512 1024 --dataset-name fmnist --n-sources 10 --n-corrupt-sources 6 --source-size 200 --depression-strength 0.0 --n-epochs 40 -v

And for the oracle method:
    
    python test_hparam_synthetic.py --model-name MLP-[Corruption Short Name]_srb-drstd --seed 2 4 8 16 32 64 128 256 512 1024 --dataset-name fmnist --n-sources 10 --n-corrupt-sources 6 --source-size 200 --depression-strength 0.0 --n-epochs 40 -v


The oracle model is not implemented for the "No Corruption" case, since here there is no need for an oracle method.

### Different Noise on CIFAR-10

Experiments using LAP on CIFAR-10 with different noise rates can be run using the script `test_hparam_different_noise_synthetic.py`. For our experiments, we used the `.bat` file in the directory `./experiment_commands/`, which can be easily adapted to `bash` if required.

### Graphing

The graphs can be produced by following the code in `graph_synthetic.ipynb`.


### Where are things saved?

A tensorboard file will be saved in the `./runs/` directory, which shows various metrics and information about when depression was applied, how much depression was applied and the metrics on each source.

Whilst training, a figure will be created within the directory `./outputs/results/`, which shows the training and validation loss, and can be used to track an ongoing experiment.

Pytorch state dicts will be saved in the directory `./outputs/models/`. Only a completely trained model will be saved here.

Test results will then be saved in the directory `./outputs/synthetic_results/`.


## PTB-XL ECG (Real-world Data)

For testing the LAP trained and standard model, please run:

    python test_ecg.py

With the arguments that you want to test. For our experiments, we ran:

    python test_ecg.py -v --seed 2 4 8 16 32 64 128 256 512 1024 --n-jobs 6 --ds 1 
and

    python test_ecg.py -v --seed 2 4 8 16 32 64 128 256 512 1024 --n-jobs 6 --ds 0

### Graphing

To graph the results, please follow the ipython notebook `graph_ecg.ipynb`

### Where are things saved?

This will save the test results in the directory `./outputs/ecg_results/`. Each file will be named according the model and seed that the results correspond to.


## CIFAR-10N (Real-world Data)

For testing the LAP trained and standard model, please run:

    python test_cifarn_different_noise.py

With the arguments that you want to test. For our experiments, we ran:

    python test_cifarn_different_noise.py --seed 2 4 8 16 32 64 128 256 512 1024 

### Graphing


To graph the results, please follow the ipython notebook `graph_cifarn_different_noise.ipynb`

### Where are things saved?

This will save the test results in the directory `./outputs/cifarn_results/`. Each file will be named according the dataset and seed that the results correspond to.



## Examples

A IPython Notebook is located in this directory, called `examples.ipynb`, which contains examples on loading data and models and performing testing and training, in case you want to run your own experiments using these models or optimisers. It also gives an example on how LAP can be used in your own models!