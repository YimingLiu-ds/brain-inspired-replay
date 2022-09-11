# Research Project:
## Brain-inspired contrastive learning and visual attention in generative replay for Continual Learning
This project builds upon the work of Van de Ven et al, 2020, and Jack Millichamp, 2021, adding extensive modifications to their PyTorch implementation. Most alterations were made within the `models/VAE.py`, `models/attention.py`, `main_cl.py` and `train.py` Entirely new additions have been highlighted through the use of a quadruple hashtag: `####`. 


## Installation & requirements
The current version of the code has been tested with `Python 3.5.2` on both Linux and Windows operating systems with the following versions of PyTorch and Torchvision:
* `pytorch 1.9.0`
* `torchvision 0.2.2`
* `kornia`
* `scikit-learn`
* `matplotlib`
* `visdom`
* `torchmetrics`
* `scipy`


To use the code, download the repository and change into it:
```bash
git clone https://github.com/YimingLiu-ds/brain-inspired-replay.git
cd brain-inspired-replay
```
(If downloading the zip-file, extract the files and change into the extracted folder.)
 
Assuming Python3 is set up, the Python-packages used by this code can be installed into a fresh docker image via the included dockerfile.

The required datasets do not need to be explicitly downloaded, this will be done automatically the first time the code is run.


## Running the experiments
All experiments performed in the project can be run through `main_cl.py` using various flags for the diferent experiments. The main universal parameters for these experiments are:
- `--iters`: the number of iterations per segment/task (default is 5000)
- `--batch`: the batch size for samples from new classes 
- `--batch-replay`: the batch size for samples from replayed classes

### Baseline
To run the baseline experiment used for comparisons throughout the project, the following code should be run from the command line once changed into the 'brain-inspired-replay' folder:
```bash
main_cl.py --experiment=CIFAR100 --scenario=class --replay=generative --brain-inspired --si --c=1e8 --dg-prop=0.6 --pdf --batch=512
```
This will run a single experiment using the optimal hyperparameters identified by Van de Ven et al, 2020. Using a GPU this should take just over 1 hour, however, as mentioed above it is possible to reduce the number of iterations per segment from the default of 5000 using the `--iters` flag, which will significantly reduce runtime but compromise on final classifiaction precision. The `--brain-inspired --si` flags ensure that the experiment is run via the state-of-the-art BI-R with Synaptic Intelligence (SI) model.

### Multi-head attention and Simple Siamese(MASS)
To run an experiment using the Multi-head attention and Simple Siamese, please add the flag `--contrastive --c-scores --simsiam --attention --ma --c-lr=1e-6 --attention --wd=1e-1` to the baseline implementation:
```bash
main_cl.py --experiment=CIFAR100 --scenario=class --replay=generative --brain-inspired --si --c=1e8 --dg-prop=0.6 --pdf --contrastive --c-scores --simsiam --attention --ma --c-lr=1e-6 --attention --wd=1e-1 --batch=512
```
### External attention and Simple Siamese(EASS)
To run an experiment using the External attention and Simple Siamese, please add the flag `--contrastive --c-scores --simsiam --attention --c-lr=1e-6 --attention --wd=1e-1` to the baseline implementation:
```bash
main_cl.py --experiment=CIFAR100 --scenario=class --replay=generative --brain-inspired --si --c=1e8 --dg-prop=0.6 --pdf --contrastive --c-scores --simsiam --attention --c-lr=1e-6 --attention --wd=1e-1 --batch=512
```

### Reconstruction repulsion and reconstruction attraction model with confidence labels
The reconstruction-based loss can be implementated by the addition of `--recon-repulsion`, `--recon-attraction` (for repulsion and attraction respectively) and `--use-rep-f`. In addition the use of class-averaging, as described in the paper, can be achieved through the `--recon-rep-aver` flag:
```bash
main_cl.py --experiment=CIFAR100 --scenario=class --replay=generative --brain-inspired --si --c=1e8 --dg-prop=0.6 --pdf --recon-repulsion --recon-attraction --recon-rep-aver --use-rep-f
```
The optimal hyperparameters for this are set as default, however to alter the weightings of the repulsion and attraction losses please use the `--lamda-recon-rep` and `--lamda-recon-atr` flags respectively.

The default hyperparamters are the idenfied optimal ones, however to select custom hyperparameters please use the following flags:
- `--c-temp`: temperature of contrastive loss
- `--c-lr`: learning rate of contrastive learning optimiser
- `--c-drop`: drop-out rate of contrastive learning projection head
- `--wd`: weight decay of contrastive learning encoder optimizer
- `--ma_drop`: drop-out rate of multi-headed attention 

For further information on the above flag options and a full list of possible flags, please run: `main_cl.py -h`.


## On-the-fly plots during training
With this code it is possible to track progress during training with on-the-fly plots. This feature requires `visdom`.
Before running the experiments, the visdom server should be started from the command line:
```bash
python -m visdom.server
```
The visdom server is now alive and can be accessed at `http://localhost:8097` in your browser (the plots will appear
there). The flag `--visdom` should then be added when calling `main_cl.py` to run the experiments with on-the-fly plots.
