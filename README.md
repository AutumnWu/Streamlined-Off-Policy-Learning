# Streamlined-Off-Policy Pytorch Implementation
Streamlined-Off-Policy Pytorch Implementation, based on the OpenAI Spinup documentation and some of its code base. This SOP implementation is based on the OpenAI spinningup repo, and uses spinup as a dependency. 
Currently anonymized for reviewing.

## Setup environment:
To use the code you should first download this repo, and then install spinup:

The spinup documentation is here, you should read it to make sure you know the procedure: https://spinningup.openai.com/en/latest/user/installation.html

The only difference in installation is you want to install this forked repo, instead of the original repo.

The Pytorch version used is: 1.2, install pytorch:
https://pytorch.org/

Mujoco version is 150

## Run experiment
The SOP implementation can be found under `spinup/algos/sop_pytorch/`
The SOP-IG implementation can be found under `spinup/algos/sop_ig_pytorch/`
The SAC implementation can be found under `spinup/algos/sac_pytorch/`

Run experiments with pytorch sop:

In the sop_pytorch folder, run the SOP code with `python SOP.py`

Run experiments with pytorch sop-ig: 

In the sop_ig_pytorch folder, run the SOP-IG code with `python sop_invert_grad.py`

Run experiments with pytorch sac: 

In the sac_pytorch folder, run the SAC code with `python sac_pytorch.py`

Note: currently there is no parallel running for SAC and SOP (also not supported by spinup), so you should always set number of cpu to 1 when you use experiment grid.

The program structure, though in Pytorch has been made to be as close to spinup tensorflow code as possible so readers who are familiar with other algorithm code in spinup will find this one easier to work with. I also referenced rlkit's SAC pytorch implementation, especially for the policy and value models part, but did a lot of simplification. 

Consult Spinup documentation for output and plotting:

https://spinningup.openai.com/en/latest/user/saving_and_loading.html

https://spinningup.openai.com/en/latest/user/plotting.html


## Reference: 

Original SAC paper: https://arxiv.org/abs/1801.01290

OpenAI Spinup docs on SAC: https://spinningup.openai.com/en/latest/algorithms/sac.html

rlkit sac implementation: https://github.com/vitchyr/rlkit

The code will be released as a public github repo, with better documentation after the reviewing process. 
