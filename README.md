# Computer Vision Scripts for the Painless Panes Project

If you haven't already, set up the Python environment before getting started following the instructions below.
Once that is done, you can simply activate the Python environment using
```
poetry shell
```
and proceed through the following steps:
1. Follow the instructions in `01-data-generation/README.md` to download and annotate your training data.
2. Follow the instructions in `02-model-training/README.md` to run the model training.



## Setting up the Python environment

Install the Python package manager `Poetry` by following the instrucions [here](https://python-poetry.org/docs/#installation).

Once that is installed, you can create the enviornment for the project by running
```
poetry install
```
in this directory.


> ### Side note:
> 
> If creating a new poetry environment, you can install the CPU version of torch as follows:
> ```
> poetry source add --priority=explicit pytorch-cpu-src https://download.pytorch.org/whl/cpu
> poetry add --source pytorch-cpu-src torch torchvision torchaudio
> ```