[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/T_Fzxg5j)
# AutoML lecture 2023 (Freiburg & Hanover)
## Final Project.

This repository contains all things needed for the final projects.
Your task is to optimize a NN by AutoML means.
For details, please refer to the project PDF.

### (Recommended) Setup new clean environment

Use a package manager, such as the one provided by your editor, python's built in `venv`
or [miniconda](https://docs.conda.io/en/latest/miniconda.html#system-requirements).

#### Conda
Subsequently, *for example*, run these commands, following the prompted runtime instructions:
```bash
conda create -n automl python=3.10
conda activate automl
pip install -r requirements.txt
```

#### Venv

```bash
# Make sure you have python 3.8/3.9/3.10
python -V
python -m venv my-virtual-env
./my-virtual-env/bin/activate
pip install -r requirements.txt
```

#### SMAC
If you have issues installing SMAC,
follow the instructions [here](https://automl.github.io/SMAC3/main/1_installation.html).


### Data
You need to pre-download all the data required by running `python datasets.py`.

Stores by default in a `./data` directory. Takes under 20 seconds to download and extract.

### Tips

All code we provide does consider validation and training sets.
You will have to implement a method to use the test set yourself.

#### `multi_fidelity_template.py`
* Example of how to use SMAC with multi-fidelity optimization.
* The example uses image size as the fidelity.
* To get quick results, you can lower the image size to 4x4 for a quick debug signal if you like.
However, make sure when comparing to any baseline to always use the maximum fidelity of 32x32.
* The configsapce that we used to get the baseline performance is in default_configspace.json.

#### `meta_learning_template.py`
* Example of how to run SMAC.
* Provides hints for how to extend SMAC to warmstart with meta-learning.
* Provides code to read and parse the meta-data.

#### `multi_objective_template.py`
* Example of how to run SMAC with multiple objectives and scalarization weights.
* A default configuration that was generated from this example code, for an undisclosed budget, is the default setting of the provided search space `multi_objective/configuration_space.json`.
* Provides code to read and parse the meta-data.
* Note that there is no constraint on the usage of different objectives especially when its justified use (or not) dominates the given degault configuraiotion on the test set.

#### Plotting
* We do not provide plotting scripts for the examples.
  You are allowed to use everything you already know from the lecture.
  We recommend to use 'matplotlib' or 'seaborn' for plotting.
* To get an example of how to plot SMAC data (which we used in the example code), you can take a look at
the [SMAC3 documentation](https://automl.github.io/SMAC3).
  An example similar to our multi-fidelity example can be found [here](https://automl.github.io/SMAC3/v2.0.1/examples/2_multi_fidelity/1_mlp_epochs.html).
* You are free to implement and use any other plotting scripts.
