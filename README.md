# devCellPy

devCellPy is a Python package designed for hierarchical multilayered classification of cells based on single-cell RNA-sequencing (scRNA-seq). It implements the machine learning algorithm Extreme Gradient Boost (XGBoost) (Chen and Guestrin, 2016) to automatically predict cell identities across complex permutations of layers and sublayers of annotation.

Given devCellPy's highly customizable classification scheme, users can input the annotation hierarchy of their scRNA-seq datasets into devCellPy to guide the automatic classification and prediction of cells according to the provided hierarchy. devCellPy allows users to designate any identity at each layer of classification and is not constrained by cell type——for example, assigning timepoint as one of the annotation layers allows for cell identity predictions at that layer to be conditioned on the age of the cells. In addition to hierarchical cell classification, devCellPy implements the SHapley Additive exPlanations (SHAP) package (Lundberg etal, 2020), which provides the user with interpretability methods for the model and determines the positive and negative gene predictors of cell identities across all annotation layers.

For more information please refer to our publication at: 

Galdos, F.X., Xu, S., et al. devCellPy is a machine learning-enabled pipeline for automated annotation of complex multilayered single-cell transcriptomic data. Nat Commun 13, 5271 (2022). https://doi.org/10.1038/s41467-022-33045-x

To download the compiled Seurat object for the murine cardiac developmental atlas reported in our paper, please visit the following Zenodo link: https://zenodo.org/record/7183939#.Y0TuEC-B3T8 


# Documentation
Full tutorial of devCellPy usage is available under:
https://github.com/devCellPy-Team/devCellPy/tree/main/Tutorial

# System Requirements
## Hardware requirements
`devcellpy` hardware requirements vary depending on the modes used by the user.

### Train Mode Requirements:
Computer running with a minimum of 64 GB of RAM is recommended to train large scale datasets. We recommend users consider using a high performance computing cluster for training on large datasets that require greater than 64 GB. 

### Feature Ranking Mode Requirements: 
Feature Ranking mode of devCellPy requires a minimum of 64 GB of RAM to run. We recommend users consider a high performance computing cluster for conducting feature ranking on large datasets. 

### Prediction Model Requirements:
Prediction of cell types using an already trained devCellPy prediction model can be run on a standard computer with a minimum of 8 GB RAM. 

# Software requirements
devCellPy has been formatted into a wrapper function that can be easily installed through pip and run through the command line of the Terminal.
NOTE: All Python and XGBoost versions must remain the same throughout usage of all training, predicting, and feature ranking options. Ex) If Python 3.7 is used to train a dataset, Python 3.7 must be used to predict a query dataset using the trained dataset.

### OS Requirements
This package is supported for *macOS* and *Linux*. The package has been tested on the following systems:
+ macOS Intel: Big Sur (11.6)
+ macOS M1: Ventura (13.1)
+ Linux: CentOS-7 Version 7.9.2009

### Python Dependencies
`devcellpy` depends on the packages found in requirements.txt.

# Installation Guide:
devCellPy has been formatted into a wrapper function that can be easily installed through pip and run through the command line of the Terminal or Command Prompt.
NOTE: All Python and XGBoost versions must remain the same throughout usage of all training, predicting, and feature ranking options. Ex) If Python 3.7 is used to train a dataset, Python 3.7 must be used to predict a query dataset using the trained dataset.
devCellPy should install within a few minutes on a "standard" computer supporting in-memory operations. 

### Install from PyPi
```
pip3 install devcellpy
```

# License
MIT License

Copyright (c) 2021 Stanford University

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
