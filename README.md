# devCellPy

devCellPy is a Python package designed for hierarchical multilayered classification of cells based on single-cell RNA-sequencing (scRNA-seq). It implements the machine learning algorithm Extreme Gradient Boost (XGBoost) (Chen and Guestrin, 2016) to automatically predict cell identities across complex permutations of layers and sublayers of annotation.

Given devCellPy's highly customizable classification scheme, users can input the annotation hierarchy of their scRNA-seq datasets into devCellPy to guide the automatic classification and prediction of cells according to the provided hierarchy. devCellPy allows users to designate any identity at each layer of classification and is not constrained by cell type——for example, assigning timepoint as one of the annotation layers allows for cell identity predictions at that layer to be conditioned on the age of the cells. In addition to hierarchical cell classification, DevCellPy implements the SHapley Additive exPlanations (SHAP) package (Lundberg etal, 2020), which provides the user with interpretability methods for the model and determines the positive and negative gene predictors of cell identities across all annotation layers.

We provide a comprehensive tutorial on devCellPy's installation and usage as well as overall concepts in its design in the tutorial folder of the devCellPy GitHub.



# Installation Guide:

### Install from PyPi
```
pip3 install devcellpy
```
