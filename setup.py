#!/usr/bin/env python3

from setuptools import find_packages, setup

setup(
    name = 'devCellPy',
    description = 'devCellPy -- hierarchical multilayered classification of cells based on scRNA-seq',
    long_description = """\
    devCellPy is a Python package designed for hierarchical
    multilayered classification of cells based on single-cell
    RNA-sequencing (scRNA-seq). It implements the machine
    learning algorithm Extreme Gradient Boost (XGBoost)
    (Chen and Guestrin, 2016) to automatically predict cell
    identities across complex permutations of layers and
    sublayers of annotation.""",  # wrap at col 60
    url = 'https://github.com/devCellPy-Team/devCellPy',
    version = '1.2.0',
    author = 'Sidra Xu',
    author_email = 'sidraxu@stanford.edu',
    maintainer = 'Sean Wu Lab',
    maintainer_email = 'smwu@stanford.edu',
    license = 'MIT',
    python_requires = '>=3.7',
    include_package_data = True,
    packages = find_packages(),
    install_requires = ['anndata>=0.9.1', 'matplotlib>=3.7.1', 'numpy==1.23.5', 'pandas>=1.5.3',
                        'scanpy>=1.9.3', 'scikit-learn>=1.2.2', 'seaborn>=0.12.2', 'shap>=0.41.0',
                        'umap-learn>=0.5.3', 'xgboost'],
    entry_points = {
        'console_scripts': [
            'devCellPy = devCellPy.__main__:main'
        ],
    }
)