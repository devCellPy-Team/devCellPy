#!/usr/bin/env python3

import sys

try:
    from setuptools import find_packages, setup
except ImportError:
    print("""\
Error: The "setuptools" module, which is required for the
  installation of devCellPy, could not be found.
  You may install  it with `python -m pip install setuptools`
  or from a package called "python-setuptools" (or similar)
  using your system\'s package manager.
  Alternatively, install a release from PyPi with
  `python -m pip install docutils`.'
  If all this fails, try a "manual install".
  https://docutils.sourceforge.io/docs/dev/repository.html#install-manually
""")
    sys.exit(1)


def setup():
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
    entry_points = {
        'console_scripts': ['devCellPy=devCellPy.main:main'],
    },
    scripts = ['main.py']


if __name__ == '__main__':
    setup()