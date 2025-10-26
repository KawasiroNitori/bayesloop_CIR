[![bayesloop](https://raw.githubusercontent.com/christophmark/bayesloop/master/docs/images/logo_400x100px.png)](http://bayesloop.com)

[![Build status](https://github.com/christophmark/bayesloop/workflows/Tests/badge.svg?branch=master)](https://github.com/christophmark/bayesloop/actions/workflows/test.yml)
[![Documentation status](https://readthedocs.org/projects/bayesloop/badge/?version=latest)](http://docs.bayesloop.com) 
[![Coverage Status](https://codecov.io/gh/christophmark/bayesloop/branch/master/graph/badge.svg?token=637W4M2RCE)](https://codecov.io/gh/christophmark/bayesloop)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/41474112.svg)](https://zenodo.org/badge/latestdoi/41474112)

Forked version of Bayesloop package, especially for diffusing diffusivity with CIR process.

## Installation
I checked it works with python ver=3.12.
But the installation should be done by
```
pip install git+https://github.com/christophmark/bayesloop
```
not by pure pip install.


## Dependencies
*bayesloop* is tested on Python 3.8. It depends on NumPy, SciPy, SymPy, matplotlib, tqdm and cloudpickle. All except the last two are already included in the [Anaconda distribution](https://www.continuum.io/downloads) of Python. Windows users may also take advantage of pre-compiled binaries for all dependencies, which can be found at [Christoph Gohlke's page](http://www.lfd.uci.edu/~gohlke/pythonlibs/).

## Optional dependencies
*bayesloop* supports multiprocessing for computationally expensive analyses, based on the [pathos](https://github.com/uqfoundation/pathos) module. The latest version can be obtained directly from GitHub using pip (requires git):
```
pip install git+https://github.com/uqfoundation/pathos
```
**Note**: Windows users need to install a C compiler *before* installing pathos. One possible solution for 64bit systems is to install [Microsoft Visual C++ 2008 SP1 Redistributable Package (x64)](http://www.microsoft.com/en-us/download/confirmation.aspx?id=2092) and [Microsoft Visual C++ Compiler for Python 2.7](http://www.microsoft.com/en-us/download/details.aspx?id=44266).

## License
[The MIT License (MIT)](https://github.com/christophmark/bayesloop/blob/master/LICENSE)

If you have any further questions, suggestions or comments, do not hesitate to contact me: &#098;&#097;&#121;&#101;&#115;&#108;&#111;&#111;&#112;&#064;&#103;&#109;&#097;&#105;&#108;&#046;&#099;&#111;&#109;
