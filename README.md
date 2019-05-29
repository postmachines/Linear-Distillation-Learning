# Linear Distillation Learning

<img width="1253" alt="comparison" src="https://user-images.githubusercontent.com/23639048/58211371-6d5a2700-7cf4-11e9-823e-64c335b61203.png">

Repository contains experiments considering _Linear Distillation Learning_, concepts of appliying idea of knowledge distillation in linear setup.

## Structure

Repository is organized as follows
* `data` folder serving as default location for datasets
* `experiments` conducted experiments in a free way
* `ldl` library with stable scripts and functions
* `notebooks` in-between jupyter notebooks
* `results` folder for results dumping in .csv format
* `scripts` some stable scripts connected with `ldl` library
* `tests` minimal tests for `scripts`

`ldl` folder contains mini lib required to reproduce stable experiments. To install run
* `python setup.py install`

## Tests

For running tests:
* `python -m unittest tests/test_bidir_mnist.py`
* `python -m unittest tests/test_omd_mnist.py`
