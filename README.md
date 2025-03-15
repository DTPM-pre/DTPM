# DTPM: Performance Prediction for Configurable Software Systems


## Installation and Setup

To run the project, first install the NumCpp library by following the instructions here: https://github.com/dpilger26/NumCpp/blob/master/docs/markdown/Installation.md

Then, update the include_dirs in setup.py to include the path to the NumCpp header files by replacing <NumCpp installation dir> with the correct path.

running python setup.py build_ext --inplace


## To train the model:

First, sample the data by running:
python select_represent_data.py

Next, start training the model by executing:

python train.py

## To test the model, run the following command:
python test.py