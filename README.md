# miRBind

miRBind is a machine learning method based on CNN (Convolutional Neural Network). It learns the rules of miRNA:target binding and provides a probability for the potential binding of a pair of given miRNA and target molecule.

## Installation

Using Git:

git clone https://github.com/ML-Bioinfo-CEITEC/miRBind.git, or

git clone git@gitlab.com:RBP_Bioinformatics/miRBind.git


### Prerequisities

Penguinn is implemented in python using Keras and Tensorflow backend.

Required:

* python, recommended version 3.7
    * Keras 2.7.0
    * tensorflow 2.7.0
    * Biopython
    * numpy
    
    
### Installing

(1) create a virtual environment:

python -m venv venv

(2) activate it and install the necessary libraries.

source venv/bin/activate
pip install -r requirements.txt


### Prediction

The default model is trained on human Ago1 CLASH dataset with ratio 1:1 of positive:negative samples.
To run the model:

1) cd path/to/miRBind/directory
2) chmod +x penguinn.py
   #if you are not actively sourcing from the previously created virtualenv:
   source venv/bin/activate
3) #run the prediction
   ./mirbind.py --input <input_fasta_file> --output <output_file> --model <path_to_model.h5>

### Web application

....

### Contact information

CEITEC MU, RBP Bioinformatics - Panagiotis Alexiou, https://www.ceitec.eu/rbp-bioinformatics-panagiotis-alexiou/rg281
