# miRBind

miRBind is a machine learning method based on CNN (Convolutional Neural Network). It learns the rules of miRNA:target binding and provides a probability for the potential binding of a pair of given miRNA and target sequence.

### Web application

The user-friendly miRBind web application for performing predictions https://ml-bioinfo-ceitec.github.io/miRBind/

## Installation

Using Git:
```
git clone https://github.com/ML-Bioinfo-CEITEC/miRBind.git
```
```
git clone git@gitlab.com:RBP_Bioinformatics/miRBind.git
```

### Prerequisites

mRBind is implemented in python using Keras and Tensorflow backend.

Required:

* python, recommended version 3.7
    * Keras 2.7.0
    * tensorflow 2.7.0
    * pandas
    * numpy
    
    
### Installing

```
#create a virtual environment:

python -m venv venv

#activate it and install the necessary libraries.

source venv/bin/activate
pip install -r requirements.txt
```

### Prediction

Required input is a tsv file with multiple potential miRNA - target pairs consisting of first column containing miRNA sequence (20 bp long) and second column containing target sequence (50 bp long).
To run the model:

```
cd path/to/miRBind/directory
chmod +x mirbind.py
#if you are not actively sourcing from the previously created virtualenv:
source venv/bin/activate
#run the prediction
./mirbind.py --input <input_file> --output <output_file>
```


### Contact information

CEITEC MU, RBP Bioinformatics - Panagiotis Alexiou, https://www.ceitec.eu/rbp-bioinformatics-panagiotis-alexiou/rg281
