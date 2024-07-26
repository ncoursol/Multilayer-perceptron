# Multilayer-perceptron

## Introduction
This subject aims to create a multilayer perceptron, in order to predict whether a cancer is malignant or benign on a dataset of breast cancer diagnosis in the Wisconsin.

42 subject: https://cdn.intra.42.fr/pdf/pdf/112647/en.subject.pdf

The subject focuses on implementing a first approach to artificial neural networks via the creation of a multilayer perceptron, while tackling notions such as derivative manipulation, linear algebra and matrix calculations.

## Datas
The features of the dataset describe the characteristics of a cell nucleus of breast mass extracted with fine-needle aspiration.

The records are provides from [UC Irvine - Machine Learning Repository](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic).

This database is also available through the UW CS ftp server:
`ftp ftp.cs.wisc.edu`
`cd math-prog/cpo-dataset/machine-learn/WDBC/`

This data set is a csv file of 32 columns, the column
diagnosis being the label we want to learn given all the other features of an example,
it can be either the value M or B (for malignant or benign):

1) ID number
2) Diagnosis (M = malignant, B = benign)
3-32) Ten real-valued features are computed for each cell nucleus:
	a) radius (mean of distances from center to points on the perimeter)
	b) texture (standard deviation of gray-scale values)
	c) perimeter
	d) area
	e) smoothness (local variation in radius lengths)
	f) compactness (perimeter^2 / area - 1.0)
	g) concavity (severity of concave portions of the contour)
	h) concave points (number of concave portions of the contour)
	i) symmetry 
	j) fractal dimension ("coastline approximation" - 1)

The mean, standard error, and "worst" or largest (mean of the three
largest values) of these features were computed for each image,
resulting in 30 features.  For instance, field 3 is Mean Radius, field
13 is Radius SE, field 23 is Worst Radius.

All feature values are recoded with four significant digits.

> [!NOTE]
> You can find all this information in the [wdbc.names](wdbc.names) file and in the [UC Irvine - Machine Learning Repository](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic).

## Run program
The program is write and run using python3.

`sudo apt update`

`apt install python3-pip`

Install all the requirements:

`pip install -r requirements.txt`

### Use case

The multilayer perceptron needs to be trained before it can make predictions.
Training will create a 'weights.npy' file containing the model's training weights, which when run in the prediction program will create a file named results.csv representing a table of predictions by patient.

Run training on the entire dataset:
  `python3 train.py data.csv`

Run prediction on the entire dataset:
  `python3 predict.py data.csv`

Run training with 100 iterations, 3 hidden layers of 20 neurons, with mini-batch and verbose enable:
  `python3 train.py -i 100 -l 20 20 20 -b -v data.csv`
  [test1](pics/test.png)

Run training with 1000 iterations, 2 hidden layers of 15 neurons and verbose enable:
  `python3 train.py -i 1000 -l 15 15 -v data.csv`
  [test2](pics/test2.png)

Run training with 20 iterations, 2 hidden layers of 15 neurons, with mini-batch and verbose enable:
  `python3 train.py -i 20 -l 15 15 -v -b data.csv`
  [test3](pics/test3.png)

  [pred](pics/predict.png)

