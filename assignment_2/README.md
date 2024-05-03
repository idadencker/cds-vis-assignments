# Image classification


## Introduction
This program contains scripts that will load, preprocess and classify the multiclass cifar10 dataset. One script implements a logistic regression classifier, including hyperparameter tuning of defined hyperparameters to determine the model with the greatest accuracy. The other script implements an architecturally more complex model, a multilayer perceptron (MLP) model. MLP is a feedforward artificial neural network, consisting of fully connected neurons, capable of learning complex patterns and relationships within data. 
Model performance of both models are evaluated and scoring metrics are summerised in the saved classification report alongside with a loss curve plot for the MLP model.
The results from both methods are summarized and discussed. 


## Data 
The data used for classification is the CIFAR-10 dataset which consist of 6000 32x32 colour images for 10 different classes, totalling 60.000 images. The dataset can be loaded using the cifar10.load_data function, and will produce a train/test split of 50.000 training images and 10.000 test images. More details on the data are available [here](https://www.cs.toronto.edu/~kriz/cifar.html)


## Repository overview 
The repository consists of:
- 1 README.md file
- 2 bash scripts
- 1 requirenments file
- out folder for holding the saved results
- src folder containing the 2 scripts for conduction image classification 


## Reproducibility 
To make the program work do the following:

1) clone the repository 
```python
$ git clone "URL HERE"
```
2) In a terminal set your directory:
```python
$ cd assignment_2
```
3) To create a virtual environment run:
```python
$ source setup.sh
```
4) To run the 2 scripts and save results run: 
```python
$ source run.sh
```
2 classification reports and a loss curve plot for the MLP model will be saved the the out folder 

Please note that the scripts may take some time to execute. You can track information on hyperparameter tuning for logistic regression model and the iterations for the MLP model in the terminal output.


## Summary and discussion
From looking at the classification reports, the MLP model emerges as the best-performing model, achieving an accuracy score of 39%, compared to the logistic regression model with 31% accuracy. Furthermore, it's evident that some categories are easier to recognize than others for both models. For example, the category 'cat' proves particularly difficult for both models. Although neither model achieves perfect performance, both exceed the chance level by 10%.
An influence on the model's performance is the hyperparameters used for training. Grid search hyperparameter tuning was applied to the logistic regression model. As it's a multiclass model, 'sag', 'saga', and 'newton-cg' were chosen as possible solvers. Furthermore, 3 different regularization strengths and 2 different penalties were defined, resulting in 18 possible fits for the model. The best-performing model, optimized for accuracy, uses the saga solver, l1 penalty, and a regularization strength of 0.1.
While adding more parameters for hyperparameter tuning could potentially improve performance, it comes at the cost of computational resources and time. However, it's questionable whether performance will substantially improve by complicating the hyperparameter tuning process.
For the MLP model, a logistic sigmoid activation function for the hidden layer is applied to predict the probability as an output. The number of hidden layers is set to 100 to account for the dataset containing labels of 10 different classes. When inspecting the loss curve, the model reaches maximum performance after about 500 iterations. Similarly, hyperparameter tuning could be applied to the MLP model, but given that the baseline model with no hyperparameter tuning is already time-costly in execution, hyperparameter tuning is disregarded.