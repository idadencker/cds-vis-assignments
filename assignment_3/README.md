# Transfer learning 


## Introduction
This program contains scripts that will load and finetune the VGG16 model to test how well the models perform on predicting the class of different scanned documents of the ```Tobacco3482``` dataset. 
The VGG16 model is a convolutional neural network model pretrained on the ImageNet dataset which contains more than 14 million annotated images of different categories. VGG16 is considered to be one of the best computer vision models to date. 
Training CNNs from scratch is exteremly time-consuming and computationally heavy. When doing transfer learning, we can load a pretrained model and finetune it on new data to make it more task-specific. 
However one problematic apsect of CNNs are their tendency to overfit when training on small datasets. To account for such issue, batch normalization and data augmentation (DA) is applied to secure robustness. 
The program contains 3 different py scripts: 1 that finetune the VGG16 model withouth Batch Normalization (a baseline model), 1 that finetune the VGG16 model with Batch Normalization and 1 that finetune the VGG16 model with data augmentation. For each model an optimizer must be specified as either 'adam' or 'sgd'. A loss curve plot and a classification report for the model is saved in the out folder. 


## Data
The dataset is the ```Tobacco3482``` dataset which consists of tobacco-related document images belonging to 10 classes such as letter, form, email, resume, memo, etc. The dataset has 3482 images. The dataset can be found and downloaded [here](https://www.kaggle.com/datasets/patrickaudriaz/tobacco3482jpg?resource=download). 


## Repository overview 
The repository consists of:
- 1 README.md file
- 2 bash scripts
- 1 requirenments file
- in folder for holding the input data
- out folder for holding the saved results
- src folder containing the 3 scripts for implementing transfer learning


## Reproducibility 
To make the program work do the following:

1) clone the repository 
```python
$ git clone "URL HERE"
```
2) download the dataset and place the Tobacco3482-jpg folder including the 10 subfolders in 'in'
3) In a terminal set your directory:
```python
$ cd assignment_3
```
4) To create a virtual environment run:
```python
$ source setup.sh
```
5) In run.sh uncomment the script wished to be executed
6) in the terminal type what optimizer you want "adam" or "sgd" by running:
```python
$ source run.sh -o "adam"
```
A loss curve plot and a classification report will be saved the the out folder 


## Summary and discussion
From looking at the performance of the models we see that the model implementing batch normalization using the adam optimizer performs best with an accuracy score of 74%. Inspecting the loss curve for the model we see that loss for both test and validation starts to flatten out for every epoch which is good. However, we also see signs of overfitting since the validation set has a higher loss and a lower accuracy. The implimentation of the sgd optimizer is less prone to overfitting, however on the expence of slightly lower accuracy. 
Multiple things can be implemented to prevent overfitting e.g. data augmentation. However the performance of the DA models don't achieve higher accuracy, though from looking at the plots they seem to have tackled the overfitting better. Altering images by horizontal flips and 90 degress rotations might confuse the model too much, perhaps since the images allready are not of the best quality and therefore more sensitive to DA. 
Epochs is set to be 10 and from looking at the plots it seems sufficient for most models, however raising the number of epochs could improve performance on some of the models. Batch_size is set to 32 to help avoid overfitting on the small dataset and to reduce computational expences.
It should always be considered that the pretrained VGG16 model is trained on certain images in certain categories. If e.g. scanned documents of say email, letter etc. is not included, the model will perform poorly even after finetuning. 