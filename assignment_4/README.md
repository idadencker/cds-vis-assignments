# Detecting faces in historical newspapers


## Introduction
This program contains a script that will load images of scanned papers from three different Swiss newspapers to investigate the presence of human faces over the years of publication. Old newspapers are largely dominated by plain text and few illustrations, however, with the advancing technology of personal cameras that took place in the 20th century, more faces appeared in the newspapers. This program will use a pre-trained CNN model fine-tuned for face detection. Documentation for the model can be found [here](https://medium.com/%2540danushidk507/facenet-pytorch-pretrained-pytorch-face-detection-mtcnn-and-facial-recognition-b20af8771144).<br> The results are grouped by decade and 1 CSV for each newspaper is saved, showing per decade: the total count of faces and the percentage of pages that have at least 1 face on them. A plot is saved showing the latter for all 3 newspapers. The results are summarised and discussed.


## Data 
The dataset consists of scanned pages from 3 Swiss newspapers: the Journal de Genève (JDG, 1826-1994); the Gazette de Lausanne (GDL, 1804-1991); and the Impartial (IMP, 1881-2017). The dataset can be found and downloaded [here](https://zenodo.org/records/3706863). 


## Repository overview 
The repository consists of:
- 1 README.md file
- 2 bash scripts
- 1 requirements file
- in folder for storing input data
- out folder for holding the saved results
- src folder containing the script for counting faces


## Reproducibility 
To make the program work do the following:

1) Clone the repository 
```python
$ git clone https://github.com/idadencker/cds-vis-assignments.git
```
2) Download the images.zip file. Unzip it and place the images folder including the 3 subfolders in the 'in' folder.
3) In a terminal set your directory:
```python
$ cd assignment_4
```
4) To create a virtual environment run:
```python
$ source setup.sh
```
5) To run the script run:
```python
$ source run.sh 
```
A CSV for each newspaper and 1 plot for visualising the findings will be saved in the out folder 


## Summary and discussion
Results show a general increase in percentage for all three newspapers over the time course. JDG and GDL share the most similar pattern with a percentage between 0-10% for the years 1820-1880 continuously increasing to roughly 30% by 2000. <br>
Interestingly, GDL has a relatively high percentage at the start of publication in the 1790s but suddenly drops rapidly by the 1800s, which could be due to limited resources. A pattern which is not present for the other 2 newspapers. IMP deviates from the other papers by having a consistently higher percentage from the start of publication in the 1880s to a percentage of nearly 80% by 2000. Generally it can be said that all newspapers experience an increase in the presence of faces over the years of publication, though at different speeds. <br>
A final note to consider is the limitation of the face-detection model. The model produces predictions and cannot be taken as the ground truth. It is possible that the model failed to pick up on some present faces or misclassified something as being a human face, which is an important consideration to make when interpreting results.

