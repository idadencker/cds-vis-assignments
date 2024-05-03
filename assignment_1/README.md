# Building image search algorithms


## Introduction
This program contains scripts that will load images of 1360 different flowers to determine, based on features, what five other flowers are most similar to a chosen target image. One script uses the package OpenCV to extract a color histogram for each image and compare across all flowers to determine which flowers resemble the target image the most. Another script applies the pre-trained image recognition and classification model, VGG16, to extract features from the images and uses a nearest neighbor algorithm to calculate the cosine similarity between images. VGG16 is a convolutional neural network model pretrained on the ImageNet dataset, which contains more than 14 million annotated images of different categories. VGG16 is considered to be one of the best computer vision models to date.

For both image search methods, a CSV file containing the filename and distances for the target image and the five closest images is saved. Furthermore, a plot displaying the images and distances for both methods is saved. The results from both methods are summarized and discussed. Note that the target image chosen is simply the first example in the data: 'image_0001'. However, if desired, the target image can easily be replaced with any other image. For details, read the reproducibility section. 


## Data 
The dataset consists of 1360 images of flowers across 17 species and comes from the Visual Geometry Group at the University of Oxford. More details and downloads are available [here](https://www.robots.ox.ac.uk/~vgg/data/flowers/17/)


## Repository overview 
The repository consists of:
- 1 README.md file
- 2 bash scripts
- 1 requirenments file
- in folder for storing input data
- out folder for holding the saved results
- src folder containing the 2 scripts for implementing image search 


## Reproducibility 
To make the program work do the following:

1) clone the repository 
```python
$ git clone https://github.com/idadencker/cds-vis-assignments.git
```
2) download the dataset and place all the images in the 'in' folder
3) If you wish to change the target image: 
- in 'Color_hist_image_search.py' change 'filepath_target' to the filepath of the desired target image
- in 'VGG16_image_search.py' change the 0 in line 81: 'distances, indices = neighbors.kneighbors([feature_list[0]])' to any number between 0 and 1359, note that 0 refers to the first image 'image_0001', 1 refers to the second image 'image_0002' etc. 
4) In a terminal set your directory:
```python
$ cd assignment_1
```
5) To create a virtual environment run:
```python
$ source setup.sh
```
6) To run the 2 scripts and save results run: 
```python
$ source run.sh
```
2 plots and 2 CSV files will be saved the the out folder 


## Summary and discussion
When comparing the two approaches for conducting an image search to determine the most similar images, very different results are displayed. For the same target image 'image_0001', the two approaches do not agree on a single image. None of the five different images determined by the VGG16 application are present in the images for the color histogram application. From visually inspecting the two plots, it's evident that the VGG16 application has arrived at the most resembling images, and hence is the most accurate model. 
Color histograms only capture color distribution without considering important components like semantics, and hence a lot of information and detail is lost. Images are deemed similar if the distribution of colors across the image assembles, though the images might represent completely different scenes and semantics. However, the highly complex CNN model VGG16 is capable of performing high-level reasoning and can, due to its architecture, capture edges, textures, and shapes, and thus more complex and abstract features. 
Based on the findings, it can be concluded that the VGG16 application does the most accurate job in finding the five most visually similar images. These results are, however, solely based on an analysis of a single image, though the general pattern will most likely still apply if other images are analyzed.