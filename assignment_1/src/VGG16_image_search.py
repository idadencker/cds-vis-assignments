import os
import sys
import cv2 
import numpy as np
from numpy.linalg import norm
from tqdm import tqdm
import tensorflow_hub as hub
from tensorflow.keras.preprocessing.image import (load_img, 
                                                  img_to_array)
from tensorflow.keras.applications.vgg16 import (VGG16, 
                                                 preprocess_input)
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import NearestNeighbors



def define_model():
    '''
    The pretrained VGG16 model is specidfied. The classification layers of the model are not loaded. 
    Global average pooling will be applied to the output of the last convolutional block.
    '''
    model = VGG16(weights='imagenet', 
                include_top=False,
                pooling='avg',
                input_shape=(224, 224, 3))
    
    return model 



def extract_features(img_path, model):
    '''
    Extract features from image data using pretrained model:
    Define input image shape as 224, 224, 3 and load the image 
    '''
    input_shape = (224, 224, 3)
    img = load_img(img_path, target_size=(input_shape[0], 
                                          input_shape[1]))
    '''
    Convert image to array and expand to fit dimensions
    '''
    img_array = img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    '''
    A preprocessing pipeline is applied. The predict fnction is used to crete feature representation
    '''
    preprocessed_img = preprocess_input(expanded_img_array)
    features = model.predict(preprocessed_img, verbose=False)
    '''
    flatten and normalise features
    '''
    flattened_features = features.flatten()
    normalized_features = flattened_features / norm(features)

    return flattened_features



def loop_through_files(filepath_all, model):
    '''
    Get filenames for the images and empty list that will hold features 
    '''                        
    filenames = [filepath_all + "/" + name for name in sorted(os.listdir(filepath_all))]
    feature_list = []

    '''
    Iterate over all files and caculate the cosine similarity
    '''
    print("Reading images using VGG16:")
    for i in tqdm(range(len(filenames))):
        feature_list.append(extract_features(filenames[i], model))

    neighbors = NearestNeighbors(n_neighbors=10, 
                                algorithm='brute', 
                                metric='cosine').fit(feature_list)
    '''
    Distances and indices are calculated based on the first image in the feature_list (image_0001).
    Note that 0 can be changed to any number from 0 to 1359, if cosine similary is wished to be calculated on another image
    '''
    distances, indices = neighbors.kneighbors([feature_list[0]]) 
    
    return distances, indices



def save_distances(distances,indices):
    '''
    Creating lists for holding indices and distances
    '''
    idxs = []
    dist_values = []

    '''
    Get the indecies and distances for the target image and 5 closest images to put in a dataframe 
    '''
    for i in range(0, 6):  
        idxs.append(indices[0][i])
        dist_values.append(round(distances[0][i], 4))
    data = {'Index': idxs, 'Distance': dist_values}
    df = pd.DataFrame(data)

    '''
    Since indecies ranges from 0-1359 while the name of the images are 0001-1360, some steps are implemented to make the 2 distance metrics files easily comparable:

    1) 1 is added to all rows of Index column (since 0 is e.g the first image and filname is 0001.jpg)
    2) The Index column is filled based on how many digist it contain. If the number in the Index column is 1-digit 3 zeroes are added before the number, for 2-digit 2 zeroes are added etc.
    3) Aditionally 'image_' is added before all now 4-digit numbers.
    4) Lastly Index is named to Filename, as this is now more appropriate. 
    '''
    df['Index'] += 1
    df['Index'] = df['Index'].astype(str).apply(lambda x: '000' + x if len(x) == 1 else '00' + x if len(x) == 2 else '0' + x if len(x) == 3 else x)
    df['Index'] = 'image_' + df['Index'].astype(str)
    df = df.rename(columns={'Index': 'Filename'})
    df.to_csv('out/distance_metrics_VGG16.csv', index=False)
    
    return df 



def plot(df, filepath_target, filepath_all):
    closest_images_filenames = df["Filename"].tolist()

    images_to_plot = [filepath_target] + [os.path.join(filepath_all, filename + ".jpg") for filename in closest_images_filenames[1:]]

    fig, axs = plt.subplots(2, 3, figsize=(15, 15))
    fig.suptitle("Distances using VGG16", fontsize=20)
    '''
    Loops through the images and plots them with their distance to the target image
    '''
    for i, image_path in enumerate(images_to_plot):
        image = cv2.imread(image_path)
        ax = axs[i // 3, i % 3]  
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) 
        ax.axis("off")

        filename = os.path.splitext(os.path.basename(image_path))[0]
        distance = df.loc[df['Filename'] == filename, 'Distance'].iloc[0]
        ax.set_title(f'Distance: {distance}', fontsize=10)

    plt.tight_layout()
    plt.savefig("out/flowers_VGG16")



def main():
    filepath_target = os.path.join("..",
                        "..",
                        "..",
                        "..",
                        "cds-vis-data",
                        "flowers", 
                        "image_0001.jpg")
    model = define_model()
    filepath_all = os.path.join("..", 
                            "..",
                            "..",
                            "..",
                            "cds-vis-data",
                            "flowers", )
    distances, indices = loop_through_files(filepath_all, model)
    df = save_distances(distances,indices)
    plot(df, filepath_target, filepath_all)



if __name__ == "__main__":
    main()