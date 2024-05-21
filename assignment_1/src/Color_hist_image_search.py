import sys
import os
import cv2 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm



def calculate_target_image(filepath_target):
    '''
    Load in image, extract color histogram and normalise the image
    '''
    image_f1 = cv2.imread(filepath_target)
    hist_f1 = cv2.calcHist([image_f1], [0,1,2], None, [255, 255, 255],  [0,256, 0,256, 0,256])
    norm_hist_f1 = cv2.normalize(hist_f1, hist_f1, 0, 1.0, cv2.NORM_MINMAX)
    
    return norm_hist_f1



def loop_through_files(filepath_all, norm_hist_f1):
    '''
    Define dataframe to put the distances and filenames in
    '''
    distances_all = pd.DataFrame(columns= ("Filename", "Distance"))

    print("Reading images using color histogram:")
    for file in tqdm(sorted(os.listdir(filepath_all))):
            '''
            Creates a filepath and a filename for each flower and reads in the image
            '''
            filepath_fx = os.path.join(filepath_all, file)
            filename_fx = file.split(".jpg") [0] 
            image_fx = cv2.imread(filepath_fx) 
            '''
            Extract the color histogram and normalizes it
            '''
            hist_fx = cv2.calcHist([image_fx], channels = [0,1,2], mask = None, histSize = [255,255,255], ranges = [0,256, 0,256,0,256]) 
            norm_hist_fx = cv2.normalize(hist_fx, hist_fx, 0, 1.0, cv2.NORM_MINMAX)  
            '''
            Compare extracted histogram to the target histogram. Saves the filename and the distance to the target in a row and put in in the dataframe
            '''
            dist_fx = round(cv2.compareHist(norm_hist_f1, norm_hist_fx, cv2.HISTCMP_CHISQR),2) 
            row_fx = [filename_fx, dist_fx]
            distances_all.loc[len(distances_all)] = row_fx

    return distances_all



def save_distances(distances_all):
    '''
    Saves the distances for the target image + closest 5 images to the out folder
    '''
    final_df=(distances_all.nsmallest(6, ["Distance"])) 
    final_df.to_csv("out/distance_metrics_color_hist.csv", index=False)



def plot(distances_all, filepath_target, filepath_all):
    '''
    Creates a list of the images to plot
    '''
    closest_images_filenames = distances_all.nsmallest(6, ["Distance"])["Filename"].tolist()
    images_to_plot = [filepath_target] + [os.path.join(filepath_all, filename + ".jpg") for filename in closest_images_filenames[1:]]

    fig, axs = plt.subplots(2, 3, figsize=(15, 15))
    fig.suptitle("Distances using color histogram", fontsize=20)
    '''
    Loops through the images and plots them with their distance to the target image
    '''
    for i, image_path in enumerate(images_to_plot):
        image = cv2.imread(image_path)
        ax = axs[i // 3, i % 3]  
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) 
        ax.axis("off")

        filename = os.path.splitext(os.path.basename(image_path))[0]
        distance = distances_all.loc[distances_all['Filename'] == filename, 'Distance'].iloc[0]
        ax.set_title(f'Distance: {distance}', fontsize=10)

    plt.tight_layout()
    plt.savefig("out/flowers_color_hist") 



def main():
    filepath_target = os.path.join("in", "image_0001.jpg")
    norm_hist_f1 = calculate_target_image(filepath_target)
    filepath_all = os.path.join("in")
    distances_all = loop_through_files(filepath_all, norm_hist_f1)
    save_distances(distances_all)
    plot(distances_all, filepath_target, filepath_all)
    


if __name__ == "__main__":
    main()