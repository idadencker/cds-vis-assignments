import os
from tqdm import tqdm 
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



def extract_decade(filename):
    '''
    Extracts the decade using the filename
    '''
    parts = filename.split("-")
    year = int(parts[1])
    decade = (year // 10) * 10
    return decade



def get_num_faces(file_path):
    '''
    Calculates number of faces 
    '''
    mtcnn = MTCNN(keep_all=True)    
    resnet = InceptionResnetV1(pretrained='casia-webface').eval()
    img = Image.open(file_path)
    boxes, _ = mtcnn.detect(img)

    if boxes is not None:
        num_faces = boxes.shape[0]  
    else:
        num_faces = 0 
    
    return num_faces



def df_for_newspaper(perc_pages_with_faces_per_decade, faces_raw_count_per_decade, newspaper_name):
    '''
    Create DataFrame for each newspaper by merging 2 dictionaries 
    '''
    df = pd.DataFrame.from_dict(perc_pages_with_faces_per_decade, orient='index', columns=['Percentage'])
    df_2 = pd.DataFrame.from_dict(faces_raw_count_per_decade, orient='index', columns=['count'])
    done = pd.merge(df, df_2, left_index=True, right_index=True)
    done['newspaper'] = newspaper_name
    done.sort_index(axis=0, ascending=True, inplace=True)
    done.to_csv(f'out/{newspaper_name}_data.csv')



def loop_through_files(folderpath): 
    """ This function loops through all files and extract information """
    for newspaper_folder in sorted(os.listdir(folderpath)):
        """ Construct the path to the current newspaper """
        full_path = os.path.join(folderpath, newspaper_folder)

        if os.path.isdir(full_path):
            newspaper_name = newspaper_folder
            '''
            Initialize dictionaries for the current newspaper 
            '''
            pages_per_decade, faces_relative_count_per_decade, faces_raw_count_per_decade, perc_pages_with_faces_per_decade = {}, {}, {}, {}
            '''
            Loops through all files for the current newspaper
            '''
            for filename in tqdm(sorted(os.listdir(full_path)), desc=newspaper_name):
                '''
                Construct the path to the current file
                '''
                file_path = os.path.join(full_path, filename)
                if os.path.isfile(file_path) and filename.endswith(".jpg"):
                    '''
                    Extracts the decade from the filename and increment the count of pages for the current decade
                    '''
                    decade = extract_decade(filename)
                    pages_per_decade.setdefault(decade, 0) 
                    pages_per_decade[decade] += 1 
                    '''
                    Gets the number of faces and fill the dictionaries based on the number of faces
                    '''
                    num_faces = get_num_faces(file_path)
                    if num_faces > 0:
                        faces_relative_count_per_decade.setdefault(decade, 0)
                        faces_relative_count_per_decade[decade] += 1
                        faces_raw_count_per_decade.setdefault(decade, 0)
                        faces_raw_count_per_decade[decade] += num_faces

                    elif num_faces == 0: 
                        faces_relative_count_per_decade.setdefault(decade, 0)
                        faces_relative_count_per_decade[decade] += 0
                        faces_raw_count_per_decade.setdefault(decade, 0)
                        faces_raw_count_per_decade[decade] += 0
            '''
            Calculate the percentage of pages with faces for each decade using the dictionaries. Saves the results in a dataframe
            '''
            for decade, page_count in pages_per_decade.items():
                perc_pages_with_faces_per_decade[decade] = round((faces_relative_count_per_decade.get(decade, 0) / page_count) * 100, 2)

            df_for_newspaper(perc_pages_with_faces_per_decade, faces_raw_count_per_decade, newspaper_name)



def merged_df():
    csv_files = [f for f in os.listdir("out/") if f.endswith('.csv')]
    dfs = []
    for csv in csv_files:
        df = pd.read_csv(os.path.join("out", csv))
        dfs.append(df)
    final_df = pd.concat(dfs, ignore_index=True)
    final_df.rename(columns={'Unnamed: 0': 'decade'}, inplace=True)
    return final_df



def plotting(final_df):
    plt.figure(figsize=(15, 10))
    g = sns.relplot(data=final_df, kind="line", x="decade", y="Percentage", hue="newspaper")
    g.fig.subplots_adjust(top=.95)
    plt.xticks(rotation=45, fontsize=8)
    plt.title('Percentage of pages with faces per decade', fontsize=12)
    plt.savefig("out/plot_of_percentages.png")
    plt.show()



def main():
    folderpath = "in/images/"
    loop_through_files(folderpath)
    final_df = merged_df()
    plotting(final_df)



if __name__ == "__main__":
    main()
