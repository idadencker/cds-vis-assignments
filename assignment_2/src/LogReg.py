import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import tensorflow
from tensorflow.keras.datasets import cifar10
from tqdm import tqdm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report



def preprocessing(X_train, X_test):
    '''
    This function grayscales, normalizes and reshapes the data
    '''
    done_list_train = []
    print("preprocessing X_train")
    for picture in tqdm(X_train):
        greyscaled= cv2.cvtColor(picture, cv2.COLOR_BGR2GRAY) 
        normalized = greyscaled/255.0 
        done_list_train.append(normalized) 
    X_train=np.array(done_list_train) 
    X_train = X_train.reshape(-1,1024)

    done_list_test = []
    print("preprocessing X_test")
    for picture in tqdm(X_test):
        greyscaled= cv2.cvtColor(picture, cv2.COLOR_BGR2GRAY) 
        normalized = greyscaled/255.0 
        done_list_test.append(normalized) 
    X_test=np.array(done_list_test) 
    X_test= X_test.reshape(-1,1024) 

    return X_train, X_test 



def fitting_model(X_train, y_train):
    """ Defining what parameters will be optimsed """
    grid = {
        "C": [1.0, 0.1, 0.01], 
        "penalty": ["l1", "l2"],
        "solver": ['newton-cg','sag','saga']
    }
    logreg = LogisticRegression(multi_class='multinomial')
    logreg_cv = GridSearchCV(logreg, grid, scoring='accuracy', cv=10, n_jobs=-1, verbose=2)
    
    print("Tuning hyperparameters...")
    model = logreg_cv.fit(X_train, y_train)
    
    """ Print parameters for each combination tried """
    print("\nParameter combinations tried:")
    means = logreg_cv.cv_results_['mean_test_score']
    stds = logreg_cv.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, logreg_cv.cv_results_['params']):
        print(f"Accuracy: {mean:.4f} (±{std:.4f}) for {params}")

    """ Print the hyperparamerts and accuracy for the best model """
    print("Tuned hyperparameters (best parameters):", logreg_cv.best_params_)
    print("Accuracy:", logreg_cv.best_score_)

    return model



def evaluate(model, X_test, y_test):
    '''
    This function makes predictions, creates classification report and save it in the out folder
    '''
    y_pred = model.predict(X_test)  
    classifier_metrics = metrics.classification_report(y_test, y_pred, target_names =["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"])

    filepath = "out/log_classification_report.txt"

    with open(filepath, 'w') as file:
        file.write(classifier_metrics)



def main():
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    X_train, X_test = preprocessing(X_train, X_test)
    model=  fitting_model(X_train, y_train)
    evaluate(model, X_test, y_test)



if __name__ == "__main__":
    main()