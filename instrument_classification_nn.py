import numpy as np
import json
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras

dataset_path="data_10.json"

def load_data(dataset_path):
    with open(dataset_path,"r") as fp:
        data=json.load(fp)
    # Converting it into numpy array
    inputs=np.array(data["mfcc"])
    targets=np.array(data["labels"])
    


if __name__ == "__main__":
    # Load Dataset
    inputs,targets=load_data(dataset_path)

    # Split Dataset into train test sets

    inputs_train,inputs_test, targets_train,targets_test=train_test_split(inputs,targets,test_size=0.3)

    # Build the Neural Network Infrastructure

    model=keras.Sequential([
        
    ])
