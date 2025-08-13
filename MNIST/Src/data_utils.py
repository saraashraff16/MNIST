import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from tensorflow.keras.utils import to_categorical
import warnings
warnings.filterwarnings('ignore')


def load_data():
    digits=load_digits()
    x=digits.data
    y=digits.target
    return x, y, digits

def explore_data(digits,x,y):
    print(f"X Shape = {x.shape}")
    print(f"Y Shape = {y.shape}")
    print (f"Data DESCR {digits.DESCR}")

    fig, axes = plt.subplots(2, 5, figsize=(8, 4))
    for i, ax in enumerate(axes.flat):
        ax.imshow(digits.images[i], cmap='gray')
        ax.set_title(f"Label: {digits.target[i]}")
        ax.axis('off')
    plt.show()

def scale_data(x):
    scaler=MinMaxScaler()
    x_scaled=scaler.fit_transform(x)
    return x_scaled 

def split_data(x,y):
    x_train ,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
    return x_train ,x_test,y_train,y_test

def data_for_cnn(digits):

    # form the data
    X=digits.images
    Y=digits.target

    # normalize and reshape X
    X=X.reshape(-1,8,8,1).astype('float32')/16.0

    #convert y by one-hot 
    Y_encoded=to_categorical(Y,num_classes=10)
    
    return X , Y_encoded