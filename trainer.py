from caer import preprocess, saveNumpy
import numpy as np
import os

DIR = r'F:\Dogs and Cats\Train'
classes = [i for i in os.listdir(DIR)]
IMG_SIZE = 224

# Create the train data
train = preprocess(DIR, classes, 'train.npz', IMG_SIZE)

print(f'Train shape =  {train.shape}')