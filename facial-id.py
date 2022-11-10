# Standard dependencies
import cv2
import os
import random
import numpy as np
from matplotlib import pyplot as plt

# Import tensorflow dependencies
from keras.models import Model
from keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
import tensorflow as tf

# Set GPU mem consumption growth
# gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)

# Set-up paths and make directories
POS_PATH = os.path.join('data', 'positive')
NEG_PATH = os.path.join('data', 'negative')
ANC_PATH = os.path.join('data', 'anchor')

try:
    os.makedirs(POS_PATH, exist_ok=True)
    os.makedirs(NEG_PATH, exist_ok=True)
    os.makedirs(ANC_PATH, exist_ok=True)
except OSError as error:
    print("Directory can not be created")

