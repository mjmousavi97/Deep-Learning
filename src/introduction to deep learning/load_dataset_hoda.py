import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import io

def load_hoda(data_path, train_size, test_size, img_size):
    dataset = io.loadmat(data_path)

    x_train = np.squeeze(dataset['Data'][:train_size, :])
    y_train = np.squeeze(dataset['labels'][:train_size])
    x_test = np.squeeze(dataset['Data'][train_size:(train_size+test_size), :])
    y_test = np.squeeze(dataset['labels'][train_size:(train_size+test_size)])
    
    x_train_5by5 = np.array([cv2.resize(img, dsize=img_size) for img in x_train])
    x_test_5by5 = np.array([cv2.resize(img, dsize=img_size) for img in x_test])

    x_train_new = np.reshape(x_train_5by5, [-1, img_size[0]*img_size[1]])
    x_test_new = np.reshape(x_test_5by5, [-1, img_size[0]*img_size[1]])

    return x_test_new, y_train, x_test_new, y_test