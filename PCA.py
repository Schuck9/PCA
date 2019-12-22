"""
A simple implementation of principal components analysis
@data: 2019.12.22
@author: Tingyu Mo
"""
import pandas as pd
import numpy as np
import os
import math
import csv
import matplotlib.pyplot as plt
import struct
from scipy import stats
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score,accuracy_score
from sklearn.decomposition import PCA
from KNN import KNN

class PCA_SVD():
    def __init__(self,):
        self.img_shape = None
        self.img_size = (128,128)

    def load_mnist(self,path, kind='train'):
        """Load MNIST data from `path`"""
        labels_path = os.path.join(path,
                                '%s-labels-idx1-ubyte'
                                % kind)
        images_path = os.path.join(path,
                                '%s-images-idx3-ubyte'
                                % kind)
        with open(labels_path, 'rb') as lbpath:
            magic, n = struct.unpack('>II',
                                    lbpath.read(8))
            labels = np.fromfile(lbpath,
                                dtype=np.uint8)

        with open(images_path, 'rb') as imgpath:
            magic, num, rows, cols = struct.unpack('>IIII',
                                                imgpath.read(16))
            images = np.fromfile(imgpath,
                                dtype=np.uint8).reshape(len(labels), 784)
            # images = np.fromfile(imgpath,
            #                     dtype=np.uint8)
            
        return images, labels

    def PCA_decomposition(self,data,n_components="mle",svd_sovler = "full"):
        print("PCA decomposition start!")
        pca = PCA(n_components=n_components,svd_solver="full")
        compress_data = pca.fit_transform(data)
        print("number of components: ",pca.n_components)
        print("explained variance ratio: ",pca.explained_variance_ratio_)
        return compress_data,pca

if __name__=="__main__":
    Root_dir = r'D:/Pattern_Recognion/Exp5-10'
    datasets_dir = os.path.join(Root_dir,"datasets")
    os.chdir(Root_dir)
    dataset_path = os.path.join(datasets_dir,"mnist")
    PS = PCA_SVD()
    x_train, y_train =PS.load_mnist(dataset_path,"train")
    print("data loaded!")
    x_train, y_train = x_train[:1000],y_train[:1000]
    x_train,pca = PS.PCA_decomposition(x_train,n_components=50,svd_sovler = "auto")
    x_test, y_test =PS.load_mnist(dataset_path,"t10k")
    x_test, y_test = x_test[:20],y_test[:20]
    x_test = pca.transform(x_test)
    K_Nearest = KNN(kN = 5,method = "K_Nearest")
    K_Nearest.train(x_train,y_train)
    y_pred = K_Nearest.predict(x_test)
    acc,prec = K_Nearest.evaluate(y_pred,y_test)
    print("acc: {} prec:{}".format(acc,prec))


