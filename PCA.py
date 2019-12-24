"""
A simple implementation of principal components analysis
@date: 2019.12.24
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

    def cal_means(self,x,axis=0):
        return x.mean(axis=axis)

    def matrix_decompose(self,matrix):
        '''
        decompose a matrix to eigenvalue,featurevector on condition of the matrix is symmetric
        '''
        eigenvalues,featurevector = np.linalg.eig(matrix)
        return eigenvalues,featurevector

    def find_max_eigenvalues(self,eigenvalues,n_components):
        '''
        find the most representable eigenvalues as the components
        '''
        eigenvalues_index = np.argsort(eigenvalues,kind='quicksort')[::-1]
        return eigenvalues_index[:n_components]

    def draw_featurevector(self,eigenvalues_index,featurevector,n_components):
        '''
        according to the selected eigenvalues draws featurevector
        '''
        return featurevector[:,eigenvalues_index[:n_components]]

    def cov_variance_matrix(self,x,mean):
        '''
        compute the covariance matrix
        spposes the x is in form of col vector

        sigma = sum((xi-mi)*(xi-mi)T)
        '''
        return (x-mean).T.dot((x-mean))
    def feature_mapping(self,samples,components_mat):
        return np.dot(samples,components_mat)

    def matrix_symmetric(self,x):
        '''
        C=(C+C')/2
        '''
        return (x+x.T)*1.0/2

    def PCA_decomposition(self,samples,n_components=1):
        '''
        steps of principal component analasis 
        '''
        print("PCA decomposition start!")
        #计算所有样本均值
        all_samples_mean = self.cal_means(samples) 
        #计算样本总体的协方差矩阵
        cov_mat = self.cov_variance_matrix(samples,all_samples_mean)
        cov_mat = cov_mat*1.0/samples.shape[0] #归一化
        cov_mat = self.matrix_symmetric(cov_mat)#对称化处理
        #进行矩阵特征值分解
        eigenvalues,featurevector = self.matrix_decompose(cov_mat)
        #找到前n个主成分/最优主成分的相关特征值
        eigenvalues_index = self.find_max_eigenvalues(eigenvalues,n_components)
        #找到前n个主成分/最优主成分 即特征向量
        components_mat = self.draw_featurevector(eigenvalues_index,featurevector,n_components)
        print("number of components: ",n_components)
        # print("components matrix:",components_mat)
        #进行特征映射 降维
        compress_data = self.feature_mapping(samples,components_mat)
        return compress_data,components_mat

    def PCA_decomposition_sklearn(self,data,n_components="mle",svd_sovler = "full"):
        '''
        implementation of sklearn
        '''
        print("PCA decomposition start!")
        pca = PCA(n_components=n_components,svd_solver="full")
        compress_data = pca.fit_transform(data)
        print("number of components: ",pca.n_components)
        print("explained variance ratio:\n",pca.explained_variance_ratio_)
        return compress_data,pca

if __name__=="__main__":
    Root_dir = r'D:/Pattern_Recognion/Exp5-10'
    datasets_dir = os.path.join(Root_dir,"datasets")
    os.chdir(Root_dir)
    dataset_path = os.path.join(datasets_dir,"mnist")
    PS = PCA_SVD()
    datasets = "mnist"
    if datasets == "mnist":
        x_train, y_train =PS.load_mnist(dataset_path,"train")
        x_train, y_train = x_train[:10000],y_train[:10000]
        x_test, y_test =PS.load_mnist(dataset_path,"t10k")
        x_test, y_test = x_test[:2000],y_test[:2000]
        print("data loaded!")
    elif datasets == "normal":   
        w1_features = np.array([[0.2331, 2.3385], [1.5207, 2.1946], [0.6499, 1.6730], [0.7757, 1.6365],
            [1.0524, 1.7844], [1.1974, 2.0155], [0.2908, 2.0681], [0.2518, 2.1213],
            [0.6682, 2.4797], [0.5622, 1.5118], [0.9023, 1.9692], [0.1333, 1.8340],
            [-0.5431, 1.8704], [0.9407, 2.2948], [-0.2126, 1.7714], [0.0507, 2.3939],
            [-0.0810, 1.5648], [0.7315, 1.9329], [0.3345, 2.2027], [1.0650, 2.4568],
            [-0.0247, 1.7523], [0.1043, 1.6991], [0.3122, 2.4883], [0.6655, 1.7259],
            [0.5838, 2.0466], [1.1653, 2.0226], [1.2653, 2.3757], [0.8137, 1.7987],
            [-0.3399, 2.0828], [0.5152, 2.0798], [0.7226, 1.9449], [-0.2015, 2.3801],
            [0.4070, 2.2373], [-0.1717, 2.1614], [-1.0573, 1.9235], [-0.2099, 2.2604]])
        w2_features = np.array([[1.4010, 1.0298], [1.2301, 0.9611], [2.0814, 0.9154], [1.1655, 1.4901],
                    [1.3740, 0.8200], [1.1829, 0.9399], [1.7632, 1.1405], [1.9739, 1.0678],
                    [2.4152, 0.8050], [2.5890, 1.2889], [2.8472, 1.4601], [1.9539, 1.4334],
                    [1.2500, 0.7091], [1.2864, 1.2942], [1.2614, 1.3744], [2.0071, 0.9387],
                    [2.1831, 1.2266], [1.7909, 1.1833], [1.3322, 0.8798], [1.1466, 0.5592],
                    [1.7087, 0.5150], [1.5920, 0.9983], [2.9353, 0.9120], [1.4664, 0.7126],
                    [2.9313, 1.2833], [1.8349, 1.1029], [1.8340, 1.2680], [2.5096, 0.7140],
                    [2.7198, 1.2446], [2.3148, 1.3392], [2.0353, 1.1808], [2.6030, 0.5503],
                    [1.2327, 1.4708], [2.1465, 1.1435], [1.5673, 0.7679], [2.9414, 1.1288]])
        w1_labels = np.ones(w1_features.shape[0]) # 1 为w1类的label
        w2_labels = -1*np.ones(w2_features.shape[0])# -1 为w2类的label
        x_test = np.array([[1, 1.5], [1.2, 1.0], [2.0, 0.9], [1.2, 1.5], [0.23, 2.33]])
        x_train = np.vstack((w1_features,w2_features))
        y_train = np.hstack((w1_labels,w2_labels))
        print("data loaded!")

    # x_train,pca = PS.PCA_decomposition_sklearn(x_train,n_components=1,svd_sovler = "auto")
    # x_test = pca.transform(x_test)
    x_train,components_mat = PS.PCA_decomposition(x_train,n_components=50)
    x_test = PS.feature_mapping(x_test,components_mat)
    K_Nearest = KNN(kN = 5,method = "K_Nearest")
    K_Nearest.train(x_train,y_train)
    y_pred = K_Nearest.predict(x_test)
    # for i in range(5):
    #     if y_pred[i] == -1 :
    #         print("样本: {} 属于w{}类".format(x_test[i,0],2))
    #     else:
    #         print("样本: {} 属于w{}类".format(x_test[i,0],1))
    acc,prec = K_Nearest.evaluate(y_pred,y_test)
    print("acc: {} prec:{}".format(acc,prec))

            
