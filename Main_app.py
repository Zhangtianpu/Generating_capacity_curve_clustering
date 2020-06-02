#-*- coding utf-8 -*-
# @Time : 2020/6/2 17:13
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import silhouette_score

#'利用SSE选择k'
def choose_k_by_sse(np_data,max_clusters):
    SSE = []  # 存放每次结果的误差平方和
    for k in range(1,max_clusters):
        #if k%2==0:
            estimator = KMeans(n_clusters=k)  # 构造聚类器
            estimator.fit(np_data)
            SSE.append(estimator.inertia_) # estimator.inertia_获取聚类准则的总和
    X = [ i for i in range(1,max_clusters)]
    plt.xlabel('k')
    plt.ylabel('SSE')
    plt.plot(X,SSE,'o-')
    plt.show()

#利用轮廓系数选择k
def choose_k_by_score(np_data,max_clusters):
    Scores = []  # 存放轮廓系数
    for k in range(2,max_clusters):
        estimator = KMeans(n_clusters=k)  # 构造聚类器
        estimator.fit(np_data)
        Scores.append(silhouette_score(np_data,estimator.labels_,metric='euclidean'))
    X = range(2,max_clusters)
    plt.xlabel('k')
    plt.ylabel('score')
    plt.plot(X,Scores,'o-')
    plt.show()


def get_Kmeans_plot(np_data, label_pred, color_list):
    for i in range(len(label_pred)):
        x = [i for i in range(np_data.shape[-1])]
        plt.plot(x, np_data[i], color_list[label_pred[i]])
    plt.show()


def get_divid_plot(np_data, label_pred, color_list):
    label_list = pd.Series(label_pred).unique()
    for label in label_list:
        for i in range(len(label_pred)):
            if label_pred[i] == label:
                x = [i for i in range(np_data.shape[-1])]
                plt.plot(x, np_data[i], color_list[label])
        plt.show()


def plot_centroids(centroids, color_list):
    label_list = pd.Series(label_pred).unique()
    for label in label_list:
        x = [i for i in range(centroids.shape[-1])]
        plt.plot(x, centroids[label], color_list[label])
        plt.show()

def my_KMeans(np_data,n_clusters):
    estimator =KMeans(n_clusters)   #构造一个聚类数为5的聚类器
    estimator.fit(np_data)   #聚类
    label_pred = estimator.labels_  #获取聚类标签
    centroids = estimator.cluster_centers_ #获取聚类中心
    return label_pred,centroids
def plot_kmeans(np_data,label_pred,centrodis,color_list):
    get_Kmeans_plot(np_data,label_pred,color_list)
    get_divid_plot(np_data,label_pred,color_list)
    plot_centroids(centroids,color_list)

def output_clustering_re(data,label_pred,output_path):
    data['label']=label_pred
    data.to_csv(output_path,sep=',',index=False)

def helper(x):
    p=x.mean()+3*x.std()
    re=x.apply(lambda y:0 if y>p else y)
    return re

def max_scale(x):
    re=x.apply(lambda y:(y-x.min())/(x.max()-x.min()))
    return re

data=pd.read_excel("./data/data.xlsx")
data_fill=data.fillna(value=0)
data_feature=data_fill.iloc[:,5:]
#使用零去替换负数，发电量不可能是负数
data_feature_drop_pos=data_feature.applymap(lambda x: 0 if x<0 else x)
data_feature_drop_pos.describe()
#利用3倍标准差去除异常数值
data_feature_drop_3std=data_feature_drop_pos.apply(helper,axis=1)
data_max_scale=data_feature_drop_3std.apply(max_scale,axis=1)
np_data=np.array(data_max_scale)
label_pred,centroids=my_KMeans(np_data,3)
#给原始数据集添加标签，并将结果输出到指定文件中
output_clustering_re(data,label_pred,"./output/ouput_data.csv")
color_list=['#e24fff','g','r']
get_Kmeans_plot(np_data,label_pred,color_list)
plot_kmeans(np_data,label_pred,centroids,color_list)