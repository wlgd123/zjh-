# -*- coding:utf8 -*-
""""
import pandas as pd
import mglearn
import sklearn.neighbors

iris_dataset = load_iris()
print(iris_dataset.DESCR[:150])
data = pd.DataFrame(iris_dataset.data,columns=iris_dataset.feature_names)
data['category'] = iris_dataset.target
print(data)

from sklearn.datasets import load_iris
iris_dataset = load_iris()
print(iris_dataset.DESCR[:150])
from sklearn.model_selection import train_test_split  #train_test_split 函数默认是将数据75%分为训练集，25%分为测试集
Xtrain,Xtest,ytrain,ytest = train_test_split(iris_dataset.data,iris_dataset.target,random_state=0)
#print(Xtrain.shape,ytrain.shape)
#print(Xtest.shape,ytest.shape)
# 利用X_train中的数据创建DataFrame
# 利用iris_dataset.feature_names中的字符串对数据列进行标记

iris_dataframe = pd.DataFrame(Xtrain,columns=iris_dataset.feature_names)
grr = pd.plotting.scatter_matrix(iris_dataframe,figsize=(15,15),marker='o',hist_kwds={'bins':20},alpha=0.8,cmap=mglearn.cm3)


#sklearn.neighbors.KNeighborsClassifier(n_neighbors=3)
#实例化一个估计器，使用fit方法进行训练，fit可用来求距离
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# 假设df是一个包含GDP和货邮吞吐量的DataFrame，其中'GDP'是GDP列，'CargoThroughput'是货邮吞吐量列
# df = pd.read_csv('your_data.csv')  # 读取你的数据

# 示例数据（这里仅作演示，你需要替换为真实数据）
years=[2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021]
GDP=[0.13,0.15,0.16,0.19,0.23,0.27,0.32,0.41,0.47,0.55,0.66,0.78,0.87,1.00,1.05,1.15,1.31,1.49,1.62,1.56,1.77]
CargoThroughput=[4.4507,5.1881,5.5022,6.1378,6.4017,7.3770,8.9596,8.9853,10.1870,11.0190,12.2760,12.8200,12.9450,14.3030,15.4660,17.5300,18.5020,22.1580,24.3190,18.9360,31.5600]
df = pd.DataFrame({'Year': years, 'GDP': GDP, 'CargoThroughput': CargoThroughput})

# 提取特征和标签
X = df['GDP'].values.reshape(-1, 1)  # 特征需要是二维数组
y = df['CargoThroughput'].values

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=3, random_state=42)  #test_size()浮点数是测试集比例，整数是有多少组数据计入测试集，random_state：随机数种⼦，种⼦不同，每次采的样本不⼀样；种⼦相同，采的样本不变
# 但由于我们已知测试集是最后三年，所以可以直接索引
X_train = X[:-3]
X_test = X[-3:]
y_train = y[:-3]
y_test = y[-3:]

lr=LinearRegression().fit(X_train,y_train)
print("R2_score:{}".format(lr.score(X_test,y_test)))






"""""""""
#数据可视化
from sklearn.datasets import load_iris
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

iris=load_iris()
import  pandas as pd
iris_data=pd.DataFrame(data=iris.data,columns=['sepal_length','sepal_width','petal_length','petal_width'])#把数据用Dataframe来储存，并更改列的名称
iris_data["target"]=iris.target#增加一列命名标签来记载鸢尾花标签
print(iris_data)
def irs_plot(data,col1,col2):
    sns.lmplot(x=col1,y=col2,data=data,hue='target',fit_reg=False)#hue 目标值是什么，fit_reg是否进行线性拟合
    plt.show()


irs_plot(iris_data,'sepal_length','petal_width')
irs_plot(iris_data,'sepal_width','petal_length')

x_train,x_test,y_train,y_test=train_test_split(iris.data,iris.target)
"""""""""""





































