# 机器学习之逻辑回归python实现



*前言：* 纸上得来终觉浅，绝知此事要躬行

[TOC]

###1. 理论基础
模型假设：
$$H(W;X)=\frac{1}{1+e^{-(wx+b)}}$$
损失函数(通过最大似然推导)：
$$L(W;X)=-\frac{1}{m}\prod_{i=0}^m[y^ilog(H(W;X)) + (1-y^i)log(1-H(W;X)]$$
权值更新：
$$w_j=w_j-\alpha(H(w;x)-y)x^j$$
$$b_j=b_j-\alpha(H(w;x)-y)$$
###2. python实现
最难最关键的地方就是梯度下降，下面贴出最重要的一行代码：
```python
self.theta = self.theta - 1.0 / m * self.alpha * np.transpose(x_train) 
                * (self.sigmoid(x_train * self.theta) - y_train)
```
这里theta是增广向量。对照着上面的权值更新看懂这行代码就行了。
梯度下降的函数如下：
```python
def __gradient_decent__(self, x_train, y_train):
    [m, n] = x_train.shape
    for i in xrange(self.iterator_num):
        print "step : %d" % i
        self.theta = self.theta - 1.0 / m * self.alpha * np.transpose(x_train) 
                        * (self.sigmoid(x_train * self.theta) - y_train)

```
随机梯度下降的函数如下：
```python
def __stochastic_gradient_decent__(self, x_train, y_train):
    [m, n] = x_train.shape
    for j in xrange(self.iterator_num):
        data_index = range(m)
        print "step : ", j
        for i in xrange(m):
            #动态调整学习率
            self.alpha = 4 / (1.0 + j + i) + 0.01
            #随机选取一个样本，随后将其从dataIndex中删除
            rand_index = int(np.random.uniform(0, len(data_index)))
            error = self.sigmoid(np.dot(x_train[rand_index, :], self.theta)) - y_train[rand_index]
            self.theta = self.theta - np.multiply(self.alpha, np.multiply(error, x_train[rand_index].T))
            del data_index[rand_index]
```
随机梯度下降相对于梯度下降，主要改动：
- 每次使用一个样本更新，减少计算，加快运行速度
- 动态调整学习率，缓解数据波动或者高频波动。学习率随着i与j的增大而减小
- 每次更新使用的样本是随机选取的，可以减少系数更新的周期性波动

3. 完整代码
LogisticRegression.py
```python
#encoding=utf-8
'''
Copyright : CNIC
Author : LiuYao
Date : 2017-8-31
Description : Define the LogisticRegression class
'''

import numpy as np

class LogisticRegression(object):
    '''
    implement the lr relative functions
    '''

    def __init__(self, alpha=0.1, iterator_num=100, optimization='sgd'):
        '''
        lr parameters init
        Args:
            alpha: the learning rate, default is 0.1.
            iterator_num: the count of iteration, default is 100.
            optimization: the optimization method, such as 'sgd', 'gd', default is 'sgd'.
        '''
        self.alpha = alpha
        self.iterator_num = iterator_num
        self.optimization = optimization

    def train(self, x_train, y_train):
        '''
        lr train function
        Args:
            x_train: the train data, shape = (m, n), m is the count of the samples, 
                    n is the count of the features
            y_train: the train labels, shape = (m, 1), m is the count of the samples
        '''
        m, n = x_train.shape
        x_train = np.mat(x_train)
        self.theta = np.mat(np.random.ranf(n + 1, 1))
        x_train = np.hstack((x_train, np.ones((m, 1))))
        y_train = np.mat(np.reshape(y_train, (m, 1)))
        if(self.optimization == 'gd'):
            self.__gradient_decent__(x_train, y_train)
        elif(self.optimization == 'sgd'):
            self.__stochastic_gradient_decent__(x_train, y_train)

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def __gradient_decent__(self, x_train, y_train):
        [m, n] = x_train.shape
        for i in xrange(self.iterator_num):
            print "step : %d" % i
            self.theta = self.theta - 1.0 / m * self.alpha * np.transpose(x_train) * (self.sigmoid(x_train * self.theta) - y_train)
    
    def __stochastic_gradient_decent__(self, x_train, y_train):
        [m, n] = x_train.shape
        for j in xrange(self.iterator_num):
            data_index = range(m)
            print "step : ", j
            for i in xrange(m):
                #动态调整学习率
                self.alpha = 4 / (1.0 + j + i) + 0.01
                #随机选取一个样本，随后将其从dataIndex中删除
                rand_index = int(np.random.uniform(0, len(data_index)))
                error = self.sigmoid(np.dot(x_train[rand_index, :], self.theta)) - y_train[rand_index]
                self.theta = self.theta - np.multiply(self.alpha, np.multiply(error, x_train[rand_index].T))
                del data_index[rand_index]


    def predict(self, x_test):
        '''
        lr predict function
        Args:
            x_test: the test data, shape = (m, 1), m is the count of the test data
        '''
        [m, n] = x_test.shape
        x_test = np.mat(x_test)
        x_test = np.hstack((x_test, np.ones((m, 1))))
        return self.sigmoid(x_test * self.theta)
```
test.py(这里只是画出分界线)
```python
#encoding=utf-8
'''
Copyright : CNIC
Author : LiuYao
Date : 2017-8-31
Description : test my algorithm
'''

import pandas as pd
import numpy as np
from marchine_learning.linear_model import LogisticRegression
import matplotlib.pyplot as plt

def load_data():
    '''
    load data
    '''
    data = pd.read_csv('./data.csv')
    x = data[['x', 'y']]
    y = data['label']
    return x, y

def plot(x_train, y_train, theta):
        [m, n] = x_train.shape
        plt.scatter(x_train.values[:, 0], x_train.values[:, 1], c=y_train)
        x1 = np.random.rand(100, 1) * 25
        x2 = (-theta[2] - x1 * theta[0]) / theta[1]
        plt.plot(x1, x2)
        plt.show()

def main():
    '''
    program entry
    '''
    x, y = load_data()
    #这里有两个选项：'gd'（梯度下降）， 'sgd'（随机梯度下降）
    lr = LogisticRegression.LogisticRegression(iterator_num=10, optimization='sgd')
    lr.train(x.values, y.values.T)
    plot(x, y, lr.theta)

if __name__ == '__main__':
    main()
```
运行结果：
<center>梯度下降-10次迭代

![梯度下降-10次迭代](http://img.blog.csdn.net/20170831200748545?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvaWxvdmV5b3VzdW5uYQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)</center>

<center>随机梯度下降-10次迭代

![随机梯度下降-10次迭代](http://img.blog.csdn.net/20170831201042267?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvaWxvdmV5b3VzdW5uYQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)</center>

<center>梯度下降-5次迭代

![梯度下降-5次迭代](http://img.blog.csdn.net/20170831201649206?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvaWxvdmV5b3VzdW5uYQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)</center>

<center>随机梯度下降-5次迭代

![随机梯度下降-5次迭代](http://img.blog.csdn.net/20170831201807656?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvaWxvdmV5b3VzdW5uYQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)</center>

总结：随机梯度下降可以减少计算量并加速计算，当数据集很大时，使用随机梯度下降会更好。