#机器学习之感知机python实现

[TOC]

##一. 理论基础
####1. 损失函数

$$L(w,b) = -\sum_{i = 0}^{m}{y_i(w x_i + b)}$$
####2. 更新参数

$$w = w + \alpha \sum_{i=0}^{m}{x_i y_i}$$
$$b = b + \alpha \sum_{i=0}^{m}{y_i}$$

##二. python实现

####1. 代码
Perceptron.py

```python
#encoding=utf-8
'''
implements the perceptron
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class Perceptron:
    def __init__(self, alpha=0.1, iterator_num=100):
        self.alpha = alpha
        self.iterator_num = iterator_num

    def train(self, x_train, y_train):
        x_train = np.mat(x_train)
        y_train = np.mat(y_train)
        [m, n] = x_train.shape
        self.theta = np.mat(np.zeros((n, 1)))
        self.b = 0
        self.__stochastic_gradient_decent__(x_train, y_train)

    def __gradient_decent__(self, x_train, y_train):
        x_train = np.mat(x_train)
        y_train = np.mat(y_train)
        for i in xrange(self.iterator_num):
            self.theta = self.theta + self.alpha * x_train.T * y_train
            self.b = self.b + self.alpha * np.sum(y_train)

    def __stochastic_gradient_decent__(self, x_train, y_train):
        x_train = np.mat(x_train)
        y_train = np.mat(y_train)
        [m, n] = x_train.shape
        for i in xrange(self.iterator_num):
            for j in xrange(m):
                self.theta = self.theta + self.alpha * x_train[j].T * y_train[j] 
                self.b = self.b + self.alpha * y_train[j]

def main():
    '''
    test unit
    '''
    print "step 1: load data..."
    data = pd.read_csv('/home/LiuYao/Documents/MarchineLearning/data.csv')
    # data = data.ix[0:30, :]
    
    x = np.mat(data[['x', 'y']].values)
    y = np.mat(data['label'].values).T
    y[y == 0] = -1
    print y[y == 1]
    print "positive samples : ", y[y == 1].shape
    print "nagetive samples : ", y[y == -1].shape

    ## step 2: training...
    print "step 2: training..."
    perceptron = Perceptron(alpha=0.1,iterator_num=100)
    perceptron.train(x, y)

    ## step 3: show the decision boundary
    print "step 3: show the decision boundary..."	
    print perceptron.theta
    x_min = np.min(x[:, 0])
    x_max = np.max(x[:, 0])
    y_min = (-perceptron.b - perceptron.theta[0] * x_min) / perceptron.theta[1]
    y_max = (-perceptron.b - perceptron.theta[0] * x_max) / perceptron.theta[1]
    plt.plot([x_min, x_max], [y_min[0,0], y_max[0,0]])
    plt.scatter(x[:, 0].getA(), x[:, 1].getA(), c=y.getA())
    plt.show()

if __name__ == '__main__':
    main()
```

####2. 结果

最终结果会有一些问题，不知道为什么，请知道的大神解答一下,梯度下降和随机梯度下降都试了，结果一样。
* 用前三十个数据的时候，分界面正确；
* 用前60个数据的时候，分界面就离谱了。
* 所有数据的时候分界面明显有一些偏差；

<center>
前30个数据

![这里写图片描述](http://img.blog.csdn.net/20170904135250367?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvaWxvdmV5b3VzdW5uYQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)</center>

<center>
前60个数据

![这里写图片描述](http://img.blog.csdn.net/20170904135301158?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvaWxvdmV5b3VzdW5uYQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)</center>

<center>
所有数据

![所有数据](http://img.blog.csdn.net/20170904135012985?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvaWxvdmV5b3VzdW5uYQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
</center>

数据如下：

```
x,y,label
10.6,13.5,0
12.55,12.1,0
12.05,13.95,0
10.85,15.05,0
7.5,12.75,0
9.45,11.25,0
8.95,13.3,0
8.45,15.5,0
12.15,12.2,0
5.15,8.25,0
17.45,6.0,0
18.55,5.8,1
16.1,4.45,1
13.95,6.75,1
15.4,7.85,1
17.7,9.25,1
19.3,9.8,1
20.5,8.1,1
8.15,2.05,1
11.7,4.9,1
21.1,4.6,1
21.1,9.75,1
17.65,11.4,1
6.95,9.9,1
5.8,12.05,1
7.35,10.0,0
8.15,11.05,0
7.4,11.65,0
4.55,11.35,0
4.4,15.2,0
4.2,16.6,0
7.85,17.1,0
13.45,18.95,0
15.35,18.9,0
18.35,17.1,0
16.85,15.75,0
15.75,10.8,0
13.95,9.25,0
10.25,10.7,0
9.85,12.05,0
14.25,17.45,0
10.15,17.55,0
7.0,14.1,0
4.85,11.8,0
4.75,8.6,0
3.25,6.65,0
1.9,9.55,0
2.1,14.75,0
1.5,10.9,0
5.75,9.65,0
7.65,8.1,0
9.6,9.1,0
10.1,2.0,1
12.2,2.75,1
8.0,6.3,1
6.8,5.1,1
7.35,3.65,1
9.5,4.65,1
13.05,7.7,1
17.85,5.15,1
24.35,7.4,1
20.4,13.1,1
14.55,15.4,1
24.95,11.05,1
22.15,11.15,1
22.85,5.85,1
22.5,4.15,1
19.3,1.6,1
15.6,0.25,1
14.5,1.55,1
14.5,3.95,1
10.35,7.1,1
13.65,6.75,1
14.0,5.55,1
12.15,4.8,1
10.5,4.15,1
22.95,8.75,1
21.25,7.05,1
17.05,7.9,1
17.05,7.9,1

```