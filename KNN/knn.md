#机器学习之KNN算法python实现

[TOC]

##一. 理论基础
####1. 距离度量
特征空间中两个实例点的距离是两个实例点相似程度的反映。一般采用欧氏距离，但也可以是其他距离，如cosine距离，曼哈顿距离等.

####2. k值选择
* k值越大，意味着模型越简单，学习近似误差大，估计误差小，欠拟合；
* k值越小，意味着模型越复杂，学习近似误差小，估计误差大，过拟合，而且对近邻的实例点敏感.
>通常采取交叉验证选取最优的k值。
####3. 分类决策规则
多数表决，即由输入实例的K个近邻的多数类决定输入实例的类别。
####4. kd树
高效实现k近邻，类似于二分查找，只不过是在高维的二分查找。
kd树更适用于训练实例数远大于空间维数时的k近邻搜索，当空间维数接近训练实例数时，它的效率会迅速下降，几乎接近线性扫描。

##二. python实现
实现了knn的暴力搜索，也实现了kd-tree搜索，但是kd-tree只能找最近邻，即k=1，当k>1时，还未实现，初步想法：可以考虑k次搜索kd-tree，每次搜索后将最近邻节点删除，继续搜索，就找到了top k近邻搜索；这样的话就得实现kd-tree的删除插入。

####1. 代码

knn.py

```python
#encoding=utf-8

'''
implement the knn algorithm
'''

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.stats import mode
import matplotlib.pyplot as plt

class KNN:
    
    def __init__(self):
        pass

    def predict(self, x_train, y_train, x_test, k=3):
        self.k = k
        m_train = x_train.shape[0]
        m_test = x_test.shape[0]
        x_train = np.mat(x_train)
        y_train = np.mat(y_train)
        x_test = np.mat(x_test)

        #1. get the distances between each sample in train samples and each sample in test samples,
        #the distances matrix's shape is (m_test, m_train).
        dists = self.__distance__(x_train, x_test)
        #2. sort the distances by row, and get the sort index
        sort_idx = np.argsort(dists, axis=1)
        #3. get the x index and y index, which is top k distance sample index
        x_idx = np.tile(np.mat(range(m_test)).T, [1, self.k])
        y_idx = sort_idx[:, 0 : self.k]
        #4. get the top k distance labels, and the matrix's shape is (m_test, k)
        labels = np.tile(y_train.T, [m_test, 1])
        p_labels = labels[x_idx, y_idx]
        #5. get the mode of each row, which means the most labels
        y_predict = np.mat(mode(p_labels, axis=1)[0])
        return y_predict
    
    def __distance__(self, x_train, x_test):
        '''
        force compute to get the distance between each sample in train samples and each sample in test samples
        '''
        m_train = x_train.shape[0]
        m_test = x_test.shape[0]
        dists = np.zeros((m_test, m_train))
        count = 0
        for test in x_test:
            test =  np.tile(test, [m_train, 1])
            distance = np.sum(np.multiply(x_train - test, x_train - test), axis=1)
            dists[count] = distance.T
            count += 1
        return dists

    def create_kd_tree(self, datalist):
        '''
        create KD tree
        Args:
            data: data list
        '''
        root = KDNode()
        self.build_tree(root, datalist)
        self.kd_tree = root
        return root

    def build_tree(self, parent, datalist):
        '''
        recursive build tree function
        Args:
            parent: parent node
        '''
        m = datalist.shape[0]
        #if the length of data is equal to 1, the node is a leaf node
        if m == 1:
            parent.data = datalist
            return
        
        #compute the best split demension by the variance of each demension of the data
        demension = np.argmax(np.var(datalist, axis=0))
        #sort the data by the chosen demension
        sorted_index = np.argsort(datalist[:, demension], axis=0)
        #get the index of the middle value in the datalist
        middle = m / 2
        #get the left data
        l_data = datalist[np.squeeze(sorted_index[0 : middle].getA()), :]
        #get the right data
        r_data = datalist[np.squeeze(sorted_index[middle + 1 : ].getA()), :]

        #assign the property of the parent node
        parent.data = datalist[np.squeeze(sorted_index[middle, :].getA())]
        parent.demension = demension
        parent.split_value = datalist[np.squeeze(sorted_index[middle, :].getA()), demension]

        #recursive build the child node if the length of rest data is not equal to zero
        if len(l_data) != 0:
            l_node = KDNode()
            parent.left = l_node
            self.build_tree(l_node, l_data)
        
        if len(r_data) != 0:
            r_node = KDNode()
            parent.right = r_node
            self.build_tree(r_node, r_data)

    def __distance_by_kd_tree__(self, x_test):
        '''
        get nearest neighbors matrix by kd_tree search
        '''
        m = x_test.shape[0]
        dists = np.zeros((m, 1))
        count = 0
        for x in x_test:
            dists[count] = self.__find_neighbor__(x, self.kd_tree)
            count += 1
        return np.mat(dists)
            
    
    def __find_neighbor__(self, x, node):
        '''
        recursive find the neighbor of x in kd-tree
        Args:
            the root node of current child tree
        
        steps:
            1. if the current is leaf node, return the data in the node as the nearest neighbor
            2. if the value of x is less than the split value, take the neighbor of left child
               tree as nearest neighbor. And then check if another child tree has the more nearest
               neighbor;
               if the value of x is more than the split value, do it as like mentioned above;
            3. check if the current node and x has more nearest distance
        '''
        
        if node.demension == None: 
            return node.data
        
        if (x[0, node.demension] <= node.split_value) and node.left:
            neighbor = self.__find_neighbor__(x, node.left)
            if node.right \
                and (np.abs(x[0, node.demension] - node.split_value) < self.__euclidean_distance__(x, neighbor)) \
                and (self.__euclidean_distance__(self.__find_neighbor__(x, node.right), x) < self.__euclidean_distance__(x, neighbor)):
                    neighbor = self.__find_neighbor__(x, node.right)
        elif (x[0, node.demension] > node.split_value) and node.right:
            neighbor = self.__find_neighbor__(x, node.right)
            if node.left \
                and (np.abs(x[0, node.demension] - node.split_value) < self.__euclidean_distance__(x, neighbor)) \
                and (self.__euclidean_distance__(self.__find_neighbor__(x, node.left), x) < self.__euclidean_distance__(x, neighbor)):
                    neighbor = self.__find_neighbor__(x, node.left)
        else:
            # this happens as like:
            # x = 6, node = 5
            #         5
            #        /
            #       4
            neighbor = node.data

        if self.__euclidean_distance__(x, node.data) < self.__euclidean_distance__(x, neighbor):
            neighbor = node.data
        return neighbor

    def __euclidean_distance__(self, x1, x2):
        '''
        compute the euclidean distance
        '''
        return np.sum(np.multiply(x1 - x2, x1 - x2))

class KDNode:
    def __init__(self, data=None, demension=None, split_value=None, left=None, right=None):
        self.data = data
        self.demension = demension
        self.split_value = split_value
        self.left = left
        self.right = right

def main():
    '''
    KNN test unit
    '''

    #1. load data
    print "1. loading data..."
    data = pd.read_csv('/home/LiuYao/Documents/MarchineLearning/multi_data.csv')
    data['label'] = data['label'] + 1
    x_train, x_test, y_train, y_test = train_test_split(
                                                    data.values[:, 0:2], 
                                                    data.values[:, 2], 
                                                    test_size=0.2, 
                                                    random_state=0
                                                    )

    x_train = np.mat(x_train)
    x_test = np.mat(x_test) 
    y_train = np.mat(y_train).T
    y_test = np.mat(y_test).T

    #2. predict
    print '2. predicting...'
    knn = KNN()
    y_predict = knn.predict(x_train, y_train, x_test, k=1)

    #3. show the results
    print '3. show the results...'
    plt.scatter(x_train.getA()[:, 0], x_train.getA()[:, 1], c=y_train.T.getA()[0], marker='o')
    plt.scatter(x_test.getA()[:, 0], x_test.getA()[:, 1], c=y_predict.T.getA()[0], marker='*')
    plt.show()

    

def test_build_tree():
    '''
    test building the kd tree
    '''
    datalist = np.mat([[3, 1, 4],
                       [2, 3, 7],
                       [2, 1, 3],
                       [2, 4, 5],
                       [1, 4, 4],
                       [0, 5, 7],
                       [6, 1, 4],
                       [4, 3, 4],
                       [5, 2, 5],
                       [4, 0, 6],
                       [7, 1, 6]])
    knn = KNN()
    tree = knn.create_kd_tree(datalist)
    res = knn.__find_neighbor__(np.mat([[3,1,5]]), tree)
    print res

if __name__ == '__main__':
    main()

```

####2. 结果
图中五角星表示预测数据，圆点表示训练数据，可能需要将图片放大才能看清楚。

![knn_results](http://img.blog.csdn.net/20170905160201864?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvaWxvdmV5b3VzdW5uYQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

####3. 数据

```
x,y,label
14.7,17.85,0
17.45,17.45,0
18.85,15.15,0
17.25,13.7,0
13.9,12.5,0
10.5,15.65,0
8.4,20.5,0
11.1,21.85,0
17.6,21.65,0
23.0,19.75,0
24.45,12.4,0
16.25,3.3,0
8.85,5.05,0
5.55,8.8,0
6.05,11.75,0
26.45,6.9,0
28.95,6.6,0
21.75,8.35,0
21.05,10.95,0
23.9,17.05,0
19.7,18.2,0
12.4,19.3,0
9.25,18.4,0
10.3,8.95,0
16.65,8.6,0
37.3,15.1,0
32.5,10.0,1
33.05,11.45,1
25.75,17.1,1
20.15,17.8,1
12.85,20.75,1
12.8,8.65,1
14.65,5.6,1
24.2,6.3,1
24.1,11.45,1
22.05,10.85,1
17.2,12.85,1
13.7,15.55,1
6.4,19.45,1
8.1,11.5,1
14.9,10.35,1
10.05,12.65,1
25.3,1.55,1
16.5,3.8,1
17.0,6.25,1
17.85,7.35,1
23.75,9.7,1
21.65,16.3,1
16.3,19.8,1
13.9,19.85,1
13.1,14.35,1
16.55,17.9,1
16.3,18.15,1
15.3,17.7,1
13.35,18.3,1
12.8,17.5,1
13.9,15.65,1
15.65,16.5,1
31.35,7.2,1
31.35,6.95,1
29.45,5.6,1
27.15,4.85,1
26.6,5.2,1
27.8,7.35,1
28.7,8.35,1
28.8,10.25,1
5.65,11.25,1
7.8,9.7,1
7.5,11.9,1
3.55,14.45,1
3.5,13.65,1
5.1,10.95,1
5.1,11.05,1
18.65,9.1,2
19.4,10.95,2
20.1,12.7,2
17.25,14.85,2
14.6,15.25,2
14.8,11.75,2
13.5,6.4,2
14.75,5.25,2
18.05,4.05,2
21.25,3.3,2
23.75,3.85,2
32.65,5.5,2
33.65,7.05,2
32.15,13.2,2
30.8,15.25,2
30.15,16.5,2
24.7,18.0,2
22.05,19.45,2
20.1,21.5,2
20.0,22.05,2
26.8,22.45,2
29.7,21.8,2
30.95,21.35,2
30.85,19.15,2
28.4,18.7,2
26.35,19.65,2
26.5,19.9,2
30.05,19.35,2
32.75,16.35,2
33.95,14.65,2
34.05,14.6,2
30.05,18.3,3
27.65,20.6,3
25.05,21.85,3
24.1,18.2,3
23.8,15.3,3
25.6,14.45,3
28.1,12.4,3
29.35,10.95,3
29.85,8.25,3
30.55,14.1,3
28.45,15.7,3
31.85,18.15,3
18.2,19.3,3
16.85,19.8,3
7.45,9.35,3
13.35,13.9,3
32.4,9.75,3
23.8,1.05,3
30.75,4.05,4
30.5,5.3,4
30.35,5.95,4
28.9,9.0,4
27.7,9.9,4
24.75,11.4,4
21.65,13.8,4
19.75,17.45,4
23.4,20.05,4
18.2,21.75,4
9.65,18.4,4
5.6,13.45,4
8.8,9.75,4
11.25,11.2,4
5.35,15.95,4
6.1,16.0,4
24.25,15.95,4
31.55,17.0,4
32.45,14.0,4
24.05,12.4,4
12.3,12.85,4
7.15,19.3,4
21.35,22.4,4
27.95,17.65,4
24.3,7.7,4
17.5,3.6,4
12.7,6.95,4
11.25,10.7,4
9.0,15.2,4
7.05,19.15,4
17.45,13.4,4
16.0,10.75,4
16.75,12.0,4
18.25,11.5,4
18.15,9.15,4
17.1,9.5,4
17.0,10.25,4
12.8,7.75,4
17.0,6.7,4
21.15,8.5,4
20.35,9.35,4
19.45,10.0,4
18.45,10.05,4
18.0,8.0,4
20.15,8.0,4
21.45,6.65,4
19.2,6.45,4
15.25,8.4,4
14.8,9.5,4
14.45,7.7,4
16.45,6.6,4
18.0,5.85,4
18.85,5.7,4
19.6,6.1,4
29.9,14.15,6
31.4,15.8,6
32.15,15.3,6
33.25,13.65,6
33.8,11.95,6
33.85,10.9,6
33.9,10.35,6
32.6,10.75,6
32.1,12.55,6
34.15,12.55,6
35.35,11.8,6
35.15,10.4,6
34.65,9.1,6
34.3,8.9,6
35.55,9.25,6
36.35,12.45,6
37.75,9.4,6
37.75,8.5,6
36.4,8.2,6
35.0,8.05,6
35.65,7.15,6
37.55,6.4,6
39.2,7.1,6
36.5,9.85,0
36.8,9.35,0
37.5,7.7,0
34.05,9.8,0
20.2,20.3,0
26.45,21.1,0
27.9,20.65,0
27.15,16.9,0
25.5,13.1,0
24.05,10.2,0
23.45,5.3,0
20.8,10.95,0
18.95,14.65,0
17.15,16.25,0
11.3,17.0,0
11.65,11.1,0
15.95,4.8,0
21.45,3.25,0
13.9,3.05,0
10.75,6.2,0
9.3,16.85,0
10.25,19.5,0
12.7,15.95,0
13.3,14.3,0
15.7,11.45,0
16.1,10.9,0
14.1,14.2,0
14.35,13.65,0
15.3,14.1,0
15.65,14.7,0
15.75,15.85,0
15.75,15.85,0
19.2,18.4,0
19.2,17.05,0
19.3,15.6,0
20.45,14.1,0
21.4,11.65,0
26.3,11.85,2
20.5,18.75,2
17.55,19.95,2
13.1,16.65,2
9.55,12.7,2
7.85,15.65,2
7.75,16.8,2
9.1,10.35,2
21.25,8.6,2
22.65,5.0,2
11.75,10.7,1
11.05,14.55,1
13.85,8.45,1
11.7,6.65,1
10.75,4.95,1
10.95,3.75,1
6.85,8.35,1
11.35,5.7,1
13.25,4.6,1
7.45,7.95,1
15.7,13.35,1
16.85,14.3,1
13.55,10.4,1
9.55,7.3,1
34.3,8.0,1
28.15,8.45,1
25.15,8.75,1
22.3,14.6,1
29.5,15.55,1
28.2,14.1,1
32.95,10.15,1
29.15,11.4,1
20.85,18.95,1
22.0,17.8,1
12.5,2.35,2
7.25,14.5,3
6.1,18.25,3
8.85,20.5,3
7.55,22.25,3
15.8,19.2,3
16.2,18.0,3
16.95,17.45,3
17.35,18.2,3
18.25,17.45,3
17.85,17.15,3
18.25,16.25,3
19.75,16.15,3
20.85,16.95,3
21.8,17.95,3
22.75,19.0,3
24.2,19.15,3
25.05,18.55,3
25.25,17.45,3
13.95,11.1,3
15.7,10.0,3
14.3,9.85,3
14.3,10.65,3
14.85,11.1,3
10.9,7.9,3
9.1,8.5,3
11.55,16.5,2
11.05,18.5,2
20.4,7.5,1
19.95,8.7,1
27.55,2.05,4
26.6,2.5,4
27.05,3.65,4
28.65,4.1,4
30.6,11.8,4
29.55,12.35,4
29.0,13.05,4
30.5,13.4,4
31.6,14.05,4
31.55,14.9,4
28.6,17.1,4
30.35,17.75,4
34.7,12.25,4
31.0,11.75,4
16.0,12.7,4
8.8,17.95,4
14.45,3.7,4
28.6,4.75,4
29.7,10.85,4
24.15,14.2,4
14.85,12.0,4
6.9,11.05,4
```