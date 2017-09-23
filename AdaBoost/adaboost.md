#机器学习算法之AdaBoost算法python实现

##一. 理论基础

结合后面代码看理论基础，将会更加清楚。

####1. 算法描述

 AdaBoost算法是boosting方法的代表性算法。给定一个训练集，adaboost求一系列比较粗糙的分类器（弱分类器），每学完一个弱分类器，改变训练数据的概率分布（权值分布），使正确分类的数据权值减小，错误分类的数据权值增大，最后组合这些弱分类器，构成一个强分类器。

####2. 算法步骤
1. 初始化权值分布

$$D_1 = (w_{11},w_{12},\ldots, w_{1N}), \ \ \ \ \ w_{1i}=\frac{1}{N}$$

2. 迭代训练弱分类器，对每个分类训练步骤：
    a. 使用具有权值分布$D_m$的训练数据集学习，得到基本分类器

    $$G_m(x):\chi \rightarrow{-1, 1}$$

    b. 计算$G_m(x)$在训练数据集上的分类误差率

    $$e_m = \sum_{i=1}^{N}{w_{mi}I(y_i \ne G_m(x_i))}$$

    c. 计算分类器$G_m(x)$权值

    $$\alpha_m = \frac{1}{2}\ln\frac{1-e_m}{e_m}$$

    d. 更新训练数据的权值，为下一步迭代做准备

    $$w_{m+1,i} = \frac{w_{mi}}{Z_m} e^{-\alpha_m y_i G_m(x_i)}$$

    其中，
    $$Z_m = \sum_{i=1}^{N}{e^{-\alpha_m y_i G_m(x_i)}}$$

3. 当错误率（这里的错误率是指前m个弱分类器的线性组合后的分类器的错误率）达到阈值或者迭代次数（弱分类器个数）达到指定次数后，将所有的弱分类器线性组合起来

$$G(x) = sign(f(x)) = sign(\sum_{m=1}^{M}{\alpha_m G_m(x) })$$

####3. 训练误差分析

最终分类器的误差率满足：

$$\frac{1}{N}\sum_{i=1}^{N}I(G(x) \ne y_i) \le \frac{1}{N}\sum_{i=1}^{N}e^{-y_i f(x_i)} = \prod_{i=1}^{m}Z_m$$

这个定理说明，每一轮选取适当的$G_m$使得$Z_m$最小，从而使训练误差下降最快。

上述定理证明如下：

当$G(x) \ne y_i$时， $y_i f(x_i) < 0$：
$$e^{-y_i f(x_i)} > I(G(x) \ne y_i)  =1$$

当$G(x) = y_i$时， $y_i f(x_i) > 0$：
$$1 > e^{-y_i f(x_i)} > I(G(x) \ne y_i)  =0$$

所以定理左半部分不等式成立.

下面的证明会用到公式(在算法步骤中出现过)：

$$Z_m w_{m+1,i} = w_{mi} e^{-\alpha_m y_i G_m(x_i)}$$

右边等式部分证明如下：

$$
\begin{aligned}
\frac{1}{N}\sum_{i=1}^{N}e^{-y_i f(x_i)} 
&=\frac{1}{N}\sum_{i=1}^{N}e^{-y_i {\sum_{m=1}^{M}{[\alpha_m G_m(x_i)]}}}\\
&=\sum_{i=1}^{N}w_{1i}\prod_{m=1}^{M}{e^{-y_i \alpha_m G_m(x_i)}} \\
&=Z_1\sum_{i=1}^{N}w_{2i}\prod_{m=2}^{M}{e^{-y_i \alpha_m G_m(x_i)}} \\
&=Z_1Z_2\sum_{i=1}^{N}w_{3i}\prod_{m=3}^{M}{e^{-y_i \alpha_m G_m(x_i)}} \\
&=\ldots \\
&=\prod_{m=1}^{M}{Z_m}
\end{aligned}$$

所以定理右半部分成立.

####4. 算法的理论解释与推导

另一种解释认为，AdaBoost是模型为加法模型，损失函数为指数函数，学习算法为前向分步算法时的二类分类学习方法。

#####1. 前向分步算法

加法模型：

$$f(x) = \sum_{m=1}^{M}{\beta_m b(x;\gamma_m)}$$

其中， $b(x;\gamma_m)$为基函数，$\gamma_m$为基函数的参数，$\beta_m$为基函数的系数.

损失函数：$L(x,f(x))$

目标：

$$\min_{\beta_m,\gamma_m}\sum_{i=1}^{N}{L(y_i,\sum_{m=1}^{M}{\beta_m b(x_i;\gamma_m)})}$$

通常这是一个复杂的优化问题。前向分步算法求解这一优化问题的想法是：因为学习的是加法模型，如果能够从前向后，每一步只学习一个基函数及其系数，逐步逼近优化目标函数式，那么就可以简化优化的复杂度.具体地，每步只需优化如下损失函数:

$$\min_{\beta,\gamma}\sum_{i=1}^{N}L(y_i,\beta b(x_i;\gamma))$$

#####2. 前向分步算法步骤

1. 初始化$f_0(x)=0$
2. 对$m=1,2,\cdots,M$
a. 极小化损失函数
$$(\beta_m, \gamma_m)=arg\min_{\beta, \gamma}{\sum_{i=1}^{N}L(y_i, f_{m-1}(x-i)+\beta b(x_i;\gamma))}$$
得到参数$\beta_m, \gamma_m$
b. 更新
$$f_m(x) = f_{m-1}(x)+\beta_m b(x;\gamma_m)$$
3. 得到加法模型
$$f(x)=f_M(x)=\sum_{m=1}^{M}{\beta_m b(x;\gamma_m)}$$

这样，前向分步算法将同时求解从m=1到M所有参数$\beta_m, \gamma_m$的优化问题简化为逐次求解各个$\beta_m, \gamma_m$的优化问题.

#####3. 前向分步算法与AdaBoost

AdaBoost算法是前向分步加法算法的特例。这时，模型是由基本分类器组成的加法模型，损失函数是指数函数。

当基函数为基本分类器，基函数的系数为基本分类器的权值时，该加法模型等价于AdaBoost的最终分类器
$$f(x)=\sum_{m=1}^{M}{\alpha_m G_m{x}}$$

损失函数为指数函数：
$$L(y,f(x))=e^{-yf(x)}$$

假设经过m-1轮迭代前向分步算法已经得到$f_{m-1}(x)$，在第m轮迭代得到$\alpha_m,G_m(x),f_m(x)$.
$$f_m(x)=f_{m-1}(x)+\alpha_mG_m(x)$$
目标是使前向分步算法得到的$\alpha_m,G_m(x)$使$f_m(x)$在训练数据集T上的指数损失最小，即
$$
\begin{aligned}
(\alpha_m,G_m(x))&=arg\min_{\alpha,G}\sum_{i=1}^{N}e^{-y_i (f_{m-1}(x)+\alpha G(x_i))}\\
&=arg\min_{\alpha,G}\sum_{i=1}^{N}{\bar{\omega}_{mi}e^{-y_i \alpha G(x_i)}}
\end{aligned}
$$

其中，$\bar{\omega}_{mi}=e^{-y_if_{m-1}{(x_i)}}$，因为$\bar{\omega}_{mi}$既不依赖$\alpha$也不依赖于G，所以与最小化无关，但$\bar{\omega}_{mi}$依赖于$f_{m-1}(x)$，随着每一轮迭代而发生变化.

现证使上式达到最小的$\alpha^*_m,G_m^*(x)$就是AdaBoost算法所得到的$\alpha_m$和$G_m(x)$.

首先，求$G^*_m(x)$.对任意$\alpha > 0$，使上式最小的$G(x)$由下式得到:

$$G^*_m(x)=arg\min_{G}{\sum_{i=1}^{N}\bar{\omega}_{mi}I(y_i \ne G(x_i))}$$

此分类器$G^*_m(x)$即为AdaBoost算法的基本分类器$G_m(x)$，因为它是使第ｍ轮加权训练数据分类错误率最小的基本分类器. 

之后，求$\alpha^*_m$.
<!-- $$arg\min_{\alpha,G}\sum_{i=1}^{N}{\bar{\omega}_{mi}e^{-y_i \alpha G(x_i)}}=\sum_{y_i=G_m(x_i)}{\bar{\omega}_{mi}e^{-\alpha}} + \sum_{y_i \ne G_m(x_i)}{\bar{\omega}_{mi}e^{\alpha}}
=(e^\alpha - e^{-\alpha})\sum_{i=1}^{N}{\bar{\omega}_{mi}I(y_i \ne G(x_i))} + e^{-\alpha}\sum_{i=1}^{N}\bar{\omega}_{mi}
=(e^{\alpha} - e^{-\alpha})e_m + e^{-\alpha}$$ -->

$$
\begin{aligned} \\
arg\min_{\alpha,G}\sum_{i=1}^{N}{\bar{\omega}_{mi}e^{-y_i \alpha G(x_i)}} & =\sum_{y_i=G_m(x_i)}{\bar{\omega}_{mi}e^{-\alpha}} + \sum_{y_i \ne G_m(x_i)}{\bar{\omega}_{mi}e^{\alpha}} \\
& =(e^\alpha - e^{-\alpha})\sum_{i=1}^{N}{\bar{\omega}_{mi}I(y_i \ne G(x_i))} + e^{-\alpha}\sum_{i=1}^{N}\bar{\omega}_{mi} \\
& =(e^{\alpha} - e^{-\alpha})e_m + e^{-\alpha}
\end{aligned}
$$


将上式对$\alpha$求导并令其等于０，求得

$$\alpha^*_m=\frac{1}{2}{\ln{\frac{1-e_m}{e_m}}}$$

$e_m$为分类错误率.
这里的$\alpha^*_m$与AdaBoost算法第二中的$\alpha_m$完全一致.

最后来看每一轮样本权值的更新，由

$$f_m(x)=f_{m-1}(x)+\alpha_mG_m(x)$$

以及$\bar{\omega}_{mi}=e^{-y_if_{m-1}{(x_i)}}$，可得

$$\bar{\omega}_{m+1,i}=\bar{\omega}_{mi}e^{-y_i \alpha_m G_m(x)}$$

这与AdaBoost算法第2步中的样本权值更新只差规范化因子，因而等价.

##二. python实现

####1. 代码
这里的弱分类器是单层的决策树。
```python
#encoding=utf-8
######################################################################
#Copyright: CNIC
#Author: LiuYao
#Date: 2017-9-11
#Description: implements the adaboost algorithm
######################################################################
'''
implements the adaboost
'''

import numpy as np
import matplotlib.pyplot as plt

class AdaBoost:
    '''
    implements the adaboost classifier
    '''

    def __init__(self):
        pass

    def load_simple_data(self):
        '''
        make a simple data set
        '''
        data = np.mat([[1.0, 2.0],
                    [2.0, 1.1],
                    [1.3, 1.0],
                    [1.0, 1.0],
                    [2.0, 1.0]])
        labels = [1.0, 1.0, -1.0, -1.0, 1.0]
        return data, labels

    def classify_results(self, x_train, demension, thresh, op):
        '''
        get the predict results by the data, thresh, op and the special demension
        Args:
            x_train: train data
            demension: the special demension
            thresh: the spliting value
            op: the operator, including '<=', '>'
        '''
        y_predict = np.ones((x_train.shape[0], 1))
        if op == 'le':
            y_predict[x_train[:, demension] <= thresh] = -1.0
        else:
            y_predict[x_train[:, demension] > thresh] = -1.0
        return y_predict

    def get_basic_classifier(self, x_train, y_train, D):
        '''
        generate basic classifier by the data and the weight of data
        Args:
            x_train: train data
            y_train: train label
            D: the weight of the data
        '''
        x_mat = np.mat(x_train)
        y_mat = np.mat(y_train).T
        D_mat = np.mat(D)
        [m,n] = x_mat.shape
        num_steps = 10.0
        min_error = np.inf
        best_basic_classifier = {}
        best_predict = np.mat(np.zeros((m, 1)))
        #traverse all demensions to find best demension
        for demension in range(n):
            step_length = (x_mat[:, demension].max() - x_mat[:, demension].min()) / num_steps
            #traverse all spliting range in the special demension to find best spliting value
            for step in range(-1, int(num_steps) + 1):
                #determine which op has lower error
                for op in ['le', 'g']:
                    thresh = x_mat[:, demension].min() + step * step_length
                    y_predict = self.classify_results(x_mat, demension, thresh, op)
                    error = np.sum(D_mat[np.mat(y_predict) != y_mat])
                    if error < min_error:
                        min_error = error
                        best_predict = np.mat(y_predict).copy()
                        best_basic_classifier['demension'] = demension
                        best_basic_classifier['thresh'] = thresh
                        best_basic_classifier['op'] = op
        return best_basic_classifier, min_error, best_predict


    def train(self, x_train, y_train, max_itr=50):
        '''
        train function
        '''
        m = len(x_train)
        n = len(x_train[0])
        D = [1.0/m for i in range(m)]
        D = np.mat(D).T
        self.basic_classifier_list = []
        acc_label = np.mat(np.zeros((m, 1)))
        #generate each basic classifier
        for i in range(max_itr):
            #generate basic classifier
            basic_classifier, error, y_predict = self.get_basic_classifier(x_train, y_train, D)
            print 'y_predict:', y_predict.T
            #compute the basic classifier weight
            alpha = 0.5 * np.log((1 - error) / max(error, 1e-16))
            #compute the data weight
            D = np.multiply(D, np.exp(-1 * alpha * np.multiply(np.mat(y_train).T, np.mat(y_predict))))
            D = D / D.sum()
            print 'D:', D.T
            basic_classifier['alpha'] = alpha
            #store the basic classifier
            self.basic_classifier_list.append(basic_classifier)
            #accmulate the predict results
            acc_label += alpha * y_predict
            print 'acc_label', acc_label
            #compute the total error of all basic classifier generated until now
            total_error = np.sum(np.sign(acc_label) != np.mat(y_train).T) / float(m)
            print 'total_error:', total_error

            #if total error equals to the thresh, then stop
            if total_error == 0.0: 
                break
        return self.basic_classifier_list
        
    def predict(self, x_test):
        '''
        adaboost predict function
        '''
        x_mat = np.mat(x_test)
        m = x_mat.shape[0]
        acc_label = np.mat(np.zeros((m, 1)))
        for i in range(len(self.basic_classifier_list)):
            predict = self.classify_results(x_mat, 
                                self.basic_classifier_list[i]['demension'],
                                self.basic_classifier_list[i]['thresh'],
                                self.basic_classifier_list[i]['op'])
            # accmulate the predict results of each basic classifier
            acc_label += self.basic_classifier_list[i]['alpha'] * predict
        print acc_label
        return np.sign(acc_label)

def main():
    adaboost = AdaBoost()
    data, labels = adaboost.load_simple_data()
    adaboost.train(data, labels, max_itr=9)
    print adaboost.predict([[5,5], [0,0]])

if __name__ == '__main__':
    main()
```

####2. 结果
结果用来验证实现的adaboost算法是否正确。
```
y_predict: [[-1.  1. -1. -1.  1.]]
D: [[ 0.5    0.125  0.125  0.125  0.125]]
acc_label [[-0.69314718]
 [ 0.69314718]
 [-0.69314718]
 [-0.69314718]
 [ 0.69314718]]
total_error: 0.2
y_predict: [[ 1.  1. -1. -1. -1.]]
D: [[ 0.28571429  0.07142857  0.07142857  0.07142857  0.5       ]]
acc_label [[ 0.27980789]
 [ 1.66610226]
 [-1.66610226]
 [-1.66610226]
 [-0.27980789]]
total_error: 0.2
y_predict: [[ 1.  1.  1.  1.  1.]]
D: [[ 0.16666667  0.04166667  0.25        0.25        0.29166667]]
acc_label [[ 1.17568763]
 [ 2.56198199]
 [-0.77022252]
 [-0.77022252]
 [ 0.61607184]]
total_error: 0.0
[[ 2.56198199]
 [-2.56198199]]
[[ 1.]
 [-1.]]
```