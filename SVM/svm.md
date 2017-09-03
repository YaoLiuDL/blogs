#机器学习之支持向量机python实现

*前言：*纸上得来终觉浅，绝知此事要躬行。

[TOC]

##一. 理论基础
####1. 函数间隔与几何间隔
函数间隔: $\hat{\gamma}=y_i(w*x_i+b)$
几何间隔: $\gamma=y_i(\frac{w*x_i+b}{\mid\mid{w}\mid\mid})$
####2. 优化目标：

$$\min_{\alpha}\ \ \ \ \ \ \frac{1}{2}\sum_{i=0}^{m}\sum_{j=0}^{m}{\alpha_i\alpha_jy_iy_jx_ix_j}-\sum_{i=0}^{m}\alpha_i$$
$$s.t.\ \ \ \sum_{i=0}^{m}\alpha_iy_i=0$$
$$s.t.\ \ \ 0\le\alpha_i\le{C},\ \ \ (i=1,2,\cdots,m)$$

####3. 推导过程：
我们的目标是最大化距离边界最近的点的几何间隔，可以形式化
$$\max_{w,b}\gamma$$
$$s.t.\ \ \ \ \ \ \ \ y_i(\frac{w*x_i+b}{\mid\mid{w}\mid\mid})\ge\gamma\ \ \ ,\ \ \ (i=1,2,\ldots,m)$$

根据几何间隔与函数间隔的关系
$$\max_{w,b}\frac{\hat{\gamma}}{\mid\mid{w}\mid\mid}$$
$$s.t.\ \ \ \ \ \ \ \ y_i(w*x_i+b)\ge\hat{\gamma}\ \ \ ,\ \ \ (i=1,2,\ldots,m)$$

由于$\hat{\gamma}$与w，b同步变化，所以$\gamma$的缩放对上述约束条件没有影响，令$\hat{\gamma}=1$
$$\max_{w,b}\frac{1}{\mid\mid{w}\mid\mid}$$
$$s.t.\ \ \ \ \ \ \ \ y_i(w*x_i+b)\ge1\ \ \ ,\ \ \ (i=1,2,\ldots,m)$$

最大化$\frac{1}{\mid\mid{w}\mid\mid}$与最小化$\frac{1}{2}\mid\mid{w}\mid\mid^2$等价
$$\min_{w,b}\frac{1}{2}\mid\mid{w}\mid\mid^2$$
$$s.t.\ \ \ \ \ \ \ \ y_i(w*x_i+b)\ge1\ \ \ ,\ \ \ (i=1,2,\ldots,m)$$

加上软间隔，适用于训练数据线性不可分情形
$$\min_{w,b,\xi}\frac{1}{2}\mid\mid{w}\mid\mid^2 + C\sum_{i=1}^{m}{\xi_i}$$
$$s.t.\ \ \ \ \ \ \ \ y_i(w*x_i+b)\ge{1-\xi_i}\ \ \ ,\ \ \ (i=1,2,\ldots,m)$$
$$\xi_i\ge0\ \ \ ,\ \ \ (i=1,2,\ldots,m)$$

这是一个凸优化问题，应用拉格朗日对偶性([参考](http://www.cnblogs.com/90zeng/p/Lagrange_duality.html))，通过求解对偶问题得到原始问题的最优解。这样做的优点：
* 对偶问题往往更容易求解；
* 对偶形式可以很自然的引入核函数，进而推广到非线性分类问题；

首先建立拉格朗日函数：
$$L(w,b,\xi,\alpha,\beta)=\frac{1}{2}\mid\mid{w}\mid\mid^2 + C\sum_{i=1}^{m}{\xi_i}+\sum_{i=0}^{m}\alpha_i[{1-\xi_i-y_i(w*x_i+b)}]-\sum_{i=0}^{m}\beta\xi_i$$

根据拉格朗日的对偶性，原始问题的对偶问题是极大极小问题：
$$\max_{\alpha,\beta}\min_{w,b,\xi}L(w,b,\xi,\alpha,\beta)$$
1. 求$\min_{w,b,\xi}L(w,b,\xi,\alpha,\beta)$
对$w,b,\xi$求偏导数并令其等于0
$$\nabla_wL(w,b,\xi,\alpha,\beta)=w-\sum_{i=0}^{m}{\alpha_iy_ix_i}=0$$
$$\nabla_bL(w,b,\xi,\alpha,\beta)=-\sum_{i=0}^{m}{\alpha_iy_i}=0$$
$$\nabla_{\xi_i}L(w,b,\xi,\alpha,\beta)=C-\alpha_i-\beta_i=0$$

得

$$w=\sum_{i=0}^{m}{\alpha_iy_ix_i}$$
$$\sum_{i=0}^{m}{\alpha_iy_i}=0$$
$$C=\alpha_i+\beta_i$$

代入拉格朗日函数得
$$L(w,b,\xi,\alpha,\beta)=-\frac{1}{2}\sum_{i=0}^{m}\sum_{j=0}^{m}{\alpha_i\alpha_jy_iy_jx_ix_j}+\sum_{i=0}^{m}\alpha_i$$

2. 求$\max_{\alpha}L(w,b,\xi,\alpha,\beta)$

$$\max_{\alpha}\ \ \ \ \ \ -\frac{1}{2}\sum_{i=0}^{m}\sum_{j=0}^{m}{\alpha_i\alpha_jy_iy_jx_ix_j}+\sum_{i=0}^{m}\alpha_i$$
$$s.t.\ \ \ \sum_{i=0}^{m}\alpha_iy_i=0$$
$$s.t.\ \ \ \alpha_i\ge0,\ \ \ (i=1,2,\cdots,m)$$

等价于
$$\in_{\alpha}\ \ \ \ \ \ \frac{1}{2}\sum_{i=0}^{m}\sum_{j=0}^{m}{\alpha_i\alpha_jy_iy_jx_ix_j}-\sum_{i=0}^{m}\alpha_i$$
$$s.t.\ \ \ \sum_{i=0}^{m}\alpha_iy_i=0$$
$$s.t.\ \ \ \alpha_i\ge0,\ \ \ (i=1,2,\cdots,m)$$

####4. 求解
求出上述优化问题的解，根据KKT条件就能得到原始问题的解。
设$\alpha^*$是对偶问题的解，可以根据KKT条件，得
$$\nabla_wL(w^*,b^*,\xi^*,\alpha^*,\beta^*)=w^*-\sum_{i=0}^{m}{\alpha_i^*y_ix_i}=0$$
$$\nabla_bL(w^*,b^*,\xi^*,\alpha^*,\beta^*)=-\sum_{i=0}^{m}{\alpha_i^*y_i}=0$$
$$\nabla_{\xi_i}L(w^*,b^*,\xi^*,\alpha^*,\beta^*)=C-\alpha_i^*-\beta_i^*=0$$
$$\alpha_i^*({1-\xi_i^*-y_i(w^**x_i+b^*)})=0$$
$$\beta_i^*\xi_i^*=0$$
$$\xi_i^*\ge0$$
$$\alpha_i^*\ge0$$
$$\beta_i^*\ge0$$
则至少存在一个$\alpha^*_j>0$（反证法，如果$\alpha^*=0$，则可得$w*=0$， 而$w*=0$不是原始最优化问题的解，产生矛盾），对此$j$由上述公式推导得
$$y_j(w^**x_j+b^*)-1=0$$
因为$y_j^2=1$，将上式两边同乘以$y_j$，则得
$$b^*=y_j-\sum_{i=0}^{m}\alpha^*_iy_i(x_ix_j)$$
同时，
$$w^*=\sum_{i=0}^{m}{\alpha_i^*y_i^*x_i^*}$$
即得到原问题的优化解。
####5.支持向量
在线性不可分的情况下，将对偶问题的解$\alpha^*$中对应于$\alpha_j>0$的样本点的实例$x_i$称为支持向量，软间隔的支持向量可以在间隔边界上，或者在间隔边界与分离超平面之间，或者在分离超平面误分一侧.由KKT得到的公式可以推得以下结论：
如果$0<\alpha_i<C$，则$\xi_i=0$，支持向量恰好落在间隔边界上；
如果$\alpha^*=C, 0<\xi_i<1$，则分类正确，支持向量在间隔边界与分离超平面之间；
如果$\alpha^*=C, \xi_i=1$，则支持向量在分离超平面上；
如果$\alpha^*=C, \xi_i>1$，则支持向量位于分离超平面误分一侧.
####6. SMO(序列最小最优化算法)
优化目标为：
$$\min_{\alpha}\ \ \ \ \ \ \frac{1}{2}\sum_{i=0}^{m}\sum_{j=0}^{m}{\alpha_i\alpha_jy_iy_jx_ix_j}-\sum_{i=0}^{m}\alpha_i$$
$$s.t.\ \ \ \sum_{i=0}^{m}\alpha_iy_i=0$$
$$\ \ \ \ \ \ \ \ \ \ \ \ \ \ 0\le\alpha_i\le{C},\ \ \ (i=1,2,\cdots,m)$$
基本思路：如果所有变量的解都满足此最优化问题的KKT条件，那么这个最优化问题的解就得到了，因为KKT条件是该最优化问题的充分必要条件。否则，选择两个变量，固定其他变量，针对这两个变量构建一个二次规划问题，这个二次规划问题关于这两个变量的解应该更接近原始二次规划问题的解，因为这会使得原始二次规划问题的目标函数值变得更小。重要的是，这时子问题可以通过解析方法求解，这样就可以大大提高整个算法的计算速度。子问题有两个变量，一个是违反KKT条件最严重的那一个，另一个由约束条件自动确定。如此，SMO算法将原问题不断分解为子问题并对子问题求解，进而达到求解原问题的目的。([参考](http://www.cnblogs.com/jerrylead/archive/2011/03/18/1988419.html))

算法步骤：
1. 取初值$\alpha^{(0)}=0$，令k=0，k为迭代轮次;
2. 选取优化变量$\alpha^{k}_1,\alpha^{k}_2$，解析求解两个变量的最优化问题，求得最优解$\alpha^{k+1}_1,\alpha^{k+1}_2$，更新$\alpha$为$\alpha^{k+1}$;
3. 若在精度$\epsilon$范围内满足停机条件
$$\sum_{i=0}^{m}{\alpha_i y_i} = 0$$
$$0 \le \alpha_i \le C$$
$$y_i g(x_i) = \begin{cases}
               \ge 1,& \{ x_i | \alpha_i = 0 \} \\
               = 1,& \{ x_i | 0 \le \alpha_i \le C \} \\
               \le 1, & \{ x_i | \alpha_i = C \}
               \end{cases}$$
其中， $g(x_i) = \sum_{j=0}^{m} {\alpha_j y_j K(x_j,x_i) + b}$
则转第4步；否则令k=k+1，转第2步；
4. 取$\hat{\alpha}=\alpha^{(k+1)}$.

####7. 核技巧
对于输入空间中的非线性分类问题，可以通过非线性变换将它转化为某个高维特征空间中的线性分类问题，在高维特征空间中学习线性支持向量机。由于在线性支持向量机学习的对偶问题里，目标函数和分类决策函数都只涉及实例与实例之间的内积，所以不要显示地指定非线性变换，而是用核函数来替换当中的内积。核函数表示，通过一个非线性转换后的两个实例间的内积。具体的，$K(x,z)$是一个核函数，或者正定核，以为着存在一个从输入空间$\chi$到特征空间$H$的映射$\phi(x):\chi\rightarrow H$，对任意$x,z\in\chi$，有
$$K(x,z)=\phi(x)*\phi(z)$$
对称函数$K(x,z)$为正定核的充要条件如下：对任意$x_i\in\chi$，任意正整数m，对称函数$K(x,z)$对应的Gram矩阵是半正定的。
所以，在线性支持向量机学习的对偶问题中，用核函数$K(x,z)$替代内积，求解得到的就是非线性支持向量机
$$f(x)=sign(\sum_{i=0}^{m}{\alpha^*_iy_iK(x,z)}+b^*)$$ 

##二. python实现（[引自](http://blog.csdn.net/zouxy09/article/details/17292011)）
```python
from numpy import *
import time
import matplotlib.pyplot as plt


# calulate kernel value
def calcKernelValue(matrix_x, sample_x, kernelOption):
    kernelType = kernelOption[0]
    numSamples = matrix_x.shape[0]
    kernelValue = mat(zeros((numSamples, 1)))

    if kernelType == 'linear':
        kernelValue = matrix_x * sample_x.T
    elif kernelType == 'rbf':
        sigma = kernelOption[1]
        if sigma == 0:
            sigma = 1.0
        for i in xrange(numSamples):
            diff = matrix_x[i, :] - sample_x
            kernelValue[i] = exp(diff * diff.T / (-2.0 * sigma**2))
    else:
        raise NameError('Not support kernel type! You can use linear or rbf!')
    return kernelValue


# calculate kernel matrix given train set and kernel type
def calcKernelMatrix(train_x, kernelOption):
    numSamples = train_x.shape[0]
    kernelMatrix = mat(zeros((numSamples, numSamples)))
    for i in xrange(numSamples):
        kernelMatrix[:, i] = calcKernelValue(
            train_x, train_x[i, :], kernelOption)
    return kernelMatrix


# define a struct just for storing variables and data
class SVMStruct:
    def __init__(self, dataSet, labels, C, toler, kernelOption):
        self.train_x = dataSet  # each row stands for a sample
        self.train_y = labels  # corresponding label
        self.C = C             # slack variable
        self.toler = toler     # termination condition for iteration
        self.numSamples = dataSet.shape[0]  # number of samples
        # Lagrange factors for all samples
        self.alphas = mat(zeros((self.numSamples, 1)))
        self.b = 0
        self.errorCache = mat(zeros((self.numSamples, 2)))
        self.kernelOpt = kernelOption
        self.kernelMat = calcKernelMatrix(self.train_x, self.kernelOpt)


# calculate the error for alpha k
def calcError(svm, alpha_k):
    output_k = float(multiply(svm.alphas, svm.train_y).T *
                     svm.kernelMat[:, alpha_k] + svm.b)
    error_k = output_k - float(svm.train_y[alpha_k])
    return error_k


# update the error cache for alpha k after optimize alpha k
def updateError(svm, alpha_k):
    error = calcError(svm, alpha_k)
    svm.errorCache[alpha_k] = [1, error]


# select alpha j which has the biggest step
def selectAlpha_j(svm, alpha_i, error_i):
    svm.errorCache[alpha_i] = [1, error_i]  # mark as valid(has been optimized)
    candidateAlphaList = nonzero(svm.errorCache[:, 0].A)[
        0]  # mat.A return array
    maxStep = 0
    alpha_j = 0
    error_j = 0

    # find the alpha with max iterative step
    if len(candidateAlphaList) > 1:
        for alpha_k in candidateAlphaList:
            if alpha_k == alpha_i:
                continue
            error_k = calcError(svm, alpha_k)
            if abs(error_k - error_i) > maxStep:
                maxStep = abs(error_k - error_i)
                alpha_j = alpha_k
                error_j = error_k
    # if came in this loop first time, we select alpha j randomly
    else:
        alpha_j = alpha_i
        while alpha_j == alpha_i:
            alpha_j = int(random.uniform(0, svm.numSamples))
        error_j = calcError(svm, alpha_j)

    return alpha_j, error_j


# the inner loop for optimizing alpha i and alpha j
def innerLoop(svm, alpha_i):
    error_i = calcError(svm, alpha_i)

    # check and pick up the alpha who violates the KKT condition
    # satisfy KKT condition
    # 1) yi*f(i) >= 1 and alpha == 0 (outside the boundary)
    # 2) yi*f(i) == 1 and 0<alpha< C (on the boundary)
    # 3) yi*f(i) <= 1 and alpha == C (between the boundary)
    # violate KKT condition
    # because y[i]*E_i = y[i]*f(i) - y[i]^2 = y[i]*f(i) - 1, so
    # 1) if y[i]*E_i < 0, so yi*f(i) < 1, if alpha < C, violate!(alpha = C will be correct)
    # 2) if y[i]*E_i > 0, so yi*f(i) > 1, if alpha > 0, violate!(alpha = 0 will be correct)
    # 3) if y[i]*E_i = 0, so yi*f(i) = 1, it is on the boundary, needless optimized
    if (svm.train_y[alpha_i] * error_i < -svm.toler) and (svm.alphas[alpha_i] < svm.C) or\
            (svm.train_y[alpha_i] * error_i > svm.toler) and (svm.alphas[alpha_i] > 0):

        # step 1: select alpha j
        alpha_j, error_j = selectAlpha_j(svm, alpha_i, error_i)
        alpha_i_old = svm.alphas[alpha_i].copy()
        alpha_j_old = svm.alphas[alpha_j].copy()

        # step 2: calculate the boundary L and H for alpha j
        if svm.train_y[alpha_i] != svm.train_y[alpha_j]:
            L = max(0, svm.alphas[alpha_j] - svm.alphas[alpha_i])
            H = min(svm.C, svm.C + svm.alphas[alpha_j] - svm.alphas[alpha_i])
        else:
            L = max(0, svm.alphas[alpha_j] + svm.alphas[alpha_i] - svm.C)
            H = min(svm.C, svm.alphas[alpha_j] + svm.alphas[alpha_i])
        if L == H:
            return 0

        # step 3: calculate eta (the similarity of sample i and j)
        eta = 2.0 * svm.kernelMat[alpha_i, alpha_j] - svm.kernelMat[alpha_i, alpha_i] \
            - svm.kernelMat[alpha_j, alpha_j]
        if eta >= 0:
            return 0

        # step 4: update alpha j
        svm.alphas[alpha_j] -= svm.train_y[alpha_j] * (error_i - error_j) / eta

        # step 5: clip alpha j
        if svm.alphas[alpha_j] > H:
            svm.alphas[alpha_j] = H
        if svm.alphas[alpha_j] < L:
            svm.alphas[alpha_j] = L

        # step 6: if alpha j not moving enough, just return
        if abs(alpha_j_old - svm.alphas[alpha_j]) < 0.00001:
            updateError(svm, alpha_j)
            return 0

        # step 7: update alpha i after optimizing aipha j
        svm.alphas[alpha_i] += svm.train_y[alpha_i] * svm.train_y[alpha_j] \
            * (alpha_j_old - svm.alphas[alpha_j])

        # step 8: update threshold b
        b1 = svm.b - error_i - svm.train_y[alpha_i] * (svm.alphas[alpha_i] - alpha_i_old) \
            * svm.kernelMat[alpha_i, alpha_i] \
            - svm.train_y[alpha_j] * (svm.alphas[alpha_j] - alpha_j_old) \
            * svm.kernelMat[alpha_i, alpha_j]
        b2 = svm.b - error_j - svm.train_y[alpha_i] * (svm.alphas[alpha_i] - alpha_i_old) \
            * svm.kernelMat[alpha_i, alpha_j] \
            - svm.train_y[alpha_j] * (svm.alphas[alpha_j] - alpha_j_old) \
            * svm.kernelMat[alpha_j, alpha_j]
        if (0 < svm.alphas[alpha_i]) and (svm.alphas[alpha_i] < svm.C):
            svm.b = b1
        elif (0 < svm.alphas[alpha_j]) and (svm.alphas[alpha_j] < svm.C):
            svm.b = b2
        else:
            svm.b = (b1 + b2) / 2.0

        # step 9: update error cache for alpha i, j after optimize alpha i, j and b
        updateError(svm, alpha_j)
        updateError(svm, alpha_i)

        return 1
    else:
        return 0


# the main training procedure
def trainSVM(train_x, train_y, C, toler, maxIter, kernelOption=('rbf', 1.0)):
    # calculate training time
    startTime = time.time()

    # init data struct for svm
    svm = SVMStruct(mat(train_x), mat(train_y), C, toler, kernelOption)

    # start training
    entireSet = True
    alphaPairsChanged = 0
    iterCount = 0
    # Iteration termination condition:
    # 	Condition 1: reach max iteration
    # 	Condition 2: no alpha changed after going through all samples,
    # 				 in other words, all alpha (samples) fit KKT condition
    while (iterCount < maxIter) and ((alphaPairsChanged > 0) or entireSet):
        alphaPairsChanged = 0

        # update alphas over all training examples
        if entireSet:
            for i in xrange(svm.numSamples):
                alphaPairsChanged += innerLoop(svm, i)
            print '---iter:%d entire set, alpha pairs changed:%d' % (iterCount, alphaPairsChanged)
            iterCount += 1
        # update alphas over examples where alpha is not 0 & not C (not on boundary)
        else:
            nonBoundAlphasList = nonzero(
                (svm.alphas.A > 0) * (svm.alphas.A < svm.C))[0]
            for i in nonBoundAlphasList:
                alphaPairsChanged += innerLoop(svm, i)
            print '---iter:%d non boundary, alpha pairs changed:%d' % (iterCount, alphaPairsChanged)
            iterCount += 1

        # alternate loop over all examples and non-boundary examples
        if entireSet:
            entireSet = False
        elif alphaPairsChanged == 0:
            entireSet = True

    print 'Congratulations, training complete! Took %fs!' % (time.time() - startTime)
    return svm


# testing your trained svm model given test set
def testSVM(svm, test_x, test_y):
    test_x = mat(test_x)
    test_y = mat(test_y)
    numTestSamples = test_x.shape[0]
    supportVectorsIndex = nonzero(svm.alphas.A > 0)[0]
    supportVectors = svm.train_x[supportVectorsIndex]
    supportVectorLabels = svm.train_y[supportVectorsIndex]
    supportVectorAlphas = svm.alphas[supportVectorsIndex]
    matchCount = 0
    for i in xrange(numTestSamples):
        kernelValue = calcKernelValue(
            supportVectors, test_x[i, :], svm.kernelOpt)
        predict = kernelValue.T * \
            multiply(supportVectorLabels, supportVectorAlphas) + svm.b
        if sign(predict) == sign(test_y[i]):
            matchCount += 1
    accuracy = float(matchCount) / numTestSamples
    return accuracy


# show your trained svm model only available with 2-D data
def showSVM(svm):
    if svm.train_x.shape[1] != 2:
        print "Sorry! I can not draw because the dimension of your data is not 2!"
        return 1

    # draw all samples
    for i in xrange(svm.numSamples):
        if svm.train_y[i] == -1:
            plt.plot(svm.train_x[i, 0], svm.train_x[i, 1], 'or')
        elif svm.train_y[i] == 1:
            plt.plot(svm.train_x[i, 0], svm.train_x[i, 1], 'ob')

    # mark support vectors
    supportVectorsIndex = nonzero(svm.alphas.A > 0)[0]
    for i in supportVectorsIndex:
        plt.plot(svm.train_x[i, 0], svm.train_x[i, 1], 'oy')

    # draw the classify line
    w = zeros((2, 1))
    for i in supportVectorsIndex:
        w += multiply(svm.alphas[i] * svm.train_y[i], svm.train_x[i, :].T)
    min_x = min(svm.train_x[:, 0])[0, 0]
    max_x = max(svm.train_x[:, 0])[0, 0]
    y_min_x = float(-svm.b - w[0] * min_x) / w[1]
    y_max_x = float(-svm.b - w[0] * max_x) / w[1]
    plt.plot([min_x, max_x], [y_min_x, y_max_x], '-g')
    plt.show()
```



