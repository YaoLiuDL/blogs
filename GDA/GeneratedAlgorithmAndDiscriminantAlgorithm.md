#生成式算法与判别式算法

*前言：网易公开课机器学习 **第五课 生成学习算法**的观后感或者总结笔记*

##1. 区别

生成式算法：对p(x|y)和p(y)进行建模，也可以说是对p(x,y)进行建模，即求x，y的联合分布。比如GDA(Gaussian Discriminant Analysis)，有两个类别0和1，分别对p(x|y=0)和p(x|y=1)进行建模，还有p(y=0)和p(y=1)进行建模，最终$\hat{y} = \arg\max_{i}{p(x|y=i)p(y=i)}$。

判别式算法：对p(y|x)进行建模，相当于一个黑箱，给定一个数据集，直接根据数据集得到决策函数或规则。比如逻辑回归，直接对p(y|x)进行建模。

##2. 高斯判别分析

###1. 一元高斯分布

$$p(x;\mu,\sigma) = \frac{1}{\sqrt{2 \pi}\sigma} e^{-\frac{(x-\mu)^2} {2\sigma^2} }$$

###2. 多元高斯分布

$$p(x;\mu,\Sigma) = \frac{1}{(2 \pi)^{n/2}|\Sigma|^{1/2}}e^{-\frac{1}{2} (x-\mu)^T \Sigma^{-1} (x-\mu)}$$

其中$\mu$为均值列向量，$\Sigma$为协方差矩阵，记作$N(\mu, \Sigma)$

###3. 参数求解

模型：
$$y \sim Bernoulli(\phi)$$

$$x|y=0 \sim N(\mu_0, \Sigma)$$

$$x|y=1 \sim N(\mu_1, \Sigma)$$

分布：

$$p(y) = \phi^y (1-\phi)^{1-y}$$

$$p(x|y=0) = \frac{1}{(2 \pi)^{n/2}|\Sigma|^{1/2}}e^{-\frac{1}{2} (x-{\mu_0})^T \Sigma^{-1} (x-\mu_0)}$$

$$p(x|y=0) = \frac{1}{(2 \pi)^{n/2}|\Sigma|^{1/2}}e^{-\frac{1}{2} (x-\mu_1)^T \Sigma^{-1} (x-\mu_1)}$$

极大log似然函数：

$$
\begin{aligned}
L(\phi, \mu_0, \mu_1, \Sigma) &= log\prod_{i=1}^{m}p(x_i,y_i; \phi, \mu_0, \mu_1, \Sigma)\\
&=log\prod_{i=1}^{m}p(x_i|y_i; \mu_0, \mu_1, \Sigma)p(y_i; \phi)
\end{aligned}$$

求参：

$$\phi = \frac{\sum_{i=1}^{m}{y_i}}{m}$$

$$\mu_0 = \frac{\sum_{i=1}^{m}{x_iI(y_i = 0)}}{\sum_{i=1}^{m}I(y_i = 0)}$$

$$\mu_1 = \frac{\sum_{i=1}^{m}{x_iI(y_i = 1)}}{\sum_{i=1}^{m}I(y_i = 1)}$$

$$\Sigma = \frac{1}{m} \sum_{i=1}^{m}(x_i - \mu_{y_i}) (x_i - \mu_{y_i})^T$$

这里的$x_i$为列向量.

##3. 逻辑回归与高斯判别分析联系与区别

联系：

$$p(y=1|x; \phi,\mu_0, \mu_1, \Sigma) = \frac{1}{1 + e^{-\Theta^Tx}}$$

>上式的意思是如果我们把高斯分布看成x的函数，那么我们可以找到合适的$\Theta$（作为$\phi,{\mu_0}, {\mu_1}, \Sigma$的函数）用逻辑回归的形式来表示高斯分布。事实上，不光是高斯分布，泊松分布以及一些与e的幂有关的分布都可以这样表示。但是反过来不成立，我们并不一定可以将逻辑回归表示成高斯分布，说明了高斯分布的假设是一种强假设，而逻辑回归的假设是一种弱假设。

![GDA_logistic](http://img.blog.csdn.net/20170921172829697?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvaWxvdmV5b3VzdW5uYQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

区别：

>高斯判别分析：高斯分布是一种强假设，如果数据确实是高斯分布，或者接近高斯分布，那么即使数据很少，高斯判别分析也可以得到比逻辑回归更好的效果。
>逻辑回归：是一种弱假设，所以具有很好的鲁棒性，当数据不符合高斯分布时，逻辑回归可以得到比高斯判别分析更好的效果。

基于以上原因，因为实际应用中，我们并不知道数据的实际分布，所以逻辑回归比高斯判别分析使用的更多。

![gda](http://img.blog.csdn.net/20170921122212044?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvaWxvdmV5b3VzdW5uYQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
