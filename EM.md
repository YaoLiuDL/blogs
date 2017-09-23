#EM算法——一步步推导

可观测数据Y，不可观测数据Z，模型参数为$\Theta$，那么我们的目标就是(极大似然函数)：

$$\max_{\Theta}{\prod_{i=1}^{m}{p(x_i; \Theta)}}$$

log似然函数：

$$\max_{\Theta}{\sum_{i=1}^{m}{\log p(x_i; \Theta)}}$$

加入隐变量：

$$\max_{\Theta}{\sum_{i=1}^{m}{\log {\sum_{z_i}}p(x_i, z_i; \Theta)}}$$
令$$L(\Theta) = {\sum_{i=1}^{m}{\log {\sum_{z_i}}p(x_i, z_i; \Theta)}}$$

其实到这里，我们可以尝试对似然函数求偏导，令其为0，求解参数，但是发现这个函数由于和的对数的存在，变得无法求解。

所以事实上，EM算法是通过迭代逐步近似极大化$L(\Theta)$的。我们可以对式子做一些变换。

$$L(\Theta) = {\sum_{i=1}^{m}{\log \sum_{z_i} Q(z_i) \frac{p(x_i, z_i; \Theta)}{Q(z_i)}}}$$

其中$Q(z_i)$为隐变量的概率分布。

这里插入一点凹函数的性质，如果一个函数f(x)的二阶导数f''(x) >= 0, 那么：

$$f(E[x]) \ge E[f(x)]$$

![凹函数](http://img.blog.csdn.net/20170922174436110?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvaWxvdmV5b3VzdW5uYQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

如果f''(x) > 0,那么该凹函数为严格凹函数，那么等号成立当且仅当x为常数, 或者说 p( x = E[x] ) = 1。

而$\sum_{z_i} Q(z_i) \frac{p(x_i, z_i; \Theta)}{Q(z_i)}$可以看做是在求期望，并且函数logx的二阶导数为-1/x^2 < 0,　所以为严格凹函数，所以：

$$
\begin{aligned}
L(\Theta) &= {\sum_{i=1}^{m}{\log \sum_{z_i} Q(z_i) \frac{p(x_i, z_i; \Theta)}{Q(z_i)}}} \\
& = {\sum_{i=1}^{m}{\log E[ \frac{p(x_i, z_i; \Theta)}{Q(z_i)}]}}\\
& \ge \sum_{i=1}^{m} E[\log \frac{p(x_i, z_i; \Theta)}{Q(z_i)}] \\
& = \sum_{i=1}^{m} Q(z_i) \log \frac{p(x_i, z_i; \Theta)}{Q(z_i)}
\end{aligned}
$$

为了让上面严格凸函数的等号成立，也就是为了让$L(\Theta)$的下界紧贴，我们令

$$\frac{p(x_i, z_i; \Theta)}{Q(z_i)} = c $$

其中ｃ为常数.
而且

$$
\begin{aligned}
Q(z_i) &= \frac{p(x_i, z_i; \Theta)}{\sum_{z_i}{p(x_i, z ; \Theta)}} \\
& = \frac{p(x_i, z_i; \Theta)}{p(x_i; \Theta)}\\
& = p(z_i ; x_i, \Theta)
\end{aligned}$$

所以EM算法，先随机初始化参数$\Theta$：

E-step：

$$Q(z_i) = p(z_i; x_i, \Theta)$$

M-step:

$$\Theta = \arg\max_{\Theta} \sum_{i=1}^{m} Q(z_i) \log \frac{p(x_i, z_i; \Theta)}{Q(z_i)}$$

所以EM算法是一个不断更新下界，不断最大化下界的过程。

![EM](http://img.blog.csdn.net/20170922183507327?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvaWxvdmV5b3VzdW5uYQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

具体在M-step我们还有个约束

$$\sum_{i=1}^{K}Q(z_i) = 1$$

我们可以用拉格朗日函数来求偏导，从而得到参数的值。