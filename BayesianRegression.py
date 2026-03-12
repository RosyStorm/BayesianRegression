import numpy as np
import matplotlib.pyplot as plt
# import pandas as pd
from scipy.stats import multivariate_normal, norm
from torch.distributions.multivariate_normal import MultivariateNormal

class BayesianRegression:
    def __init__(self, x_train, y_train, beta, alpha, prior_param = 0, style = 'poly', n = None, centers = None, lengthscale = None):
        '''
        :param x_data: x values (n, dim)
        :param y_data: y values (n,)
        :param prior_beta: prior beta  int    噪声精度
        :param alpha: prior alpha (dim',)  参数先验分布方差
        :param prior_param: prior param (dim',)  参数先验分布均值
        '''

        self.style       = style        # 基本函数类型
        self.n           = n            # 基本函数参数
        self.centers     = centers      # 基本函数参数
        self.lengthscale = lengthscale  # 基本函数参数

        self.x_train = x_train                  # x train
        self.y       = y_train                  # y train
        self.nsample = self.x_train.shape[0]    # sample num
        self._xdim   = self.x_train.shape[1]    # dimension of x
        self.x       = self.basic_func_selector(self.x_train, style, n, centers, lengthscale)
        self.xdim    = self.x.shape[1]

        self.beta        = beta             # prior beta
        self.alpha       = alpha            # prior alpha
        self.prior_param = prior_param * np.ones((self.xdim,1)) # prior param

        self.poster_mN = np.zeros((self.xdim,1))      # (dim,)

        self.Sprior     = 1/self.alpha * np.eye(self.xdim)  # prior covarance 先验参数协方差矩阵
        self.SN_posterior = None                              # posterior covariance

        self.x_fit = None
        self.ymean = None
        self.ystd = None

    def fit(self):
        # 计算后验参数
        self.SN_posterior      = np.linalg.inv(self.beta*self.x.T @ self.x + np.linalg.inv(self.Sprior))
        self.poster_mN = self.SN_posterior @ (self.beta*self.x.T @ self.y + np.linalg.inv(self.Sprior) @ self.prior_param)

        M = self.beta * self.x.T @ self.x
        _lambda    = np.linalg.eigh(M)[0]
        self.gamma = np.sum(_lambda/(self.alpha + _lambda))
        self.alpha = self.gamma/np.linalg.norm(self.poster_mN)**2
        self.beta  = (self.nsample - self.gamma)/np.linalg.norm(self.y - self.x @ self.poster_mN)**2

    def predict(self, x_test, n=None):
        '''
        预测x_test的y值
        :param x_test: x values (m, dim)
        :return: y values (m,)
        '''
        self.x_fit = x_test
        x_test = self.basic_func_selector(x_test, self.style, self.n, self.centers, self.lengthscale)
        mean = x_test @ self.poster_mN
        cov = 1/self.beta + np.sum(x_test @ self.SN_posterior * x_test, axis=1).reshape(-1,1)
        
        self.ymean = mean
        self.ystd = np.sqrt(cov)
        return mean, cov
    
    def basic_func_poly(self, x, n):
        dim = x.shape[0]
        bias = np.ones((dim, 1))
        for i in range(self._xdim):
            xi = x[:, i].reshape(dim, 1)
            for j in range(1, n + 1):
                poly = np.power(xi, j).reshape(dim, 1)
                bias = np.concatenate([bias, poly], axis=1)
        return bias

    def basic_func_rbf(self, x, centers, lengthscale):
        dim = x.shape[0]
        bias = np.ones((dim, 1))
        for i in range(self._xdim):
            xi = x[:, i].reshape(dim, 1)
            for j in range(centers.shape[0]):
                rbf = np.exp(-np.square(xi - centers[j])/(2*lengthscale**2))
                bias = np.concatenate([bias, rbf], axis=1)
        return bias

    def basic_func_sin(self, x, n):
        dim = x.shape[0]
        bias = np.ones((dim, 1))
        for i in range(self._xdim):
            xi = x[:, i].reshape(dim, 1)
            for j in range(1, n + 1):
                sin = np.sin(j * xi).reshape(dim, 1)
                bias = np.concatenate([bias, sin], axis=1)
        return bias

    def basic_func_selector(self, x, style='poly', n=None, centers = None, lengthscale = None):
        if style == 'poly' and n is not None:
            return self.basic_func_poly(x, n)
        elif style == 'rbf' and centers is not None and lengthscale is not None:
            return self.basic_func_rbf(x, centers, lengthscale)
        elif style == 'sin' and n is not None:
            return self.basic_func_sin(x, n)
        else:
            raise ValueError('style not supported')

    def add_train_data(self, x_train, y_train):
        # 增加数据再次拟合
        self.x_train = np.concatenate([self.x_train, x_train], axis=0)
        self.y = np.concatenate([self.y, y_train], axis=0)
        self.nsample = self.x_train.shape[0]
        self.x = self.basic_func_selector(self.x_train, self.style, self.n, self.centers, self.lengthscale)
        self.prior_param = self.poster_mN
        self.Sprior = self.SN_posterior

    def plot(self):
        num_samples = 25
        mean = self.poster_mN.flatten() 
        cov = self.SN_posterior
        w_samples = np.random.multivariate_normal(mean, cov, size=num_samples)  # Shape: (S, D)
        x = self.basic_func_selector(self.x_fit, self.style, self.n, self.centers, self.lengthscale)
        f_samples = (x @ w_samples.T).T  # Shape: (S, M)
        plt.figure(figsize=(14, 10))

        # 子图 2: RBF 基函数展示（取前 8 个画出来）
        ax2 = plt.subplot2grid((3,2),(0,1))
        M = self.x_fit.shape[0]
        K_to_plot = 5
        func_plot = self.basic_func_selector(self.x_fit, self.style, self.n, self.centers, self.lengthscale)[:,1:K_to_plot+1]  # 去掉偏置列
        for i in range(K_to_plot):
            ax2.plot(self.x_fit, func_plot[:,i], lw=2, label=f'rbf {i}', alpha=0.9)
        ax2.set_title(f'{self.style} basic func(eigth shown)')
        ax2.set_xlim(np.min(self.x_fit), np.max(self.x_fit))
        ax2.legend(loc='upper right', fontsize='small')

        # 子图 3: 后验预测均值 + 95% 置信带 + 采样函数
        ax3 = plt.subplot2grid((3,2),(1,0),colspan=2)
        for i in range(num_samples):
            ax3.plot(self.x_fit, f_samples[i], color='magenta', alpha=0.15)
        ax3.plot(self.x_fit[:,0], self.ymean[:,0], color='r', label='Posterior Predictive Mean')
        ax3.plot(self.x_fit[:,0], self.ymean[:,0], 'r.', label='Posterior Predictive Dots')
        ax3.fill_between(self.x_fit[:,0],
                        (self.ymean[:,0] - 1.96*self.ystd[:,0]),
                        (self.ymean[:,0] + 1.96*self.ystd[:,0]),
                        color='cyan', alpha=0.3, label=' 95% CI')
        # training data and true function
        ax3.scatter(self.x_train, self.y, color='tab:orange', s=40)
        ax3.set_title('Posterior Predictive + 95% CI + Samples')
        ax3.legend()

        # 子图 4: 参数后验的若干边际直方图（取前三个参数）
        ax4 = plt.subplot2grid((3,2),(2,0))
        # 从后验多变量正态中抽大量样本做直方图
        w_samps_hist = np.random.multivariate_normal(mean, cov, size=3000)
        for i,c in enumerate([0,1,2]):  # bias, rbf0, rbf1
            ax4.hist(w_samps_hist[:,c], bins=30, alpha=0.6, label=f'w[{c}]', density=True)
        ax4.set_title('Histogram of posterior marginals (first 3)')
        ax4.legend()

        # 子图 5: 参数后验前两维的联合等高线（或散点）
        ax5 = plt.subplot2grid((3,2),(2,1))
        # 取大量样本，绘制散点 + 估计核密度等高线（这里用散点和置信椭圆）
        w2 = w_samps_hist[:, :2]
        ax5.scatter(w2[:,0], w2[:,1], s=6, alpha=0.4, color='purple')
        # 计算并画出二维正态等高线（基于 mN 与 SN 的前两维）
        ax5.set_title('Equal Line of Scaters of posterian (first 2)')

        plt.tight_layout()
        plt.show()
        

        
