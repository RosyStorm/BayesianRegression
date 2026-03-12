from BayesianRegression import BayesianRegression
import numpy as np

# 1) 生成数据
N = 7
x_train = np.linspace(-3, 3, N).reshape(N, 1)  # shape (N,1)
def f_true(x): return np.sin(2.0 * x)
y_true = f_true(x_train)
noise_std = 0.2
y_train = y_true + noise_std * (2*np.random.random(y_true.shape) - 1)  # shape (N,1)

beta = 1/noise_std**2
alpha = 2
prior_param = 0
centers = np.linspace(-3, 3, 15)
lengthscale = 0.6

brg = BayesianRegression(x_train, y_train, beta, alpha, style='rbf', n=1, prior_param = prior_param, centers=centers, lengthscale=lengthscale)
brg.fit()
x_test = np.linspace(-5, 5, 400).reshape(400, 1)
mean, cov = brg.predict(x_test)
brg.plot()