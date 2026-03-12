# 贝叶斯线性回归模型

本项目实现了一个基于贝叶斯推断的线性回归模型，支持多项式、径向基函数（RBF）和正弦（Sin）三种基函数形式。代码包含了从模型训练、参数推断到预测可视化的完整流程。

## 目录
- [数学原理](#数学原理)
- [代码结构](#代码结构)
- [使用示例](#使用示例)
- [功能说明](#功能说明)

---

## 数学原理

贝叶斯线性回归的核心思想是将模型参数  $ \mathbf{w} $  视为随机变量，通过观测数据  $ \mathcal{D} = \{\mathbf{X}, \mathbf{y}\} $  来更新其分布。

### 1. 模型定义

假设目标变量  $ t $  由基函数  $ \phi(\mathbf{x}) $  的线性组合加上高斯噪声产生：
 
 $$
   t = \mathbf{w}^T \phi(\mathbf{x}) + \epsilon  
   $$

其中噪声  $ \epsilon \sim \mathcal{N}(0, \beta^{-1}) $ ， $ \beta $  为噪声精度（方差的倒数）。

### 2. 先验分布

假设参数向量  $ \mathbf{w} $  服从高斯先验分布：
 
 $$
   p(\mathbf{w}) = \mathcal{N}(\mathbf{w} | \mathbf{m}_0, \mathbf{S}_0)  
   $$

在代码中，初始先验通常设为  $ \mathbf{m}_0 = \mathbf{0} $ ，协方差矩阵  $ \mathbf{S}_0 = \alpha^{-1} \mathbf{I} $ ，其中  $ \alpha $  为超参数控制先验的方差。

### 3. 后验分布

给定观测数据，参数的后验分布也是高斯分布：
 
 $$
   p(\mathbf{w}|\mathbf{X}, \mathbf{y}) = \mathcal{N}(\mathbf{w} | \mathbf{m}_N, \mathbf{S}_N)  
   $$


其中后验协方差  $ \mathbf{S}_N $  和均值  $ \mathbf{m}_N $  计算如下：
 
 $$
   \mathbf{S}_N^{-1} = \mathbf{S}_0^{-1} + \beta \mathbf{\Phi}^T \mathbf{\Phi}  
   $$

 
 $$
   \mathbf{m}_N = \mathbf{S}_N (\mathbf{S}_0^{-1}\mathbf{m}_0 + \beta \mathbf{\Phi}^T \mathbf{y})  
   $$


代码中的 `fit` 方法实现了上述公式的矩阵运算。

### 4. 预测分布

对于新输入  $ \mathbf{x} $ ，其预测值  $ t $  的分布为：
 
 $$
   p(t|\mathbf{x}, \mathbf{X}, \mathbf{y}) = \mathcal{N}(t | \mathbf{m}_N^T \phi(\mathbf{x}), \sigma^2(\mathbf{x}))  
   $$


预测方差  $ \sigma^2(\mathbf{x}) $  由两部分组成：数据噪声和参数不确定性：
 
 $$
   \sigma^2(\mathbf{x}) = \frac{1}{\beta} + \phi(\mathbf{x})^T \mathbf{S}_N \phi(\mathbf{x})  
   $$


代码中的 `predict` 方法计算了预测的均值和方差。

### 5. 基函数

代码支持三种非线性基函数扩展，使模型能够拟合非线性数据：

1.  **多项式基函数**:
     
    $$
    \phi_i(x) = x^i  
    $$

2.  **径向基函数 (RBF)**:
     
    $$
     \phi_j(x) = \exp \left( - \frac{(x - \mu_j)^2}{2s^2} \right)  
    $$

3.  **正弦基函数**:
     
    $$
    \phi_i(x) = \sin(i x)  
    $$


---

## 代码结构

### 类 `BayesianRegression`

该类封装了贝叶斯回归的所有逻辑。

`__init__(self, x_train, y_train, beta, alpha, prior_param=0, style='poly', n=None, centers=None, lengthscale=None)`
初始化模型。
- **参数**:
  - `x_train`: 训练输入数据。
  - `y_train`: 训练目标数据。
  - `beta`: 噪声精度（初始值）。
  - `alpha`: 参数先验精度（初始值）。
  - `style`: 基函数类型 ('poly', 'rbf', 'sin')。
  - `n`: 多项式或正弦基函数的阶数。
  - `centers`: RBF 基函数的中心点数组。
  - `lengthscale`: RBF 基函数的宽度参数。

#### `fit(self)`
执行贝叶斯推断，计算参数的后验分布  $ \mathbf{m}_N $  和  $ \mathbf{S}_N $ 。此外，该方法还包含了基于证据框架的超参数更新逻辑（更新  $ \alpha $  和  $ \beta $ ）。

#### `predict(self, x_test)`
对新数据进行预测。
- **返回**: 预测均值和预测方差。

#### `add_train_data(self, x_train, y_train)`
增量学习。将新数据合并到现有训练集中，并将当前的后验分布作为新训练阶段的先验分布，重新进行拟合。

#### `basic_func_selector(self, x, style, ...)`
根据 `style` 参数选择并调用相应的基函数变换方法。

#### `plot(self)`
可视化模型结果，包含以下四个子图：
1.  **基函数展示**: 展示前几个基函数的形状。
2.  **预测结果**: 展示预测均值、95% 置信区间以及从后验分布采样的多条函数曲线。
3.  **参数边缘分布**: 展示前几个模型参数的后验边缘分布直方图。
4.  **参数联合分布**: 展示前两个模型参数的二维后验散点图。

---

## 使用示例

```python
import numpy as np
from BayesianRegression import BayesianRegression

# 1. 生成模拟数据
np.random.seed(42)
x_train = np.linspace(-3, 3, 20).reshape(-1, 1)
y_train = np.sin(x_train) + np.random.normal(0, 0.1, size=x_train.shape)

# 2. 初始化模型 (使用 RBF 基函数)
centers = np.linspace(-3, 3, 10)
lengthscale = 0.5
brg = BayesianRegression(x_train, y_train, beta=25.0, alpha=1.0, 
                         style='rbf', centers=centers, lengthscale=lengthscale)

# 3. 拟合模型
brg.fit()

# 4. 预测
x_test = np.linspace(-3.5, 3.5, 100).reshape(-1, 1)
mean, cov = brg.predict(x_test)

# 5. 绘图
brg.plot()
```

---

## 功能说明

1.  **非线性拟合**: 通过 `style` 参数切换不同的基函数，可以轻松拟合线性、多项式或周期性数据。
2.  **不确定性量化**: 模型不仅输出预测均值，还输出预测方差（通过 `predict` 方法和 `plot` 中的 95% 置信带展示），这在风险评估等场景非常重要。
3.  **增量学习**: `add_train_data` 方法允许模型在获得新数据后进行更新，而不需要从头开始训练。
4.  **可视化**: `plot` 方法提供了丰富的可视化手段，帮助理解基函数的作用、模型的拟合效果以及参数的不确定性。