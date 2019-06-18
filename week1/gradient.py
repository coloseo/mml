import numpy as np
import pandas as pd

# 导入数据
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
submit = pd.read_csv('data/sample_submit.csv')

# 初始化设置
beta = [1, 1]
alpha = 0.2
tol_L = 0.1
batch_size = 16

# 对x进行归一化
max_x = max(train['id'])
x = train['id'] / max_x
y = train['questions']

#  定义计算梯度的函数
def compute_grad(beta, x, y):
    """
    计算梯度
    """
    grad = [0, 0]
    grad[0] = 2. * np.mean(beta[0] + beta[1] * x - y)
    grad[1] = 2. * np.mean(x * (beta[0] + beta[1] * x - y))
    return np.array(grad)

# 定义随机梯度计算的函数
def compute_grad_SGD(beta, x, y):
    grad = [0, 0]
    r = np.random.randint(0, len(x))
    grad[0] = 2. * np.mean(beta[0] + beta[1] * x[r] - y[r])
    grad[1] = 2. * np.mean(x[r] * (beta[0] + beta[1] * x[r] - y[r]))
    return np.array(grad)

# 定义小批量随机梯度计算的函数
def compute_grad_batch(beta, batch_size, x, y):
    grad = [0, 0]
    r = np.random.choice(range(len(x)), batch_size, replace=False)
    grad[0] = 2. * np.mean(beta[0] + beta[1] * x[r] - y[r])
    grad[1] = 2. * np.mean(x[r] * (beta[0] + beta[1] * x[r] - y[r]))
    return np.array(grad)

# 定义更新beta的函数
def update_beta(beta, alpha, grad):
    """
    计算 $\beta^{i+1}$
    """
    new_beta = np.array(beta) - alpha * grad
    return new_beta

# 定义计算RMSE的函数
# 计算平方和的根，测量预测向量与目标向量之间的距离，又称为欧几里得范数，L2范数
def rmse(beta, x, y):
    squared_err = (beta[0] + beta[1] * x - y) ** 2
    res = np.sqrt(np.mean(squared_err))
    return res


# 进行第一次计算
grad = compute_grad(beta, x, y)
loss = rmse(beta, x, y)
beta = update_beta(beta, alpha, grad)
loss_new = rmse(beta, x, y)

# 开始迭代
i = 1
while np.abs(loss_new - loss) > tol_L:
    beta = update_beta(beta, alpha, grad)
    grad = compute_grad(beta, x, y)
    loss = loss_new
    loss_new = rmse(beta, x, y)
    i += 1
    print(f'Round {i} Diff RMSE {abs(loss_new - loss)}')
print(f'Coef: {beta[1]} \nIntercept {beta[0]}')
