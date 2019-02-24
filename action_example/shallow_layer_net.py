import numpy as np
from layers import *
from utils.gradient_check import eval_numerical_gradient_array, eval_numerical_gradient

class ShallowLayerNet(object):
    def __init__(self, input_dim=3 * 32 * 32,
        hidden_dim=100, num_classes=10,
        weight_scale=1e-3, reg=0.0):
        self.params = {}
        self.reg = reg

        self.params['W1'] = weight_scale * np.random.randn(input_dim, hidden_dim)
        self.params['b1'] = np.zeros(hidden_dim)

        self.params['W2'] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params['b2'] = np.zeros(num_classes)

    def loss(self, X, y=None):
        scores = None

        out1, cache1 = affine_relu_forward(X, self.params['W1'], self.params['b1'])
        scores, cache2 = affine_forward(out1, self.params['W2'], self.params['b2'])

        if y is None:
            return scores

        loss, grads = 0, {}

        loss, dy = softmax_loss(scores, y)
        loss += 0.5 * self.reg * (np.sum(self.params['W1'] * self.params['W1'])
            + np.sum(self.params['W2'] * self.params['W2']))
        dx2, dw2, grads['b2'] = affine_backward(dy, cache2)
        grads['W2'] = dw2 + self.reg * self.params['W2']
        dx, dw1, grads['b1'] = affine_relu_backward(dx2, cache1)
        grads['W1'] = dw1 + self.reg * self.params['W1']

        return loss, grads


    def train(self, X, y, X_val, y_val,
        learning_rate=1e-3, learning_rate_decay=0.95,
        reg=1e-5, num_iters=100,
        batch_size=200, verbose=False):
        num_train = X.shape[0]
        self.reg = reg
        iterations_per_epoch = max(num_train / batch_size, 1)
        loss_history = []
        train_acc_history = []
        val_acc_history = []
        best_val = -1
        for it in range(num_iters):
            X_batch = None
            y_batch = None

            sample_index = np.random.choice(num_train, batch_size, replace=True)
            X_batch = X[sample_index, :]
            y_batch = y[sample_index]

            loss, grads = self.loss(X_batch, y=y_batch)
            loss_history.append(loss)

            self.params['W1'] += -learning_rate * grads['W1']
            self.params['W2'] += -learning_rate * grads['W2']
            self.params['b1'] += -learning_rate * grads['b1']
            self.params['b2'] += -learning_rate * grads['b2']

            if verbose and it % 100 == 0:
                print ('迭代次数 %d / %d: 损失值 %f' % (it, num_iters, loss))

            if it % iterations_per_epoch == 0:
                train_acc = (self.predict(X_batch) == y_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(val_acc)
                if (best_val < val_acc):
                    best_val = val_acc

                learning_rate *= learning_rate_decay

        return {
          'loss_history': loss_history,
          'train_acc_history': train_acc_history,
          'val_acc_history': val_acc_history,
          'best_val_acc':best_val
        }

    def predict(self, X):
        y_pred = None
        out1, cache1 = affine_relu_forward(X, self.params['W1'], self.params['b1'])
        scores, cache2 = affine_forward(out1, self.params['W2'], self.params['b2'])
        y_pred = np.argmax(scores, axis=1)

        return y_pred

def rel_error(x, y):
  #计算相对错误
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

N, D, H, C = 3, 5, 50, 7
X = np.random.randn(N, D)
y = np.random.randint(C, size=N)

std = 1e-2
model = ShallowLayerNet(input_dim=D, hidden_dim=H, num_classes=C, weight_scale=std)

print ('测试初始化 ... ')
W1_std = abs(model.params['W1'].std() - std)
b1 = model.params['b1']
W2_std = abs(model.params['W2'].std() - std)
b2 = model.params['b2']
assert W1_std < std / 10, '第一层权重初始化有问题'
assert np.all(b1 == 0), '第一层偏置初始化有问题'
assert W2_std < std / 10, '第二层权重初始化有问题'
assert np.all(b2 == 0), '第二层偏置初始化有问题'

print ('测试前向传播过程 ... ')
model.params['W1'] = np.linspace(-0.7, 0.3, num=D*H).reshape(D, H)
model.params['b1'] = np.linspace(-0.1, 0.9, num=H)
model.params['W2'] = np.linspace(-0.3, 0.4, num=H*C).reshape(H, C)
model.params['b2'] = np.linspace(-0.9, 0.1, num=C)
X = np.linspace(-5.5, 4.5, num=N*D).reshape(D, N).T
scores = model.loss(X)
correct_scores = np.asarray(
  [[11.53165108,  12.2917344,   13.05181771,  13.81190102,  14.57198434, 15.33206765,  16.09215096],
   [12.05769098,  12.74614105,  13.43459113,  14.1230412,   14.81149128, 15.49994135,  16.18839143],
   [12.58373087,  13.20054771,  13.81736455,  14.43418138,  15.05099822, 15.66781506,  16.2846319 ]])
scores_diff = np.abs(scores - correct_scores).sum()
assert scores_diff < 1e-6, '前向传播有问题'

print ('测试训练损失(无正则化)')
y = np.asarray([0, 5, 1])
loss, grads = model.loss(X, y)
correct_loss = 3.4702243556
assert abs(loss - correct_loss) < 1e-10, '训练阶段的损失值(无正则化)有问题'

print ('测试训练损失(正则化0.1)')
model.reg = 1.0
loss, grads = model.loss(X, y)
correct_loss = 26.5948426952
assert abs(loss - correct_loss) < 1e-10, '训练阶段的损失值(有正则化)有问题'

for reg in [0.0, 0.7]:
  print ('梯度检验，正则化系数 = ', reg)
  model.reg = reg
  loss, grads = model.loss(X, y)

  for name in sorted(grads):
    f = lambda _: model.loss(X, y)[0]
    grad_num = eval_numerical_gradient(f, model.params[name], verbose=False)
    print ('%s 相对误差: %.2e' % (name, rel_error(grad_num, grads[name])))


