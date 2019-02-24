import time
import numpy as np

import matplotlib.pyplot as plt

from utils.data_utils import get_CIFAR10_data
from layers import affine_forward, affine_backward, \
    relu_forward, relu_backward, \
    affine_relu_forward, affine_relu_backward, \
    softmax_loss

from utils.gradient_check import eval_numerical_gradient_array, eval_numerical_gradient
from shallow_layer_net import ShallowLayerNet

plt.rcParams['figure.figsize'] = (10.0, 8.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

def rel_error(x, y):
  #计算相对错误
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

data = get_CIFAR10_data()
X_train = data['X_train']
y_train = data['y_train']
X_val = data['X_val']
y_val = data['y_val']
X_test = data['X_test']
y_test = data['y_test']
for k, v in data.items():
    print ('%s: ' % k, v.shape)

input_size = 32 * 32 * 3
hidden_size = 100
num_classes = 10
net = ShallowLayerNet(input_size, hidden_size, num_classes)

# 训练网络
stats = net.train(X_train, y_train, X_val, y_val,
            num_iters=2000, batch_size=500,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=0.6, verbose=True)

# 验证结果
val_acc = (net.predict(X_val) == y_val).mean()
print ('最终验证正确率: ', val_acc)
print ('历史最佳验证正确率: ', stats['best_val_acc'])


plt.subplot(2, 1, 1)
plt.plot(stats['loss_history'])
plt.title('Loss history')
plt.xlabel('Iteration')
plt.ylabel('Loss')

plt.subplot(2, 1, 2)
plt.plot(stats['train_acc_history'], label='train')
plt.plot(stats['val_acc_history'], label='val')
plt.plot([0.5] * len(stats['val_acc_history']), 'k--')
plt.title('Classification accuracy history')
plt.xlabel('Epoch')
plt.ylabel('Clasification accuracy')
plt.show()
