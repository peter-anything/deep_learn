import random
import numpy as np
from utils.data_utils import get_CIFAR10_data
from utils.gradient_check import grad_check_sparse

from softmax_loss import softmax_loss_vectorized, softmax_loss_naive
from softmax import Softmax
import matplotlib.pyplot as plt

cifar10_dir = 'datasets/cifar-10-batches-py'

X_train, y_train, X_val, y_val, X_test, y_test, X_sample, y_sample = get_CIFAR10_data()

# W = np.random.randn(3073, 10) * 0.0001
# loss, grad = softmax_loss_vectorized(W, X_sample, y_sample, 0.0)
# f = lambda w: softmax_loss_vectorized(w, X_sample, y_sample, 0.0)[0]

# grad_numerical = grad_check_sparse(f, W, grad, 10)

# loss, grad = softmax_loss_vectorized(W, X_sample, y_sample, 0.01)
# f = lambda w: softmax_loss_vectorized(w, X_sample, y_sample, 0.01)[0]

# grad_numerical = grad_check_sparse(f, W, grad, 10)

# softmax = Softmax()
# loss_hist = softmax.train(X_sample, y_sample, learning_rate=1e-7, reg=5e-4, num_iters=3500, verbose=True)
# plt.plot(loss_hist)
# plt.xlabel('iteration number')
# plt.ylabel('loss value')
# plt.show()
# y_train_pred = softmax.predict(X_sample)
# print('train dataset num: %f, training correct rate:%f' % (X_sample.shape[0], np.mean(y_sample == y_train_pred)))
# y_train_pred = softmax.predict(X_val)
# print('train dataset num: %f, training correct rate:%f' % (X_val.shape[0], np.mean(y_val == y_train_pred)))

results = {}
best_val = -1
best_softmax = None
################################################################################
#                            任务:                                             #
#               使用全部训练数据训练一个最佳softmax                            #
################################################################################
learning_rates = [1.4e-7, 1.45e-7, 1.5e-7, 1.55e-7, 1.6e-7]
regularization_strengths = [2.3e4, 2.6e4, 2.7e4, 2.8e4, 2.9e4]
for l in learning_rates:
    for r in regularization_strengths:
        softmax = Softmax()
        loss_hist = softmax.train(X_train, y_train, learning_rate=l,
                                  reg=r,num_iters=2000, verbose=True)
        y_train_pred = softmax.predict(X_train)
        train_accuracy= np.mean(y_train == y_train_pred)
        y_val_pred = softmax.predict(X_val)
        val_accuracy= np.mean(y_val == y_val_pred)
        results[(l,r)]=(train_accuracy,val_accuracy)
        if (best_val < val_accuracy):
            best_val = val_accuracy
            best_softmax = softmax
################################################################################
#                            结束编码                                          #
################################################################################

for lr, reg in sorted(results):
    train_accuracy, val_accuracy = results[(lr, reg)]
    print ('lr %e reg %e train accuracy: %f val accuracy: %f' % (
                lr, reg, train_accuracy, val_accuracy))

print ('最佳验证精度为: %f' % best_val)


w = best_softmax.W[:-1, :]
w = w.reshape(32, 32, 3, 10)
w_min, w_max = np.min(w), np.max(w)
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
for i in range(10):
    plt.subplot(2, 5, i+1)
    wimg = 255.0 * (w[:, :, :, i].squeeze() - w.min()) / (w.max() - w.min())
    plt.imshow(wimg.astype('uint8'))
    plt.title(classes[i])
plt.show()
