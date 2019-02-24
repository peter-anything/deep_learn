import numpy as np
from layers import *
from utils.gradient_check import eval_numerical_gradient_array, eval_numerical_gradient

class FullyConnectedNet(object):
    def __init__(self, input_dim=3*32*32, hidden_dims=[50, 50],
        num_classes=10, reg=0.0, weight_scale=1e-3):

        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.params = {}
        layers_dims = [input_dim] + hidden_dims + [num_classes]

        for i in range(self.num_layers):
            self.params['W' + str(i + 1)] = weight_scale * np.random.randn(
                layers_dims[i], layers_dims[i+1]
            )

            self.params['b' + str(i + 1)] = np.zeros((1, layers_dims[i + 1]))

    def loss(self, X, y=None):
        scores = None
        cache_relu, outs, cache_out = {}, {}, {}
        outs[0] = X
        num_h = self.num_layers - 1

        for i in range(num_h):
            outs[i + 1], cache_relu[i + 1] = affine_relu_forward(outs[i],
                self.params['W' + str(i + 1)],
                self.params['b' + str(i + 1)],
            )

        scores, cache_out = affine_forward(
            outs[num_h],
            self.params['W' + str(num_h + 1)],
            self.params['b' + str(num_h + 1)]
        )

        loss, grads = 0.0, {}

        dout, daffine = {}, {}
        loss, dy = softmax_loss(scores, y)
        h = self.num_layers - 1
        for i in range(self.num_layers):
            loss += 0.5 * self.reg * (np.sum(
                self.params['W' + str(i + 1)]*
                self.params['W' + str(i + 1)]
            ))
        dout[h], grads['W' + str(h + 1)], grads['b' + str(h + 1)] = affine_backward(dy, cache_out)
        grads['W' + str(h + 1)] += self.reg * self.params['W' + str(h + 1)]
        for i in range(h):
            dout[h - i - 1], grads['W' + str(h - i)], grads['b' + str(h - i)] = affine_relu_backward(
                dout[h - i],
                cache_relu[h - i]
            )
            # dout[h-i-1],grads['W'+str(h-i)],grads['b'+str(h-i)] =affine_relu_backward(
            #     dout[h-i],cache_relu[h-i])

            # grads['W' + str(h -i)] += self.reg * self.params['W' + str(h - i)]
            grads['W'+str(h-i)] += self.reg*self.params['W'+str(h-i)]
        # for i in range(h):
        #     dout[h-i-1],grads['W'+str(h-i)],grads['b'+str(h-i)] =affine_relu_backward(
        #         dout[h-i],cache_relu[h-i])
        #     grads['W'+str(h-i)] += self.reg*self.params['W'+str(h-i)]
        return loss, grads

    def train(self, X, y, X_val,
              y_val,learning_rate=1e-3, learning_rate_decay=0.95,
              num_iters=100,batch_size=200, verbose=False):
        """
        使用随机梯度下降训练神经网络
        Inputs:
        - X: 训练数据
        - y: 训练类标.
        - X_val: 验证数据.
        - y_val:验证类标.
        - learning_rate: 学习率.
        - learning_rate_decay: 学习率衰减系数
        - reg: 权重衰减系数.
        - num_iters: 迭代次数.
        - batch_size: 批量大小.
        - verbose:是否在训练过程中打印结果.
        """
        num_train = X.shape[0]
        iterations_per_epoch = max(num_train / batch_size, 1)

        loss_history = []
        train_acc_history = []
        val_acc_history = []
        best_val=-1
        for it in range(num_iters):
          X_batch = None
          y_batch = None


          sample_index = np.random.choice(num_train, batch_size, replace=True)
          X_batch = X[sample_index, :]  # (batch_size,D)
          y_batch = y[sample_index]  # (1,batch_size)


          #计算损失以及梯度
          loss, grads = self.loss(X_batch, y=y_batch)
          loss_history.append(loss)

          #修改权重
          ############################################################################
          #                    任务：修改深层网络的权重                              #
          ############################################################################
          for i,j in self.params.items():
                self.params[i] += -learning_rate*grads[i]
          ############################################################################
          #                              结束编码                                    #
          ############################################################################


          if verbose and it % 100 == 0:
            print ('iteration %d / %d: loss %f' % (it, num_iters, loss))


          if it % iterations_per_epoch == 0:
            # 检验精度
            train_acc = (self.predict(X) == y).mean()
            val_acc = (self.predict(X_val) == y_val).mean()
            train_acc_history.append(train_acc)
            val_acc_history.append(val_acc)
            if (best_val < val_acc):
                best_val = val_acc

            # 学习率衰减
            learning_rate *= learning_rate_decay

        return {
          'loss_history': loss_history,
          'train_acc_history': train_acc_history,
          'val_acc_history': val_acc_history,
          'best_val_acc':best_val
        }

    def predict(self, X):
        """
        Inputs:
        - X: 输入数据
        Returns:
        - y_pred: 预测类别
        """
        y_pred = None

        ###########################################################################
        #                   任务： 执行深层网络的前向传播，                       #
        #                  然后使用输出层得分函数预测数据类标                     #
        ###########################################################################
        outs ={}
        outs[0] = X
        num_h =self.num_layers-1
        for i in range(num_h):
            outs[i+1],_ =affine_relu_forward(
                outs[i],self.params['W'+str(i+1)],
                self.params['b'+str(i+1)])

        scores,_= affine_forward(
            outs[num_h],self.params['W'+str(num_h+1)],
            self.params['b'+str(num_h+1)])
        y_pred = np.argmax(scores, axis=1)
        ###########################################################################
        #                             结束编码                                    #
        ###########################################################################

        return y_pred

def rel_error(x, y):
    #计算相对错误
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

# import time
# import numpy as np

# import matplotlib.pyplot as plt

# from utils.data_utils import get_CIFAR10_data
# from layers import affine_forward, affine_backward, \
#     relu_forward, relu_backward, \
#     affine_relu_forward, affine_relu_backward, \
#     softmax_loss

# from utils.gradient_check import eval_numerical_gradient_array, eval_numerical_gradient

# data = get_CIFAR10_data()
# X_train = data['X_train']
# y_train = data['y_train']
# X_val = data['X_val']
# y_val = data['y_val']
# X_test = data['X_test']
# y_test = data['y_test']

# input_size = 32 * 32 * 3
# num_classes = 10
# net = FullyConnectedNet(input_size,[100,100],num_classes,reg=0.6,weight_scale=2e-2)
# # 训练网络
# stats = net.train(X_train, y_train, X_val, y_val,
#             num_iters=2000, batch_size=500,
#             learning_rate=8e-3, learning_rate_decay=0.95,
#             verbose=False)
# # 测试性能
# val_acc = (net.predict(X_val) == y_val).mean()
# print ('验证精度: ', val_acc)
# print ('最佳验证精度: ', stats['best_val_acc'])
# plt.subplot(2, 1, 1)
# plt.plot(stats['loss_history'],'o')
# plt.title('Loss history')
# plt.xlabel('Iteration')
# plt.ylabel('Loss')

# plt.subplot(2, 1, 2)
# plt.plot(stats['train_acc_history'], label='train')
# plt.plot(stats['val_acc_history'], label='val')
# plt.title('Classification accuracy history')
# plt.xlabel('Epoch')
# plt.ylabel('Clasification accuracy')
# plt.show()
# test_acc = (best_net.predict(X_test) ==y_test).mean()
# print ('Test accuracy: ', test_acc)
