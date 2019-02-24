import numpy as np
from layers import *
from utils.gradient_check import eval_numerical_gradient_array, eval_numerical_gradient

class FullyConnectedNet(object):
    def __init__(self, input_dim=3*32*32, hidden_dims=[100, 100],
        num_classes=10, dropout=0,
        reg=0.0, weight_scale=1e-2,
        seed=None):

        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.params = {}
        layers_dims = [input_dim] + hidden_dims + [num_classes]
        self.use_dropout = dropout > 0

        for i in range(self.num_layers):
            self.params['W' + str(i + 1)] = weight_scale * np.random.randn(
                layers_dims[i], layers_dims[i+1]
            )

            self.params['b' + str(i + 1)] = np.zeros((1, layers_dims[i + 1]))

        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}

            if seed is not None:
                self.dropout_param['seed'] = seed

    def loss(self, X, y=None):
        mode = 'test' if y is None else 'train'
        if self.dropout_param is not None:
            self.dropout_param['mode'] = mode

        scores = None
        outs, cache = {}, {}
        outs[0] = X
        num_h = self.num_layers - 1

        for i in range(num_h):
            if self.use_dropout:

                outs[i + 1], cache[i + 1] = affine_relu_dropout_foward(outs[i],
                    self.params['W' + str(i + 1)],
                    self.params['b' + str(i + 1)],
                    self.dropout_param
                )
            else:
                outs[i + 1], cache[i + 1] = affine_relu_forward(outs[i],
                    self.params['W' + str(i + 1)],
                    self.params['b' + str(i + 1)],
                )

        scores, cache[num_h + 1] = affine_forward(
            outs[num_h],
            self.params['W' + str(num_h + 1)],
            self.params['b' + str(num_h + 1)]
        )

        if mode == 'test':
            return scores

        loss, grads = 0.0, {}

        dout, daffine = {}, {}
        loss, dy = softmax_loss(scores, y)
        h = self.num_layers - 1
        for i in range(self.num_layers):
            loss += 0.5 * self.reg * (np.sum(
                self.params['W' + str(i + 1)]*
                self.params['W' + str(i + 1)]
            ))
        dout[h], grads['W' + str(h + 1)], grads['b' + str(h + 1)] = affine_backward(dy, cache[h + 1])
        grads['W' + str(h + 1)] += self.reg * self.params['W' + str(h + 1)]
        for i in range(h):
            if self.use_dropout:
                dout[h - i - 1], grads['W' + str(h - i)], grads['b' + str(h - i)] = affine_relu_dropout_backward(
                    dout[h - i],
                    cache[h - i]
                )
                grads['W'+str(h-i)] += self.reg*self.params['W'+str(h-i)]
            else:
                dout[h - i - 1], grads['W' + str(h - i)], grads['b' + str(h - i)] = affine_relu_backward(
                    dout[h - i],
                    cache[h - i]
                )
                grads['W'+str(h-i)] += self.reg*self.params['W'+str(h-i)]
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

# N, D, H1, H2, C = 2, 15, 20, 30, 10
# X = np.random.randn(N, D)
# y = np.random.randint(C, size=(N,))

# for dropout in [0, 0.2, 0.5,0.7]:
#   print ('检验 dropout率 = ', dropout)
#   model = FullyConnectedNet(input_dim=D,hidden_dims=[H1, H2],  num_classes=C,
#                             weight_scale=5e-2, dropout=dropout, seed=13)

#   loss, grads = model.loss(X, y)
#   print ('初始化 loss: ', loss)

#   for name in sorted(grads):
#     f = lambda _: model.loss(X, y)[0]
#     grad_num = eval_numerical_gradient(f, model.params[name], verbose=False, h=1e-5)
#     print ('%s 相对误差: %.2e' % (name, rel_error(grad_num, grads[name])))
