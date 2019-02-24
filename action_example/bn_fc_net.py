import numpy as np
from layers import *
from utils.gradient_check import eval_numerical_gradient_array, eval_numerical_gradient

class FullyConnectedNet(object):
    def __init__(self, input_dim=3*32*32,hidden_dims=[100],  num_classes=10,
                dropout=0, use_batchnorm=False, reg=0.0,
                weight_scale=1e-2, seed=None):

        self.use_batchnorm = use_batchnorm
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

            if self.use_batchnorm and i < len(hidden_dims):
                self.params['gamma' + str(i + 1)] = np.ones((1, layers_dims[i + 1]))
                self.params['beta' + str(i + 1)] = np.zeros((1, layers_dims[i + 1]))

        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}

            if seed is not None:
                self.dropout_param['seed'] = seed

        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]

    def loss(self, X, y=None):
        mode = 'test' if y is None else 'train'
        if self.dropout_param is not None:
            self.dropout_param['mode'] = mode
        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param[mode] = mode

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
            elif self.use_batchnorm:
                gamma = self.params['gamma' + str(i + 1)]
                beta = self.params['beta' + str(i + 1)]
                outs[i + 1], cache[i + 1] = affine_bn_relu_forward(outs[i],
                    self.params['W' + str(i + 1)],
                    self.params['b' + str(i + 1)],
                    gamma, beta,
                    self.bn_params[i]
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
            elif self.use_batchnorm:
                dx, dw, db, dgamma, dbeta = affine_bn_relu_backward(dout[h - i], cache[h -i])
                dout[h - 1 - i] = dx
                grads['W' + str(h - i)] = dw
                grads['b' + str(h - i)] = db
                grads['gamma' + str(h - i)] = dgamma
                grads['beta' + str(h - i)] = dbeta
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

# 使用BN训练深层神经网络
hidden = [100, 100, 100, 100, 100]

from utils.data_utils import get_CIFAR10_data
from trainer import Trainer
import matplotlib.pyplot as plt

data = get_CIFAR10_data()
X_train = data['X_train']
y_train = data['y_train']
X_val = data['X_val']
y_val = data['y_val']
X_test = data['X_test']
y_test = data['y_test']
hidden = [50, 50, 50, 50, 50, 50, 50]

num_train = 1000
small_data = {
  'X_train': data['X_train'][:num_train],
  'y_train': data['y_train'][:num_train],
  'X_val': data['X_val'],
  'y_val': data['y_val'],
}

bn_trainers = {}
trainers = {}
weight_scales = np.logspace(-4, 0, num=20)
import time
t1 = time.time()

for i, weight_scale in enumerate(weight_scales):
  print ('Running weight scale %d / %d' % (i + 1, len(weight_scales)))
  bn_model = FullyConnectedNet(hidden_dims=hidden, weight_scale=weight_scale, use_batchnorm=True)
  model = FullyConnectedNet(hidden_dims=hidden, weight_scale=weight_scale, use_batchnorm=False)

  bn_trainer = Trainer(bn_model, small_data,
                  num_epochs=10, batch_size=50,
                  update_rule='adam',
                  updater_config={
                    'learning_rate': 3e-3,
                  },
                  verbose=False, print_every=200)
  bn_trainer.train()
  bn_trainers[weight_scale] = bn_trainer

  trainer = Trainer(model, small_data,
                  num_epochs=10, batch_size=50,
                  update_rule='adam',
                  updater_config={
                    'learning_rate': 3e-3,
                  },
                  verbose=False, print_every=200)
  trainer.train()
  trainers[weight_scale] = trainer
t2 = time.time()
print ('time: %.2f' % (t2 - t1))


best_train_accs, bn_best_train_accs = [], []
best_val_accs, bn_best_val_accs = [], []
final_train_loss, bn_final_train_loss = [], []

for ws in weight_scales:
  best_train_accs.append(max(trainers[ws].train_acc_history))
  bn_best_train_accs.append(max(bn_trainers[ws].train_acc_history))

  best_val_accs.append(max(trainers[ws].val_acc_history))
  bn_best_val_accs.append(max(bn_trainers[ws].val_acc_history))

  final_train_loss.append(np.mean(trainers[ws].loss_history[-100:]))
  bn_final_train_loss.append(np.mean(bn_trainers[ws].loss_history[-100:]))

plt.subplots_adjust(left=0.08, right=0.95, wspace=0.25, hspace=0.3)
plt.subplot(3, 1, 1)
plt.title('Best val accuracy vs weight initialization scale',fontsize=18)
plt.xlabel('Weight initialization scale',fontsize=18)
plt.ylabel('Best val accuracy',fontsize=18)
plt.semilogx(weight_scales, best_val_accs, '-D', label='baseline')
plt.semilogx(weight_scales, bn_best_val_accs, '-*', label='batchnorm')
plt.legend(ncol=2, loc='lower right')

plt.subplot(3, 1, 2)
plt.title('Best train accuracy vs weight initialization scale',fontsize=18)
plt.xlabel('Weight initialization scale',fontsize=18)
plt.ylabel('Best training accuracy',fontsize=18)
plt.semilogx(weight_scales, best_train_accs, '-D', label='baseline')
plt.semilogx(weight_scales, bn_best_train_accs, '-*', label='batchnorm')
plt.legend()

plt.subplot(3, 1, 3)
plt.title('Final training loss vs weight initialization scale',fontsize=18)
plt.xlabel('Weight initialization scale',fontsize=18)
plt.ylabel('Final training loss',fontsize=18)
plt.semilogx(weight_scales, final_train_loss, '-D', label='baseline')
plt.semilogx(weight_scales, bn_final_train_loss, '-*', label='batchnorm')
plt.legend()

plt.gcf().set_size_inches(10, 15)
plt.show()
