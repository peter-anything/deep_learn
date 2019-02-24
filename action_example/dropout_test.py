import time
import numpy as np
import matplotlib.pyplot as plt
from utils.data_utils import get_CIFAR10_data
from layers import *
from utils.gradient_check import eval_numerical_gradient_array, eval_numerical_gradient
from dropout_fc_net import FullyConnectedNet
from trainer import Trainer

plt.rcParams['figure.figsize'] = (10.0, 8.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

def rel_error(x, y):
  """ 返回相对误差 """
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

data = get_CIFAR10_data()
model = None
trainer = None

# D,H,C,std,r= 3*32*32,200,10,1e-2,0.6
# model = FullyConnectedNet(input_dim=D, hidden_dims=[H], num_classes=C, weight_scale=std)
# trainer = Trainer(model, data,
#         update_rule='sgd',
#         updater_config={'learning_rate': 1e-3},
#         lr_decay=0.95,
#         num_epochs=20,
#         batch_size=200,
#         print_every=200
#     )
# trainer.train()

# plt.subplot(2, 1, 1)
# plt.title('Traning loss')
# plt.plot(trainer.loss_history, 'o')
# plt.xlabel('Iteration')

# plt.subplot(2, 1, 2)
# plt.title('Accuracy')

# plt.plot(trainer.train_acc_history, '-o', label='train')
# plt.plot(trainer.val_acc_history, '-o', label='val')

# plt.plot([0.5] * len(trainer.val_acc_history), 'k--')
# plt.xlabel('Epoch')
# plt.legend(loc='lower right')
# plt.gcf().set_size_inches(15, 12)
# plt.show()


# num_train = 500
# small_data = {
#   'X_train': data['X_train'][:num_train],
#   'y_train': data['y_train'][:num_train],
#   'X_val': data['X_val'],
#   'y_val': data['y_val'],
# }

# solvers = {}
# dropout_choices = [0, 0.3,0.7]
# for dropout in dropout_choices:
#   model = FullyConnectedNet(hidden_dims=[600], dropout=dropout)
#   print ("dropout激活概率(0表示不使用dropout)%f:"%dropout)

#   trainer = Trainer(model, small_data,
#                   num_epochs=30, batch_size=100,
#                   update_rule='sgd',
#                   updater_config={
#                     'learning_rate': 5e-4,
#                   },
#                   verbose=True, print_every=200)
#   trainer.train()
#   solvers[dropout] = trainer


# train_accs = []
# val_accs = []
# for dropout in dropout_choices:
#   solver = solvers[dropout]
#   train_accs.append(solver.train_acc_history[-1])
#   val_accs.append(solver.val_acc_history[-1])

# plt.subplot(3, 1, 1)
# for dropout in dropout_choices:
#   plt.plot(solvers[dropout].train_acc_history, 'o', label='%.2f dropout' % dropout)
# plt.title('Train accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.legend(ncol=2, loc='lower right')

# plt.subplot(3, 1, 2)
# for dropout in dropout_choices:
#   plt.plot(solvers[dropout].val_acc_history, 'o', label='%.2f dropout' % dropout)
# plt.title('Val accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.legend(ncol=2, loc='lower right')

# plt.gcf().set_size_inches(15, 15)
# plt.show()

model = None
trainer = None

D,C,std,r= 3*32*32,10,1e-2,0.6
model = FullyConnectedNet(input_dim=D, hidden_dims=[100,50], num_classes=C, weight_scale=std, dropout=0.7)
trainer = Trainer(model,data,update_rule='sgd',
                updater_config={'learning_rate': 1e-3,},
                lr_decay=0.95,num_epochs=50, batch_size=500,print_every=300)
trainer.train()
# 可视化训练/验证结果
plt.subplot(2, 1, 1)
plt.title('Training loss')
plt.plot(trainer.loss_history, 'o')
plt.xlabel('Iteration')
plt.subplot(2, 1, 2)
plt.title('Accuracy')
plt.plot(trainer.train_acc_history, '-o', label='train')
plt.plot(trainer.val_acc_history, '-o', label='val')
plt.plot([0.5] * len(trainer.val_acc_history), 'k--')
plt.xlabel('Epoch')
plt.legend(loc='lower right')
plt.gcf().set_size_inches(15, 12)
plt.show()
