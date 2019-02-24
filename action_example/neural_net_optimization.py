import time
import numpy as np
import matplotlib.pyplot as plt
from utils.data_utils import *
from updater import sgd_momentum, rmsprop, adam
from optimazation_fc_net import FullyConnectedNet
from trainer import Trainer

plt.rcParams['figure.figsize'] = (10.0, 8.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

def rel_error(x, y):
  """ 返回相对误差 """
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

data = get_CIFAR10_data()

num_train = 4000

small_data = {
  'X_train': data['X_train'][:num_train],
  'y_train': data['y_train'][:num_train],
  'X_val': data['X_val'],
  'y_val': data['y_val'],
}

trainers = {}

for update_rule in ['sgd', 'sgd_momentum']:
  model = FullyConnectedNet(hidden_dims=[100, 100, 100, 100, 100], weight_scale=7e-2)

  trainer = Trainer(model, small_data,
                  num_epochs=10, batch_size=100,
                  update_rule=update_rule,
                  updater_config={
                    'learning_rate': 1e-3,
                  },
                  verbose=True)
  trainers[update_rule] = trainer
  trainer.train()

# plt.subplot(3, 1, 1)
# plt.title('Training loss',fontsize=18)
# plt.xlabel('Iteration',fontsize=18)
# plt.ylabel('Loss',fontsize=18)

# plt.subplot(3, 1, 2)
# plt.title('Training accuracy',fontsize=18)
# plt.xlabel('Epoch',fontsize=18)
# plt.ylabel('Accuracy',fontsize=18)

# plt.subplot(3, 1, 3)
# plt.title('Validation accuracy',fontsize=18)
# plt.xlabel('Epoch',fontsize=18)
# plt.ylabel('Accuracy',fontsize=18)
# plt.subplots_adjust(left=0.08, right=0.95, wspace=0.25, hspace=0.25)
a = {'sgd':'o', 'sgd_momentum':'*'}
# for update_rule, trainer in trainers.items():

#   plt.subplot(3, 1, 1)
#   plt.plot(trainer.loss_history, a[update_rule], label=update_rule)


#   plt.subplot(3, 1, 2)
#   plt.plot(trainer.train_acc_history, '-'+a[update_rule], label=update_rule)

#   plt.subplot(3, 1, 3)
#   plt.plot(trainer.val_acc_history, '-'+a[update_rule], label=update_rule)

# for i in [1, 2, 3]:
#   plt.subplot(3, 1, i)
#   plt.legend(loc='upper center', ncol=4)
# plt.gcf().set_size_inches(15, 15)
# plt.show()


learning_rates = {'rmsprop': 1e-4, 'adam': 1e-3}
for update_rule in ['adam', 'rmsprop']:
  model = FullyConnectedNet(hidden_dims=[100, 100, 100, 100, 100], weight_scale=7e-2)

  trainer = Trainer(model, small_data,
                      num_epochs=10, batch_size=100,
                  update_rule=update_rule,
                  updater_config={
                    'learning_rate': learning_rates[update_rule]
                  },
                  verbose=False)
  trainers[update_rule] = trainer
  trainer.train()

plt.subplot(3, 1, 1)
plt.title('Training loss',fontsize=18)
plt.xlabel('Iteration',fontsize=18)
plt.ylabel('Loss',fontsize=18)
plt.subplot(3, 1, 2)
plt.title('Training accuracy',fontsize=18)
plt.xlabel('Epoch',fontsize=18)
plt.ylabel('Accuracy',fontsize=18)

plt.subplot(3, 1, 3)
plt.title('Validation accuracy',fontsize=18)
plt.xlabel('Epoch',fontsize=18)
plt.ylabel('Accuracy',fontsize=18)

plt.subplots_adjust(left=0.08, right=0.95, wspace=0.25, hspace=0.25)
a['adam'] = 'D'
a['rmsprop'] = 'v'
for update_rule, trainer in trainers.items():
  plt.subplot(3, 1, 1)
  plt.plot(trainer.loss_history, a[update_rule], label=update_rule)

  plt.subplot(3, 1, 2)
  plt.plot(trainer.train_acc_history, '-'+a[update_rule], label=update_rule)

  plt.subplot(3, 1, 3)
  plt.plot(trainer.val_acc_history, '-'+a[update_rule], label=update_rule)

for i in [1, 2, 3]:
  plt.subplot(3, 1, i)
  plt.legend(loc='upper center', ncol=4)
plt.gcf().set_size_inches(15, 15)
plt.show()
