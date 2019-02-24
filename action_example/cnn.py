import numpy as np
from cnn_layers import *

class ThreeLayerConvNet(object):
    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
        hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,):
        self.params = {}
        self.reg = reg
        C, H, W = input_dim
        self.params['W1'] = weight_scale * np.random.rand(num_filters, C, filter_size, filter_size)
        self.params['b1'] = np.zeros(num_filters)
        self.params['W2'] = weight_scale * np.random.randn(
            int((H / 2) * (W / 2)) * num_filters, hidden_dim
        )
        self.params['b2'] = np.zeros(hidden_dim)
        self.params['W3'] = weight_scale * np.random.randn(
            hidden_dim, num_classes
        )
        self.params['b3'] = np.zeros(num_classes)

    def loss(self, X, y=None):
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': int((filter_size - 1) / 2)}
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
        scores = None

        conv_forward_out_1, cache_forward_1 = conv_relu_pool_forward(
            X, self.params['W1'], self.params['b1'], conv_param, pool_param
        )
        affine_forward_out_2, cache_forward_2 = affine_forward(
            conv_forward_out_1, self.params['W2'], self.params['b2']
        )
        affine_relu_2, cache_relu_2 = relu_forward(affine_forward_out_2)
        scores, cache_forward_3 = affine_forward(
            affine_relu_2, self.params['W3'], self.params['b3']
        )

        if y is None:
            return scores
        loss, grads = 0, {}
        loss, dout = softmax_loss(scores, y)
        loss += self.reg * 0.5*(np.sum(self.params['W1'] ** 2)
                            + np.sum(self.params['W2'] ** 2)
                            + np.sum(self.params['W3'] ** 2))

        dX3, grads['W3'], grads['b3'] = affine_backward(dout, cache_forward_3)
        dX2 = relu_backward(dX3, cache_relu_2)
        dX2, grads['W2'], grads['b2'] = affine_backward(dX2, cache_forward_2)
        dX1, grads['W1'], grads['b1'] = conv_relu_pool_backward(dX2, cache_forward_1)

        grads['W3'] = grads['W3'] + self.reg * self.params['W3']
        grads['W2'] = grads['W2'] + self.reg * self.params['W2']
        grads['W1'] = grads['W1'] + self.reg * self.params['W1']

        return loss, grads

from utils.gradient_check import eval_numerical_gradient
from utils.common_utils import rel_error
from utils.data_utils import get_CIFAR10_data
from trainer import Trainer
import matplotlib.pyplot as plt

data = get_CIFAR10_data()

num_train = 100
small_data = {
  'X_train': data['X_train'][:num_train],
  'y_train': data['y_train'][:num_train],
  'X_val': data['X_val'],
  'y_val': data['y_val'],
}

model = ThreeLayerConvNet(weight_scale=0.001, hidden_dim=200, reg=0.001)

trainer = Trainer(model, data,
                num_epochs=10, batch_size=50,
                update_rule='adam',
                updater_config={
                  'learning_rate': 1e-3,
                },
                verbose=True, print_every=2)
trainer.train()
