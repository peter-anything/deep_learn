import numpy as np
#stochastic gradient descent
def sgd(w, dw, config=None):

    if config is None:
        config = {}

    config.setdefault('learning_rate', 1e-2)

    w -= config['learning_rate'] * dw

    return w, config

def sgd_momentum(w, dw, config=None):
    if config is None:
        config = {}
    config.setdefault('learning_rate', 1e-2)
    config.setdefault('momentum', 0.9)
    v = config.setdefault('velocity', np.zeros_like(w))

    new_w = None
    v = config['momentum'] * config['velocity'] - config['learning_rate'] * dw
    new_w = w + v

    config['velocity'] = v

    return new_w, config

def rmsprop(w, dw, config=None):
    if config is None:
        config = {}

    config.setdefault('learning_rate', 1e-2)
    config.setdefault('decay_rate', 0.99)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('cache', np.zeros_like(w))

    next_w = None
    config['cache'] = config['decay_rate'] * config['cache'] + (1 - config['decay_rate']) * dw ** 2
    next_w = w - config['learning_rate'] * dw / (np.sqrt(config['cache']) + config['epsilon'])

    return next_w, config

def adam(w, dw, config=None):
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-3)
    config.setdefault('beta1', 0.9)
    config.setdefault('beta2', 0.999)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('m', np.zeros_like(w))
    config.setdefault('v', np.zeros_like(w))
    config.setdefault('t', 0)


    next_w = None

    config['t'] += 1
    beta1 = config['beta1']
    beta2 = config['beta2']
    epsilon = config['epsilon']
    learning_rate = config['learning_rate']

    config['m'] = beta1 * config['m'] + (1 - beta1) * dw
    config['v'] = beta2 * config['v'] + (1 - beta2) * dw ** 2
    mb = config['m'] / (1 - beta1 ** config['t'])
    vb = config['v'] / (1 - beta2 ** config['t'])

    next_w = w - learning_rate * mb / (np.sqrt(vb) + epsilon)

    return next_w, config
