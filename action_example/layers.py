import numpy as np

def affine_forward(x, w, b):
    out = None
    N = x.shape[0]
    x_new = x.reshape(N, -1)
    out = np.dot(x_new, w) + b

    cache = (x, w, b)
    return out, cache

def affine_backward(dout, cache):
    x, w, b = cache
    dx, dw, db = None, None, None

    db = np.sum(dout, axis=0)
    xx = x.reshape(x.shape[0], -1)
    dw = np.dot(xx.T, dout)
    dx = np.dot(dout, w.T)
    dx = np.reshape(dx, x.shape)

    return dx, dw, db

def relu_forward(x):
    out = None

    out = np.maximum(0, x)
    cache = x

    return out, cache

def relu_backward(dout, cache):
    dx, x = None, cache
    dx = dout
    dx[x<=0] = 0

    return dx

def affine_relu_forward(x, w, b):
    a, fc_cache = affine_forward(x, w, b)
    out, relu_cache = relu_forward(a)

    cache = (fc_cache, relu_cache)

    return out, cache

def affine_relu_backward(dout, cache):
    fc_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = affine_backward(da, fc_cache)

    return dx, dw, db

def softmax_loss(x, y):
    probs = np.exp(x - np.max(x, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    N = x.shape[0]
    loss = -np.sum(np.log(probs[np.arange(N), y])) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N

    return loss, dx

def dropout_forward(x, dropout_param):
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':
        mask = (np.random.rand(*x.shape) < p) / p
        out = x * mask

    elif mode == 'test':
        out = x

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)
    return out, cache

def dropout_backward(dout, cache):
    dropout_param, mask = cache
    mode = dropout_param['mode']

    if mode == 'train':
        dx = dout * mask
    elif mode == 'test':
        dx = dout

    return dx

def affine_relu_dropout_foward(x, w, b, dropout_param):
    out_dropout = None
    cache = None

    out_affine, cache_affine = affine_forward(x, w, b)
    out_relu, cache_relu = relu_forward(out_affine)
    out_dropout, cache_dropout = dropout_forward(out_relu, dropout_param)
    cache = (cache_affine, cache_relu, cache_dropout)

    return out_dropout, cache

def affine_relu_dropout_backward(dout, cache):
    cache_affine, cache_relu, cache_dropout = cache

    ddropout = dropout_backward(dout, cache_dropout)
    drelu = relu_backward(ddropout, cache_relu)
    dx, dw, db = affine_backward(drelu, cache_affine)

    return dx, dw, db

def batchnorm_forward(x, gamma, beta, bn_param):
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    if mode == 'train':
        mu = 1 / float(N) * np.sum(x, axis=0)
        xmu = x - mu
        carre = xmu ** 2
        var = 1 / float(N) * np.sum(carre, axis=0)
        sqrtvar = np.sqrt(var + eps)
        invvar = 1 / sqrtvar
        va2 = xmu * invvar
        va3 = gamma * va2
        out = va3 + beta
        running_mean = momentum * running_mean + (1.0 - momentum) * mu
        running_var = momentum * running_var + (1.0 - momentum) * var
        cache = (mu, xmu, carre, var, sqrtvar, invvar, va2, va3, gamma, beta, x, bn_param)
    elif mode == 'test':
        mu = running_mean
        var = running_var
        xhat = (x - mu) / np.sqrt(var + eps)
        out = gamma * xhat + beta
        cache = (mu, var, gamma, beta, bn_param)
    else:
        raise ValueError('无法识别的BN模式： "%s"' % mode)

    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache

def batchnorm_backward(dout, cache):
    dx, dgamma, dbeta = None, None, None
    mu, xmu, carre, var, sqrtvar, invvar, va2, va3, gamma, beta, x, bn_param = cache
    eps = bn_param.get('eps', 1e-5)
    N, D = dout.shape

    dva3 = dout
    dbeta = np.sum(dout, axis=0)

    dva2 = gamma * dva3
    dgamma = np.sum(va2 * dva3, axis=0)

    dxmu = invvar * dva2
    dinvvar = np.sum(xmu * dva2, axis=0)

    dsqrtvar = -1 / (sqrtvar ** 2) * dinvvar
    dvar = 0.5 * (var + eps) ** (-0.5) * dsqrtvar

    dcarre = 1 / float(N) * np.ones((carre.shape)) * dvar
    dxmu += 2 * xmu * dcarre

    dx = dxmu
    dmu = - np.sum(dxmu, axis=0)

    dx += 1 / float(N) * np.ones((dxmu.shape)) * dmu

    return dx, dgamma, dbeta

def batchnorm_backward_alt(dout, cache):
    dx, dgamma, dbeta = None, None, None
    mu, xmu, carre, var, sqrtvar, invvar, va2, va3, gamma, beta, x, bn_param = cache
    eps = bn_param.get('eps', 1e-5)
    N, D = dout.shape
    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum((x - mu) * (var + eps)**(-1. / 2.) * dout, axis=0)
    dx = (1./N) * gamma * (var + eps)**(-1./2.)*(N*dout-np.sum(
                dout, axis=0)-(x-mu)*(var+eps)**(-1.0)*np.sum(dout*(x-mu),axis=0))

    return dx, dgamma, dbeta

def affine_bn_relu_forward(x, w, b, gamma, beta, bn_param):
    x_affine, cache_affine = affine_forward(x, w, b)
    x_bn, cache_bn = batchnorm_forward(x_affine, gamma, beta, bn_param)
    out, cache_relu = relu_forward(x_bn)
    cache = (cache_affine, cache_bn, cache_relu)

    return out, cache

def affine_bn_relu_backward(dout, cache):
    cache_affine, cache_bn, cache_relu = cache
    drelu = relu_backward(dout, cache_relu)
    dbn, dgamma, dbeta = batchnorm_backward_alt(drelu, cache_bn)
    dx, dw, db = affine_backward(dbn, cache_affine)

    return dx, dw, db, dgamma, dbeta
