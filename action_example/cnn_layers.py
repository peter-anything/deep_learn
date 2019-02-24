import numpy as np
from layers import *
from utils.gradient_check import eval_numerical_gradient_array
from utils.common_utils import rel_error

def conv_forward_naive1(x, w, b, conv_param):
    out = None
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    pad = conv_param['pad']
    stride = conv_param['stride']
    x_pad = np.pad(x, ((0, ), (0, ), (pad, ), (pad, )), 'constant')
    Hhat = int(1 + (H + 2 * pad - HH) / stride)
    What = int(1 + (W + 2 * pad - WW) / stride)
    out = np.zeros((N, F, Hhat, What))

    for n in range(N):
        for f in range(F):
            for i in range(Hhat):
                for j in range(What):
                    xx = x_pad[n, :, i * stride : i * stride + HH, j * stride : j * stride + WW]
                    out[n, f, i, j] = np.sum(xx * w[f]) + b[f]

    cache = (x, w, b, conv_param)

    return out, cache

def conv_forward_naive(x, w, b, conv_param):
    out = None
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    pad = conv_param['pad']
    stride = conv_param['stride']
    x_pad = np.pad(x, ((0, ), (0, ), (pad, ), (pad, )), 'constant')
    Hhat = int(1 + (H + 2 * pad - HH) / stride)
    What = int(1 + (W + 2 * pad - WW) / stride)
    out = np.zeros((N, F, Hhat, What))

    for i in range(Hhat):
        for j in range(What):
            x_pad_masked = x_pad[:, :, i * stride : i * stride + HH, j * stride : j * stride + WW]
            for k in range(F):
                out[:, k, i, j] = np.sum(x_pad_masked * w[k, :, :, :], axis=(1, 2, 3))

    out = out + (b)[None, :, None, None]
    cache = (x, w, b, conv_param)

    return out, cache

def conv_backward_naive(dout, cache):
    dx, dw, db = None, None, None
    x, w, b, conv_param = cache
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    stride, pad = conv_param['stride'], conv_param['pad']
    H_out = int(1 + (H + 2 * pad - HH) / stride)
    W_out = int(1 + (W + 2 * pad - WW) / stride)

    x_pad = np.pad(x, ((0,), (0,), (pad,), (pad,)),
                 mode='constant',constant_values=0)
    dx = np.zeros_like(x)
    dx_pad = np.zeros_like(x_pad)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)

    db = np.sum(dout, axis=(0, 2, 3))
    x_pad = np.pad(x, ((0,),(0,),(pad,),(pad,)),
                 mode='constant', constant_values=0)

    for i in range(H_out):
        for j in range(W_out):
            x_pad_masked = x_pad[:, :, i * stride : i * stride + HH, j * stride : j * stride + WW]
            for k in range(F):
                dw[k, :, :, :] += np.sum(x_pad_masked * (dout[:, k, i, j])[:, None, None, None], axis=0)

            for n in range(N):
                dx_pad[n, :, i * stride : i * stride + HH, j * stride : j * stride + WW] += np.sum(
                    (w[:, :, :, :] * (dout[n, :, i, j])[:, None, None, None]),
                    axis=0
                )
    dx = dx_pad[:, :, pad : -pad, pad : -pad]

    return dx, dw, db

def conv_backward_naive2(dout, cache):
    dx, dw, db = None, None, None
    x, w, b, conv_param = cache
    P = conv_param['pad']
    x_pad = np.pad(x, ((0, ), (0, ), (P, ), (P, )), 'constant')
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    N, F, Hh, Hw = dout.shape
    S = conv_param['stride']
    dw = np.zeros((F, C, HH, WW))
    for fprime in range(F):
        for cprime in range(C):
            for i in range(HH):
                for j in range(WW):
                    sub_xpad = x_pad[:, cprime, i : i + Hh * S : S, j : j + Hw * S : S]
                    dw[fprime, cprime, i, j] = np.sum(dout[:, fprime, :, :] * sub_xpad)

    db = np.zeros((F))
    for fprime in range(F):
        db[fprime] = np.sum(dout[:, fprime, :, :])

    dx = np.zeros((N, C, H, W))
    for nprime in range(N):
        for i in range(H):
            for j in range(W):
                for f in range(F):
                    for k in range(Hh):
                        for l in range(Hw):
                            mask1 = np.zeros_like(w[f, :, :, :])
                            mask2 = np.zeros_like(w[f, :, :, :])
                            if (i + P - k * S) < HH and (i + P - k*S) >=0:
                                mask1[:, i + P - k * S, :] = 1.0
                            if (j + P - l * S) < WW and (j + P - l * S) >= 0:
                                mask2[:, j + P - l * S, :] = 1.0
                            w_masked = np.sum(w[f, :, :, :] * mask1 * mask2, axis=(1, 2))
                            dx[nprime, :, i, j] += dout[nprime, f, k, l] * w_masked

    return dx, dw, db

def conv_backward_naive1(dout, cache):
    """
    卷积层反向传播显式循环版本

    Inputs:
    - dout:上层梯度.
    - cache: 前向传播时的缓存元组 (x, w, b, conv_param)

    Returns 元组:
    - dx:  x梯度
    - dw:  w梯度
    - db:  b梯度
    """
    dx, dw, db = None, None, None
    #############################################################################
    #                   任务 ：实现卷积层反向传播                               #
    #############################################################################
    x, w, b, conv_param = cache
    P = conv_param['pad']
    x_pad = np.pad(x,((0,),(0,),(P,),(P,)),'constant')
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    N, F, Hh, Hw = dout.shape
    S = conv_param['stride']
    dw = np.zeros((F, C, HH, WW))
    for fprime in range(F):
            for cprime in range(C):
                for i in range(HH):
                    for j in range(WW):
                        sub_xpad =x_pad[:,cprime,i:i+Hh*S:S,j:j+Hw*S:S]
                        dw[fprime,cprime,i,j] = np.sum(
                            dout[:,fprime,:,:]*sub_xpad)

    db = np.zeros((F))
    for fprime in range(F):
        db[fprime] = np.sum(dout[:,fprime,:,:])
    dx = np.zeros((N, C, H, W))
    for nprime in range(N):
            for i in range(H):
                for j in range(W):
                    for f in range(F):
                        for k in range(Hh):
                            for l in range(Hw):
                                mask1 = np.zeros_like(w[f,:,:,:])
                                mask2 = np.zeros_like(w[f,:,:,:])
                                if (i+P-k*S)<HH and (i+P-k*S)>= 0:
                                    mask1[:,i+P-k*S,:] = 1.0
                                if (j+P-l* S) < WW and (j+P-l*S)>= 0:
                                    mask2[:,:,j+P-l*S] = 1.0
                                w_masked=np.sum(w[f,:,:,:]*mask1*mask2,axis=(1,2))
                                dx[nprime,:,i,j] +=dout[nprime,f,k,l]*w_masked

    #############################################################################
    #                             结束编码                                     #
    #############################################################################
    return dx, dw, db

def rel_error(x, y):
    """ 返回相对误差 """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

def max_pool_forward_naive1(x, pool_param):
    out = None
    N, C, H, W = x.shape
    HH = pool_param['pool_height']
    WW = pool_param['pool_width']

    stride = pool_param['stride']
    H_out = int((H - HH) / stride + 1)
    W_out = int((W - WW) / stride + 1)
    out = np.zeros((N, C, H_out, W_out))

    for n in range(N):
        for c in range(C):
            for w in range(W_out):
                for h in range(H_out):
                    out[n, c, h, w] = np.max(x[n, c,
                        h * stride : h * stride + HH,
                        w * stride : w * stride + WW
                    ])
    cache = (x, pool_param)
    return out, cache

def max_pool_forward_naive(x, pool_param):
    out = None
    N, C, H, W = x.shape
    HH = pool_param['pool_height']
    WW = pool_param['pool_width']

    stride = pool_param['stride']
    H_out = int((H - HH) / stride + 1)
    W_out = int((W - WW) / stride + 1)
    out = np.zeros((N, C, H_out, W_out))

    for i in range(H_out):
        for j in range(W_out):
            x_masked = x[:, :, i * stride : i * stride + HH, j * stride : j * stride + WW]
            out[:, :, i, j] = np.max(x_masked, axis=(2, 3))

    cache = (x, pool_param)
    return out, cache

def max_pool_backward_naive(dout, cache):
    dx = None
    x, pool_param = cache
    N, C, H, W = x.shape
    HH = pool_param['pool_height']
    WW = pool_param['pool_width']
    stride = pool_param['stride']
    H_out = int((H - HH) / stride + 1)
    W_out = int((W - WW) / stride + 1)
    dx = np.zeros_like(x)

    for i in range(H_out):
        for j in range(W_out):
            x_masked = x[:, :, i * stride : i * stride + HH, j * stride : j * stride + WW]
            max_x_masked = np.max(x_masked, axis=(2, 3))
            temp_binary_mask = (x_masked == (max_x_masked)[:, :, None, None])
            dx[:, :, i * stride : i * stride + HH, j * stride : j * stride + WW] += temp_binary_mask * (dout[:, :, i, j])[:, :, None, None]

    return dx

def max_pool_backward_naive1(dout, cache):
    dx = None
    x, pool_param = cache
    Hp = pool_param['pool_height']
    Wp = pool_param['pool_width']
    S = pool_param['stride']
    N, C, H, W = x.shape
    H1 = int((H - Hp) / S + 1)
    W1 = int((W - Wp) / S + 1)
    dx = np.zeros((N, C, H, W))

    for nprime in range(N):
        for cprime in range(C):
            for k in range(H1):
                for l in range(W1):
                    x_pooling = x[nprime, cprime, k * S : k * S + Hp, l * S : l * S + Wp]
                    maxi = np.max(x_pooling)
                    x_mask = x_pooling == maxi
                    dx[nprime, cprime, k * S : k * S + Hp, l * S : l * S + Wp] += dout[nprime, cprime, k, l] * x_mask

    return dx

def conv_forward_fast(x, w, b, conv_param):
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    stride, pad = conv_param['stride'], conv_param['pad']
    assert (W + 2 * pad - WW) % stride == 0, '宽度异常'
    assert (H + 2 * pad - HH) % stride == 0, '高度异常'
    p = pad
    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

    H += 2 * pad
    W += 2 * pad
    out_h = int((H - HH) / stride + 1)
    out_w = int((W - WW) / stride + 1)
    shape = (C, HH, WW, N, out_h, out_w)
    strides = (H * W, W, 1, C * H * W, stride * W, stride)
    strides = x.itemsize * np.array(strides)
    x_stride = np.lib.stride_tricks.as_strided(x_padded, shape=shape, strides=strides)
    x_cols = np.ascontiguousarray(x_stride)
    x_cols.shape = (C * HH *  WW, N * out_h * out_w)
    new_w = w.reshape(F, -1)
    res = new_w.dot(x_cols) + b.reshape(-1, 1)
    res.shape = (F, N, out_h, out_w)
    out = res.transpose(1, 0, 2, 3)
    out = np.ascontiguousarray(out)
    cache = (x, w, b, conv_param)

    return out, cache

def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    out, cache = None, None

    N, C, H, W = x.shape
    temp_output, cache = batchnorm_forward(
        x.transpose(0, 3, 2, 1).reshape((N * H * W, C)),
        gamma, beta, bn_param
    )

    out = temp_output.reshape(N, W, H, C).transpose(0, 3, 2, 1)

    return out, cache

def spatial_batchnorm_backward(dout, cache):
    dx, dgamma, dbeta = None, None, None

    N, C, H, W = dout.shape
    dx_temp, dgamma, dbeta = batchnorm_backward_alt(
        dout.transpose(0, 3, 2, 1).reshape((N * H * W, C)),
        cache
    )

    dx = dx_temp.reshape(N, W, H, C).transpose(0, 3, 2, 1)

    return dx, dgamma, dbeta

def conv_relu_forward(x, w, b, conv_param):
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    out, relu_cache = relu_forward(a)
    cache = (conv_cache, relu_cache)

    return out, cache

def conv_relu_backward(dout, cache):
    conv_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = conv_backward_naive(da, conv_cache)

    return dx, dw, db

def conv_relu_pool_forward(x, w, b, conv_param, pool_param):
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    s, relu_cache = relu_forward(a)
    out, pool_cache = max_pool_forward_naive(s, pool_param)
    cache = (conv_cache, relu_cache, pool_cache)

    return out, cache

def conv_relu_pool_backward(dout, cache):
    conv_cache, relu_cache, pool_cache = cache
    ds = max_pool_backward_naive(dout, pool_cache)
    da = relu_backward(ds, relu_cache)
    dx, dw, db = conv_backward_naive(da, conv_cache)

    return dx, dw, db
