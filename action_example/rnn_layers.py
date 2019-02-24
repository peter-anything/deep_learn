import numpy as np
from utils.common_utils import rel_error
from utils.gradient_check import *

def rnn_step_forward(x, prev_h, Wx, Wh, b):
    next_h, cache = None, None
    a = prev_h.dot(Wh) + x.dot(Wx) + b
    next_h = np.tanh(a)
    cache = (x, prev_h, Wh, Wx, b, next_h)

    return next_h, cache

def rnn_step_backward(dnext_h, cache):
    dx, dprew_h, dWx, dWh, db = None, None, None, None, None

    x, prev_h, Wh, Wx, b, next_h = cache
    dscores = dnext_h * (1 - next_h * next_h)
    dWx = np.dot(x.T, dscores)
    db = np.sum(dscores, axis=0)
    dWh = np.dot(prev_h.T, dscores)
    dx = np.dot(dscores, Wx.T)
    dprew_h = np.dot(dscores, Wh.T)

    return dx, dprew_h, dWx, dWh, db

def rnn_forward(x, h0, Wx, Wh, b):
    h, cache = None, None

    N, T, D = x.shape
    (H, ) = b.shape
    h = np.zeros((N, T, H))
    prev_h = h0
    for t in range(T):
        xt = x[:, t, :]
        next_h, _ = rnn_step_forward(xt, prev_h, Wx, Wh, b)
        prev_h = next_h
        h[:, t, :] = prev_h
    cache = (x, h0, Wh, Wx, b, h)

    return h, cache

def rnn_backward(dh, cache):
    x, h0, Wh, Wx, b, h = cache
    N, T, H = dh.shape
    _, _, D = x.shape
    next_h = h[:, T - 1, :]
    dprev_h = np.zeros((N, H))
    dx = np.zeros((N, T, D))
    dh0 = np.zeros((N, H))
    dWx = np.zeros((D, H))
    dWh = np.zeros((H, H))
    db = np.zeros((H, ))
    for t in range(T):
        t = T - 1 - t
        xt = x[:, t, :]
        if t == 0:
            prev_h = h0
        else:
            prev_h = h[:, t - 1, :]
        step_cache = (xt, prev_h, Wh, Wx, b, next_h)
        next_h = prev_h
        dnext_h = dh[:, t, :] + dprev_h
        dx[:, t, :], dprev_h, dWxt, dWht, dbt = rnn_step_backward(dnext_h, step_cache)
        dWx, dWh, db = dWx + dWxt, dWh + dWht, db + dbt
    dh0 = dprev_h

    return dx, dh0, dWx, dWh, db

def word_embedding_forward(x, W):
    out, cache = None, None

    N, T = x.shape
    V, D = W.shape

    out = np.zeros((N, T, D))
    for i in range(N):
        for j in range(T):
            out[i, j] = W[x[i, j]]
    cache = (x, W.shape)
    return out, cache

def word_embedding_backward(dout, cache):
    dW = None
    x, W_shape = cache
    dW = np.zeros(W_shape)

    np.add.at(dW, x, dout)

    return dW

def temporal_affine_forward(x, w, b):
    N, T, D = x.shape
    M = b.shape[0]

    out = x.reshape(N * T, D).dot(w).reshape(N, T, M) + b
    cache = x, w, b, out

    return out, cache

def temporal_affine_backward(dout, cache):
    x, w, b, out = cache
    N, T, D = x.shape
    M = b.shape[0]

    dx = dout.reshape(N * T, M).dot(w.T).reshape(N, T, D)
    dw = dout.reshape(N * T, M).T.dot(x.reshape(N * T, D)).T
    db = dout.sum(axis=(0, 1))

    return dx, dw, db

def temporal_softmax_loss(x, y, mask, verbose=False):
    N, T, V = x.shape

    x_flat = x.reshape(N * T, V)
    y_flat = y.reshape(N * T)
    mask_flat = mask.reshape(N * T)

    probs = np.exp(x_flat - np.max(x_flat, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    loss = - np.sum(mask_flat * np.log(probs[np.arange(N * T), y_flat])) / N
    dx_flat = probs.copy()
    dx_flat[np.arange(N * T), y_flat] -= 1
    dx_flat /= N
    dx_flat *= mask_flat[:, None]

    if verbose:
        pass
        # print('dx_flat: ', dx_flat.shape)

    dx = dx_flat.reshape(N, T, V)

    return loss, dx

def sigmoid(x):
    pos_mask = (x >= 0)
    neg_mask = (x < 0)
    z = np.zeros_like(x)
    z[pos_mask] = np.exp(-x[pos_mask])
    z[neg_mask] = np.exp(x[neg_mask])
    top = np.ones_like(x)
    top[neg_mask] = z[neg_mask]

    return top / (1 + z)

def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b):
    next_h, next_c, cache = None, None, None
    N, D = x.shape
    N, H = prev_h.shape
    input_gate = sigmoid(np.dot(x, Wx[:, 0 : H]) + np.dot(prev_h, Wh[:, 0 : H]) + b[0 : H])
    forget_gate = sigmoid(np.dot(x, Wx[:, H : 2 * H]) + np.dot(prev_h, Wh[:, H : 2 * H]) + b[H : 2 * H])
    output_gate = sigmoid(np.dot(x, Wx[:, 2 * H : 3 * H]) + np.dot(prev_h, Wh[:, 2 * H : 3 * H]) + b[2 * H : 3 * H])
    input = np.tanh(np.dot(x, Wx[:, 3 * H : 4 * H]) + np.dot(prev_h, Wh[:, 3 * H : 4 * H]) + b[3 * H : 4 * H])

    next_c = forget_gate * prev_c + input * input_gate
    next_scores_c = np.tanh(next_c)
    next_h = output_gate * next_scores_c
    cache = (x, Wx, Wh, b, input, input_gate, output_gate,
           forget_gate, prev_h, prev_c, next_scores_c)

    return next_h, next_c, cache

def lstm_step_backward(dnext_h, dnext_c, cache):
    dx, dprev_h, dc, dWx, dWh, db = None, None, None, None, None, None
    x, Wx, Wh, b, input, input_gate, output_gate, forget_gate, prev_h, prev_c, next_scores_c = cache

    N, D = x.shape
    N, H = prev_h.shape
    dWx = np.zeros((D, 4 * H))
    dxx = np.zeros((D, 4 * H))
    dWh = np.zeros((H, 4 * H))
    dhh = np.zeros((H, 4 * H))
    db = np.zeros(4 * H)
    dx = np.zeros((N, D))
    dprev_h = np.zeros((N, H))
    dc_tem = dnext_c + dnext_h * (1 - next_scores_c ** 2) * output_gate
    dprev_c = forget_gate * dc_tem
    dforget_gate = prev_c * dc_tem
    dinput_gate = input * dc_tem
    dinput = input_gate * dc_tem
    doutput_gate = next_scores_c * dnext_h

    dscores_in_gate = input_gate * (1 - input_gate) * dinput_gate
    dscores_forget_gate = forget_gate * (1 - forget_gate) * dforget_gate
    dscore_out_gate = output_gate * (1 - output_gate) * doutput_gate
    dscores_in = (1 - input ** 2) * dinput
    da = np.hstack((dscores_in_gate, dscores_forget_gate, dscore_out_gate, dscores_in))
    dWx = np.dot(x.T, da)
    dWh = np.dot(prev_h.T, da)
    db = np.sum(da, axis=0)
    dx = np.dot(da, Wx.T)
    dprev_h = np.dot(da, Wh.T)

    return dx, dprev_h, dprev_c, dWx, dWh, db

def lstm_forward(x, h0, Wx, Wh, b):
    N, T, D = x.shape
    H = int(b.shape[0] / 4)
    h = np.zeros((N, T, H))
    cache = {}
    prev_h = h0
    prev_c = np.zeros((N, H))
    for t in range(T):
        xt = x[:, t, :]
        next_h, next_c, cache[t] = lstm_step_forward(xt, prev_h, prev_c, Wx, Wh, b)
        prev_h = next_h
        prev_c = next_c
        h[:, t, :] = prev_h

    return h, cache

def lstm_backward(dh, cache):
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    N, T, H = dh.shape
    x,Wx,Wh,b,input,input_gate,output_gate,forget_gate,prev_h,prev_c,next_scores_c=cache[T-1]
    D = x.shape[1]
    dprev_h = np.zeros((N, H))
    dprev_c = np.zeros((N, H))
    dx = np.zeros((N, T, D))
    dh0 = np.zeros((N, H))
    dWx= np.zeros((D, 4*H))
    dWh = np.zeros((H, 4*H))
    db = np.zeros((4*H,))

    for t in range(T):
        t = T - 1 - t
        step_cache = cache[t]
        dnext_h = dh[:, t, :] + dprev_h
        dnext_c = dprev_c
        dx[:,t,:], dprev_h, dprev_c, dWxt, dWht, dbt = lstm_step_backward(
            dnext_h, dnext_c, step_cache)
        dWx, dWh, db = dWx + dWxt, dWh + dWht, db + dbt

    dh0 = dprev_h

    return dx, dh0, dWx, dWh, db
