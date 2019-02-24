#-*- coding: utf-8 -*-
import numpy as np
def softmax_loss_naive(W, X, y, reg):
    """
    使用显式循环版本计算Softmax损失函数
    N表示：数据个数，D表示：数据维度，C：表示数据类别个数。
    Inputs:
    - W: 形状(D, C) numpy数组，表示分类器权重（参数）.
    - X: 形状(N, D) numpy数组，表示训练数据.
    - y: 形状(N,) numpy数组，表示数据类标。
        其中 y[i] = c 意味着X[i]为第c类数据，c取值为[0,c)
    - reg: 正则化惩罚系数
    Returns  二元组(tuple):
    - loss,数据损失值
    - dW,权重W所对应的梯度，其形状和W相同
    """
    # 初始化损失值与梯度.
    loss = 0.0
    dW = np.zeros_like(W)
    #############################################################################
    #  任务：使用显式循环实现softmax损失值loss及相应的梯度dW 。                 #
    #  温馨提示： 如果不慎,将很容易造成数值上溢。别忘了正则化哟。               #
    #############################################################################
    num_train = X.shape[ 0 ]
    num_class = W.shape[ 1 ]
    for i in range(num_train):
        s = X[i].dot(W)
        scores = s - max(s)
        scores_E = np.exp(scores)
        Z = np.sum(scores_E)
        scores_target = scores_E[ y[i] ]
        loss += -np.log( scores_target/Z )
        for j in range(num_class):
            if j==y[i]:
                dW[ : ,j ] +=-( 1-scores_E[j]/Z) * X[i]
            else :
                dW[:,j] += X[i] * scores_E[j]/Z
    loss = loss/num_train + 0.5 * reg * np.sum( W * W)
    dW = dW/num_train + reg * W
    #############################################################################
    #                           结束编码                                        #
    #############################################################################
    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax损失函数，使用矢量计算版本.
    输入，输出格式与softmax_loss_naive相同
    """
    # 初始化损失值与梯度
    loss = 0.0
    dW = np.zeros_like(W)
    #############################################################################
    # 任务: 不使用显式循环计算softmax的损失值loss及其梯度dW.                    #
    # 温馨提示： 如果不慎,将很容易造成数值上溢。别忘了正则化哟。                #
    #############################################################################
    num_train = X.shape[0]
    s = np.dot(X,W)
    scores = s - np.max(s, axis =1, keepdims=True)
    scores_E = np.exp(scores)
    Z= np.sum(scores_E, axis=1, keepdims=True)
    prob = scores_E/Z
    y_trueClass = np.zeros_like(prob)
    y_trueClass[range(num_train), y] = 1.0    # （N,C)
    loss += -np.sum(y_trueClass * np.log(prob))/num_train + 0.5* reg * np.sum(W * W)
    dW += -np.dot(X.T, y_trueClass - prob) / num_train + reg * W
    #############################################################################
    #                          结束编码                                         #
    #############################################################################
    return loss, dW
