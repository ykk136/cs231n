from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange
import time

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    train_num, class_num = X.shape[0], W.shape[1]
    for i in range(train_num):
        scores = X[i, :].dot(W)
        scores -= np.max(scores)
        scores = np.exp(scores)
        sum_scores = np.sum(scores)
        scores /= sum_scores
        loss += -np.log(scores[y[i]])
        
        tmp = np.zeros((1, class_num))
        tmp[0, y[i]] = 1
        xi = X[i, :].reshape(1, -1)
        dW += (scores - tmp) * X[i, :].reshape(-1, 1)
        
    loss /= train_num
    loss += reg * np.sum(np.square(W))
    dW /= train_num
    dW += 2 * reg * W
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    N, D, C = X.shape[0], X.shape[1], W.shape[1]
    zeros_tmp = np.zeros((N, C))
    zeros_tmp[np.arange(N), y] = 1                 # N*1*C
    
    scores = X.dot(W)
    scores -= np.max(scores, axis=1, keepdims=True)
    scores = np.exp(scores)
    scores_sum = np.sum(scores, axis=1, keepdims=True)     # (N, 1)
    scores /= scores_sum                         # (N, C)
    loss = np.sum(-np.log(scores[np.arange(N), y]))
    
    dW = X.T.dot(scores - zeros_tmp)
    
    loss /= N
    loss += reg * np.sum(np.square(W))
    dW /= N
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
