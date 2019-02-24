import numpy as np

from layers import *
from rnn_layers import *

class CaptioningRNN(object):
    def __init__(self, word_to_idx, input_dim=512, wordvec_dim=128,
               hidden_dim=128, cell_type='rnn'):
        if cell_type not in {'rnn', 'lstm'}:
            raise ValueError('Invalid cell_type "%s"' % cell_type)

        self.cell_type = cell_type
        self.word_to_idx = word_to_idx
        self.idx_to_word = {i : w for w, i in word_to_idx.items()}
        self.params = {}

        vocab_size = len(word_to_idx)

        self._null = word_to_idx['<NULL>']
        self._start = word_to_idx.get('<START>', None)
        self._end = word_to_idx.get('<END>', None)

        self.params['W_embed'] = np.random.randn(vocab_size, wordvec_dim)
        self.params['W_embed'] /= 100

        # cnn param
        self.params['W_proj'] = np.random.randn(input_dim, hidden_dim)
        self.params['W_proj'] /= np.sqrt(input_dim)
        self.params['b_proj'] = np.zeros(hidden_dim)

        #rnn
        dim_mul = {'lstm': 4, 'rnn': 1}[cell_type]
        self.params['Wx'] = np.random.randn(wordvec_dim, dim_mul * hidden_dim)
        self.params['Wx'] /= np.sqrt(wordvec_dim)
        self.params['Wh'] = np.random.randn(hidden_dim, dim_mul * hidden_dim)
        self.params['Wh'] /= np.sqrt(hidden_dim)
        self.params['b'] = np.zeros(dim_mul * hidden_dim)

        self.params['W_vocab'] = np.random.randn(hidden_dim, vocab_size)
        self.params['W_vocab'] /= np.sqrt(hidden_dim)
        self.params['b_vocab'] = np.zeros(vocab_size)

    def loss(self, features, captions):
        captions_in = captions[:, :-1]
        captions_out = captions[:, 1:]

        mask = (captions_out != self._null)

        W_proj, b_proj = self.params['W_proj'], self.params['b_proj']

        W_embed = self.params['W_embed']

        Wx, Wh, b = self.params['Wx'], self.params['Wh'], self.params['b']
        W_vocab, b_vocab = self.params['W_vocab'], self.params['b_vocab']
        loss, grads = 0.0, {}
        h0, cache_h0 = affine_forward(features, W_proj, b_proj)
        x, cache_embedding = word_embedding_forward(captions_in, W_embed)
        if self.cell_type =='rnn':
            out_h,cache_rnn = rnn_forward(x, h0, Wx, Wh, b)
        elif self.cell_type == 'lstm':
            out_h,cache_rnn = lstm_forward(x, h0, Wx, Wh, b)
        else:
            raise ValueError('Invalid cell_type "%s"'%self.cell_type)

        yHat,cache_out = temporal_affine_forward(out_h, W_vocab,
                                             b_vocab)

        loss,dy = temporal_softmax_loss(yHat,captions_out,
                                mask, verbose=False)

        dout_h,dW_vocab,db_vocab = temporal_affine_backward(dy,
                                                        cache_out)

        if self.cell_type =='rnn':
            dx,dh0,dWx,dWh,db = rnn_backward(dout_h,cache_rnn)
        elif self.cell_type == 'lstm':
            dx,dh0,dWx,dWh,db = lstm_backward(dout_h,cache_rnn)
        else:
            raise ValueError('Invalid cell_type "%s"'%self.cell_type)

        dW_embed= word_embedding_backward(dx, cache_embedding)
        dfeatures,dW_proj,db_proj= affine_backward(dh0,cache_h0)
        grads['W_proj']=dW_proj
        grads['b_proj']=db_proj
        grads['W_embed']=dW_embed
        grads['Wx']=dWx
        grads['Wh']=dWh
        grads['b']=db
        grads['W_vocab']=dW_vocab
        grads['b_vocab']=db_vocab

        return loss, grads

    def sample(self, features, max_length=30):
        N = features.shape[0]

        captions = self._null * np.ones((N, max_length), dtype=np.int32)

        W_proj, b_proj = self.params['W_proj'], self.params['b_proj']
        W_embed = self.params['W_embed']
        Wx, Wh, b = self.params['Wx'], self.params['Wh'], self.params['b']
        W_vocab, b_vocab = self.params['W_vocab'], self.params['b_vocab']

        N, D = features.shape
        affine_out, affine_cache = affine_forward(features ,
                                              W_proj, b_proj)
        prev_word_idx = [self._start]*N
        prev_h = affine_out
        prev_c = np.zeros(prev_h.shape)
        captions[:,0] = self._start
        for i in range(1,max_length):
            prev_word_embed  = W_embed[prev_word_idx]
            if self.cell_type == 'rnn':
                next_h, rnn_step_cache = rnn_step_forward(
                    prev_word_embed,  prev_h, Wx, Wh, b)
            elif self.cell_type == 'lstm':
                next_h, next_c,lstm_step_cache = lstm_step_forward(
                    prev_word_embed,prev_h,prev_c, Wx, Wh, b)
                prev_c = next_c
            else:
                raise ValueError('Invalid cell_type "%s"' % self.cell_type)
            vocab_affine_out,vocab_affine_out_cache= affine_forward(
            next_h, W_vocab,b_vocab)
            captions[:,i] = list(np.argmax(vocab_affine_out, axis = 1))
            prev_word_idx = captions[:,i]
            prev_h = next_h
        return captions
