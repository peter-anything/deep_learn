import time, os, json
import numpy as np
import matplotlib.pyplot as plt
from utils.common_utils import rel_error
from coco_utils import load_coco_data, sample_coco_minibatch, decode_captions
from rnn import CaptioningRNN
from captioning_trainer import CaptioningTrainer
from image_utils import *

plt.rcParams['figure.figsize'] = (10.0, 8.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

data = load_coco_data(pca_features=True)
small_data2 = load_coco_data(max_train=5000)
good_lstm_model = CaptioningRNN(
          cell_type='lstm',
          word_to_idx=data['word_to_idx'],
          input_dim=data['train_features'].shape[1],
          hidden_dim=200,
          wordvec_dim=256 )

good_lstm_solver = CaptioningTrainer(good_lstm_model, small_data2,
           update_rule='adam',
           num_epochs=50,
           batch_size=100,
           updater_config={
             'learning_rate': 5e-3,
           },
           lr_decay=0.995,
           verbose=True, print_every=50,
         )

good_lstm_solver.train()

plt.plot(good_lstm_solver.loss_history)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training loss history')
plt.show()
for split in ['train', 'val']:
    minibatch = sample_coco_minibatch(small_data2, split=split, batch_size=2)
    gt_captions, features, urls = minibatch
    gt_captions = decode_captions(gt_captions, data['idx_to_word'])

    sample_captions = good_lstm_model.sample(features)
    sample_captions = decode_captions(sample_captions, data['idx_to_word'])

    for gt_caption, sample_caption, url in zip(gt_captions, sample_captions, urls):
        plt.imshow(image_from_url(url))
        plt.title('%s\n%s\nGT:%s' % (split, sample_caption, gt_caption))
        plt.axis('off')
        plt.show()
