#!/usr/bin/env python

import argparse
import logging
import numpy as np
import scipy
from time import time
# import time
import sys
import utils as U
import pickle as pk
import os
import tensorflow as tf

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


logger = logging.getLogger(__name__)


###############################################################################################################################
## Parse arguments
#

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", dest="data_path", type=str, metavar='<str>', required=True, help="The path to the data set")
parser.add_argument("-o", "--out-dir", dest="out_dir_path", type=str, metavar='<str>', required=True, help="The path to the output directory")
parser.add_argument("-t", "--type", dest="model_type", type=str, metavar='<str>', default='SWEM', help="Model type (SWEM|regp|breg|bregp) (default=SWEM)")
parser.add_argument("-l", "--loss", dest="loss", type=str, metavar='<str>', default='ce', help="Loss function (mse|ce) (default=ce)")
parser.add_argument("-e", "--embdim", dest="emb_dim", type=int, metavar='<int>', default=300, help="Embeddings dimension (default=50)")
parser.add_argument("-b", "--batch-size", dest="batch_size", type=int, metavar='<int>', default=32, help="Batch size (default=32)")
parser.add_argument("--trial", dest="trial_data_path", type=str, metavar='<str>', help="The path to the trial data set")
parser.add_argument("-s", "--sentiment", dest="sentiment_data_path", type=str, metavar='<str>', help="The path to the sentiment data set")
parser.add_argument("--humor", dest="humor_data_path", type=str, metavar='<str>', help="The path to the humor data set")
parser.add_argument("--sarcasm", dest="sarcasm_data_path", type=str, metavar='<str>', help="The path to the sarcasm data set")
parser.add_argument("--word_list", dest="word_list_path", type=str, metavar='<str>', help="The path to the sarcasm data set")
parser.add_argument("--dropout", dest="dropout_prob", type=float, metavar='<float>', default=0.1, help="The dropout probability. To disable, give a negative number (default=0.5)")
parser.add_argument("--vocab-path", dest="vocab_path", type=str, metavar='<str>', help="(Optional) The path to the existing vocab file (*.pkl)")
parser.add_argument("--emb", dest="emb_path", type=str, metavar='<str>', help="The path to the word embeddings file (Word2Vec format)")
parser.add_argument("--lr", dest="learn_rate", type=float, metavar='<float>', help="the learn rate")
parser.add_argument("--epochs", dest="epochs", type=int, metavar='<int>', default=20, help="Number of epochs (default=60)")
parser.add_argument("--maxlen", dest="maxlen", type=int, metavar='<int>', default=50, help="Maximum allowed number of words during training. '0' means no limit (default=0)")
parser.add_argument("--word_norm", dest="word_norm", type=int, metavar='<int>', default=1, help="0-stemming, 1-lemma, other-do nothing")
parser.add_argument("--non_gate", dest="non_gate", action='store_true', help="Model type (SWEM|regp|breg|bregp) (default=SWEM)")


args = parser.parse_args()

out_dir = args.out_dir_path

U.mkdir_p(out_dir + '/preds')
U.set_logger(out_dir=out_dir, model_type = args.model_type)
U.print_args(args)


assert args.loss in {'mse', 'ce'}

# if args.seed > 0:
#     np.random.seed(args.seed)

from model_evaluator import Evaluator
import data_reader as dataset
# if args.prompt_id:
#
# else:
#     raise NotImplementedError

###############################################################################################################################
## Prepare data
#


train_x, test_x, train_y, test_y, train_chars, test_chars, task_idx_train, task_idx_test, ruling_embedding_train, ruling_embedding_test,\
    category_embedding_train, category_embedding_test, vocab = dataset.get_data(args)
print('**********************************************************************************************')


# Dump vocab
if not args.vocab_path:
    with open(out_dir + '/vocab.pkl', 'wb') as vocab_file:
        pk.dump(vocab, vocab_file)

###############################################################################################################################
## Some statistics
#

import keras.backend as K


bincounts, mfs_list = U.bincounts(train_y)
with open('%s/bincounts.txt' % out_dir, 'w') as output_file:
    for bincount in bincounts:
        output_file.write(str(bincount) + '\n')

logger.info('Statistics:')

logger.info('  train_x shape: ' + str(np.array(train_x).shape))
logger.info('  test_x shape:  ' + str(np.array(test_x).shape))
logger.info('  train_chars shape: ' + str(np.array(train_chars).shape))
logger.info('  test_chars shape:  ' + str(np.array(test_chars).shape))

logger.info('  train_y shape: ' + str(train_y.shape))
# logger.info('  dev_y shape:   ' + str(dev_y.shape))
logger.info('  test_y shape:  ' + str(test_y.shape))



###############################################################################################################################
## Optimizaer algorithm
#
import tensorflow.keras.optimizers as opt
# optimizer = opt.RMSprop(lr=args.learn_rate)
optimizer ='rmsprop'


###############################################################################################################################
## Building model
#

from models import create_model

if args.loss == 'mse':
    loss = 'mean_squared_error'
    metric = 'mean_absolute_error'
else:
    loss = 'categorical_crossentropy'
    metric = 'accuracy'

# print(train_y.mean(axis=0))
model = create_model(args, args.maxlen, len(ruling_embedding_test[0]), vocab, len(train_y[0]))
model.compile(loss=loss, optimizer=optimizer, metrics=[metric])

###############################################################################################################################
## Save model architecture


# logger.info('Saving model architecture')
# with open(out_dir + '/model_arch.json', 'w') as arch:
#     arch.write(model.to_json(indent=2))
# logger.info('  Done')

###############################################################################################################################
## Evaluator

evl = Evaluator(args, dataset, out_dir, test_x, test_chars, task_idx_test, ruling_embedding_test, test_y, args.batch_size)

###############################################################################################################################
## Training
#

logger.info('--------------------------------------------------------------------------------------------------------------------------')
logger.info('Initial Evaluation:')
# evl.evaluate(model, -1, print_info=True)

total_train_time = 0
total_eval_time = 0
t1 = time()


for ii in range(args.epochs):
    # Training
    t0 = time()
    if args.model_type in {'CNN'}:
        train_history = model.fit(train_chars, train_y, batch_size=args.batch_size, epochs=1, validation_data=(test_chars, test_y), verbose=1)
    elif args.model_type in {'HHMM', 'HHMM_transformer'}:
        train_history = model.fit([train_x, task_idx_train, ruling_embedding_train], train_y, batch_size=args.batch_size, epochs=1, validation_data=([test_x, task_idx_test, ruling_embedding_test], test_y), verbose=1)
    else:
        train_history = model.fit(train_x, train_y, batch_size=args.batch_size, epochs=1, validation_data=(test_x, test_y), verbose=1)
    tr_time = time() - t0
    total_train_time += tr_time

    # Evaluate
    t0 = time()
    evl.evaluate(model, ii)
    evl_time = time() - t0
    total_eval_time += evl_time
    total_time = time()-t1

    # Print information

    train_loss = train_history.history['loss'][0]
    train_metric = train_history.history[metric][0]
    logger.info('Epoch %d, train: %is, evaluation: %is, toaal_time: %is' % (ii, tr_time, evl_time, total_time))
    logger.info('[Train] loss: %.4f, metric: %.4f' % (train_loss, train_metric))
    evl.print_info()

###############################################################################################################################
## Summary of the results
#

logger.info('Training:   %i seconds in total' % total_train_time)
logger.info('Evaluation: %i seconds in total' % total_eval_time)

evl.print_final_info()













