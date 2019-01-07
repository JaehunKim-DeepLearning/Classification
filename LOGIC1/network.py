# -*- coding: utf-8 -*-
# Copyright 2017 Kakao, Recommendation Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf
import numpy as np

import keras
from keras.models import Model
from keras.layers import Input, Concatenate, CuDNNLSTM, CuDNNGRU, Bidirectional, Add, Average
from keras.layers import Dense, Conv1D ,GlobalAveragePooling1D, GlobalMaxPooling1D, MaxPooling1D, AveragePooling1D
from keras.layers.merge import dot
from keras.layers.core import Reshape
from keras.layers.core import Dropout, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.embeddings import Embedding
from keras.backend import log

from misc import get_logger, Option
opt = Option('./config.json')


def top1_acc(x, y):
    return keras.metrics.top_k_categorical_accuracy(x, y, k=1)


class TextOnly:
    def __init__(self):
        self.logger = get_logger('textonly')


    def get_model(self, num_classes, activation='sigmoid'):
        max_len = opt.max_len
        max_ngram_len = opt.ngram_max_len
        voca_size = opt.unigram_hash_size + 1

        with tf.device('/gpu:0'):
    
            embd = Embedding(voca_size,
                             opt.embd_size,
                             name='uni_embd')                                                     
            ####################################
            t_uni = Input((max_len,), name="t_uni")
            t_uni_embd = embd(t_uni)  # token
            w_uni = Input((max_len,), name="w_uni")
            w_uni_mat = Reshape((max_len, 1))(w_uni)
            dot_uni_result = dot([t_uni_embd, w_uni_mat], axes=1)
            ####################################
            t_shape = Input((max_len,), name="shape")
            t_shape_embd = embd(t_shape)
            w_shape = Input((max_len,), name="w_shape")
            w_shape_mat = Reshape((max_len, 1))(w_shape)
            dot_shape_result = dot([t_shape_embd, w_shape_mat], axes=1)
            ####################################
            t_noun = Input((max_len,), name="noun")
            t_noun_embd = embd(t_noun)
            w_noun = Input((max_len,), name="w_noun")
            w_noun_mat = Reshape((max_len, 1))(w_noun)
            dot_noun_result = dot([t_noun_embd, w_noun_mat], axes=1)
            ####################################
            t_bmm = Input((max_len,), name="bmm")
            t_bmm_embd = embd(t_bmm)
            w_bmm = Input((max_len,), name="w_bmm")
            w_bmm_mat = Reshape((max_len, 1))(w_bmm)
            dot_bmm_result = dot([t_bmm_embd, w_bmm_mat], axes=1)
            ####################################
            t_ngram = Input((max_ngram_len,), name="ngram")
            t_ngram_embd = embd(t_ngram)
            w_ngram = Input((max_ngram_len,), name="w_ngram")
            w_ngram_mat = Reshape((max_ngram_len, 1))(w_ngram)
            dot_ngram_result = dot([t_ngram_embd, w_ngram_mat], axes=1)
            ####################################
            t_jamo_g = Input((max_len,), name="jamo_g")
            t_jamo_g_embd = embd(t_jamo_g)
            w_jamo_g = Input((max_len,), name="w_jamo_g")
            w_jamo_g_mat = Reshape((max_len, 1))(w_jamo_g)
            dot_jamo_g_result = dot([t_jamo_g_embd, w_jamo_g_mat], axes=1)
            ####################################
            t_jamo_g2 = Input((max_len,), name="jamo_g2")
            t_jamo_g2_embd = embd(t_jamo_g2)
            w_jamo_g2 = Input((max_len,), name="w_jamo_g2")
            w_jamo_g2_mat = Reshape((max_len, 1))(w_jamo_g2)
            dot_jamo_g2_result = dot([t_jamo_g2_embd, w_jamo_g2_mat], axes=1)
            ####################################
            t_jamo = Input((max_len,), name="jamo")
            t_jamo_embd = embd(t_jamo)
            w_jamo = Input((max_len,), name="w_jamo")
            w_jamo_mat = Reshape((max_len, 1))(w_jamo)
            dot_jamo_result = dot([t_jamo_embd, w_jamo_mat], axes=1)
            ####################################
            img = Input((2048,), name="image")


            ####################################
            uni_avg = Dropout(rate=0.5)(t_uni_embd)
            uni_avg = BatchNormalization()(uni_avg)
            uni_avg = GlobalAveragePooling1D()(uni_avg)
            ####################################
            uni_dot = Flatten()(dot_uni_result)
            uni_dot = Dropout(rate=0.5)(uni_dot)
            uni_dot = BatchNormalization()(uni_dot)
            ####################################
            shape_avg = Dropout(rate=0.5)(t_shape_embd)   
            shape_avg = BatchNormalization()(shape_avg)
            shape_avg = GlobalAveragePooling1D()(shape_avg)
            ####################################
            shape_dot = Flatten()(dot_shape_result)
            shape_dot = Dropout(rate=0.5)(shape_dot)
            shape_dot = BatchNormalization()(shape_dot)
            ####################################
            noun_avg = Dropout(rate=0.5)(t_noun_embd)
            noun_avg = BatchNormalization()(noun_avg)
            noun_avg = GlobalAveragePooling1D()(noun_avg)
            ####################################
            noun_dot = Flatten()(dot_noun_result)
            noun_dot = Dropout(rate=0.5)(noun_dot)
            noun_dot = BatchNormalization()(noun_dot)
            ####################################
            ngram_avg = Dropout(rate=0.5)(t_ngram_embd)
            ngram_avg = BatchNormalization()(ngram_avg)
            ngram_avg = GlobalAveragePooling1D()(ngram_avg)
            ####################################
            ngram_dot = Flatten()(dot_ngram_result)
            ngram_dot = Dropout(rate=0.5)(ngram_dot)
            ngram_dot = BatchNormalization()(ngram_dot) 
            ####################################
            jamo_g_dot = Flatten()(dot_jamo_g_result)
            jamo_g_dot = Dropout(rate=0.5)(jamo_g_dot)
            jamo_g_dot = BatchNormalization()(jamo_g_dot)
            ####################################
            jamo_g2_dot = Flatten()(dot_jamo_g2_result)
            jamo_g2_dot = Dropout(rate=0.5)(jamo_g2_dot)
            jamo_g2_dot = BatchNormalization()(jamo_g2_dot)
            ####################################
            jamo_dot = Flatten()(dot_jamo_result)
            jamo_dot = Dropout(rate=0.5)(jamo_dot)
            jamo_dot = BatchNormalization()(jamo_dot)
            ####################################
            bmm_dot = Flatten()(dot_bmm_result)
            bmm_dot = Dropout(rate=0.5)(bmm_dot)
            bmm_dot = BatchNormalization()(bmm_dot)
            ####################################     
            result = Concatenate()([uni_avg, uni_dot, shape_avg, shape_dot, noun_avg, noun_dot ,ngram_avg, ngram_dot, jamo_g_dot, jamo_g2_dot, jamo_dot, bmm_dot, img])
            
            result = Dropout(rate=0.5)(result)
            result = BatchNormalization()(result) 
            result = Activation('relu')(result)
            outputs = Dense(num_classes, activation=activation)(result)
            ####################################
            model = Model(inputs=[t_uni, w_uni, t_shape, w_shape, t_noun, w_noun, t_bmm, w_bmm, t_ngram, w_ngram, t_jamo_g, w_jamo_g, t_jamo_g2, w_jamo_g2, t_jamo, w_jamo, img], outputs=outputs)
            optm = keras.optimizers.adam(opt.lr)
            
            model.compile(loss='categorical_crossentropy',
                        optimizer=optm,
                        metrics=[top1_acc])
            model.summary(print_fn=lambda x: self.logger.info(x))

        return model
