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
from keras.layers import Input, Concatenate, CuDNNLSTM, CuDNNGRU, Bidirectional, Add, Average, ActivityRegularization
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

            def LAYER(input1, input2, max_len=max_len):
                Avg = Dropout(rate=0.5)(input1)
                Avg = BatchNormalization()(Avg)
                Avg = GlobalAveragePooling1D()(Avg)

                mat = Reshape((max_len, 1))(input2)
                Dot = dot([input1, mat], axes=1)
                Dot = Flatten()(Dot)
                Dot = Dropout(rate=0.5)(Dot)
                Dot = BatchNormalization()(Dot) 

                return Avg, Dot

            embd = Embedding(voca_size,
                             opt.embd_size,
                             name='uni_embd')                                                     
            ####################################
            uni = Input((max_len,), name="t_uni")
            uni_embd = embd(uni)  # token
            w_uni = Input((max_len,), name="w_uni")
            ####################################
            shape = Input((max_len,), name="shape")
            shape_embd = embd(shape)
            w_shape = Input((max_len,), name="w_shape")
            ####################################
            noun = Input((max_len,), name="noun")
            noun_embd = embd(noun)
            w_noun = Input((max_len,), name="w_noun")
            ####################################
            bmm = Input((max_len,), name="bmm")
            bmm_embd = embd(bmm)
            w_bmm = Input((max_len,), name="w_bmm")
            ####################################
            ngram = Input((max_ngram_len,), name="ngram")
            ngram_embd = embd(ngram)
            w_ngram = Input((max_ngram_len,), name="w_ngram")
            ####################################
            jamo3 = Input((max_len,), name="jamo3")
            jamo_embd3 = embd(jamo3)
            w_jamo3 = Input((max_len,), name="w_jamo3")
            ####################################
            jamo2 = Input((max_len,), name="jamo2")
            jamo_embd2 = embd(jamo2)
            w_jamo2 = Input((max_len,), name="w_jamo2")
            ####################################
            jamo1 = Input((max_len,), name="jamo1")
            jamo_embd1 = embd(jamo1)
            w_jamo1 = Input((max_len,), name="w_jamo1")
            ####################################
            img = Input((2048,), name="image")


            uni_avg, uni_dot = LAYER(uni_embd, w_uni, max_len=max_len)
            shape_avg, shape_dot = LAYER(shape_embd, w_shape, max_len=max_len)
            noun_avg, noun_dot = LAYER(noun_embd, w_noun, max_len=max_len)
            ngram_avg, ngram_dot = LAYER(ngram_embd, w_ngram, max_len=max_ngram_len)
            jamo_avg3, jamo_dot3 = LAYER(jamo_embd3, w_jamo3, max_len=max_len)
            jamo_avg2, jamo_dot2 = LAYER(jamo_embd2, w_jamo2, max_len=max_len)
            jamo_avg1, jamo_dot1 = LAYER(jamo_embd1, w_jamo1, max_len=max_len)
            bmm_avg, bmm_dot = LAYER(bmm_embd, w_bmm, max_len=max_len)

            result = Concatenate()([uni_avg, uni_dot, shape_avg, shape_dot, noun_avg, noun_dot ,ngram_avg, ngram_dot, jamo_dot3, jamo_dot2, jamo_dot1, bmm_dot, img])
            
            result = Dropout(rate=0.5)(result)
            result = BatchNormalization()(result) 
            result = Activation('relu')(result)
            outputs = Dense(num_classes, activation=activation)(result)
            ####################################
            model = Model(inputs=[uni, w_uni, shape, w_shape, noun, w_noun, bmm, w_bmm, ngram, w_ngram, jamo3, w_jamo3, jamo2, w_jamo2, jamo1, w_jamo1, img], outputs=outputs)
            optm = keras.optimizers.adam(0.0005)
            
            model.compile(loss='categorical_crossentropy',
                        optimizer=optm,
                        metrics=[top1_acc])
            model.summary(print_fn=lambda x: self.logger.info(x))

        return model
