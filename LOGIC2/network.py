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
from keras.layers import Dense, Conv1D, Conv2D, GlobalAveragePooling1D, GlobalMaxPooling1D, MaxPooling1D, AveragePooling1D, MaxPooling2D, AveragePooling2D
from keras.layers.merge import dot, multiply
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
                Avg = ActivityRegularization(l2=0.000001)(Avg) #1.026462 uni8


                return Avg            


            embd = Embedding(voca_size,
                                opt.embd_size,
                                name='uni_embd')                  
            ####################################
            t_uni = Input((max_len,), name="t_uni")
            t_uni_embd = embd(t_uni)
            ####################################
            t_shape = Input((max_len,), name="shape")
            t_shape_embd = embd(t_shape)
            ####################################
            t_noun = Input((max_len,), name="noun")
            t_noun_embd = embd(t_noun)
            ####################################
            t_ngram = Input((max_ngram_len,), name="ngram")
            t_ngram_embd = embd(t_ngram)
            ####################################
            jamo1 = Input((max_len,), name="jamo1")
            t_jamo_embd1 = embd(jamo1)
            ####################################
            jamo2 = Input((max_len,), name="jamo2")
            t_jamo_embd2 = embd(jamo2)
            ####################################
            jamo3 = Input((max_len,), name="jamo3")
            t_jamo_embd3 = embd(jamo3)
            ####################################
            img = Input((2048,), name="image")
            ####################################


            ####################################
            uni_avg = LAYER(t_uni_embd)
            shape_avg = LAYER(t_shape_embd)
            noun_avg = LAYER(t_noun_embd)
            ngram_avg = LAYER(t_ngram_embd, max_ngram_len)
            jamo_avg1 = LAYER(t_jamo_embd1)
            jamo_avg2 = LAYER(t_jamo_embd2)
            ####################################


            ####################################
            result = Concatenate()([uni_avg, shape_avg, noun_avg, ngram_avg, jamo_avg1, jamo_avg2, img]) #MODEL1
            result = Dropout(rate=0.5)(result)
            result = BatchNormalization()(result) 
            result = Activation('relu')(result)
            outputs = Dense(num_classes, activation=activation)(result)
            ####################################


            ####################################
            model = Model(inputs=[t_uni, t_shape, t_noun, t_ngram, jamo1, jamo2, img], outputs=outputs) # MODEL1
            optm = keras.optimizers.adam(0.0003) ## Category
            model.compile(loss='categorical_crossentropy',
                        optimizer=optm,
                        metrics=[top1_acc])
            model.summary(print_fn=lambda x: self.logger.info(x))
            ####################################
        
        return model
