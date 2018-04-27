# -*- coding: utf-8 -*-
from __future__ import unicode_literals
# text in Western (Windows 1252)

import pickle
import numpy as np
np.random.seed(7)

import sys
from prepare_data import *

# preprocess data
# data = Data('l', allow_shuffle_vector_generation=True, save_generated_data=False, shuffle_all_inputs=True)
data = Data('l', save_generated_data=False, shuffle_all_inputs=True)
data.generate_data('../../internal_representations/inputs/letters_word_accentuation_train',
                   '../../internal_representations/inputs/letters_word_accentuation_test',
                   '../../internal_representations/inputs/letters_word_accentuation_validate',
                   content_location='../accetuation/data/',
                   content_name='SlovarIJS_BESEDE_utf8.lex',
                   inputs_location='../accetuation/cnn/internal_representations/inputs/',
                   content_shuffle_vector='content_shuffle_vector',
                   shuffle_vector='shuffle_vector')

# combine all data (if it is unwanted comment code below)
data.x_train = np.concatenate((data.x_train, data.x_test, data.x_validate), axis=0)
data.x_other_features_train = np.concatenate((data.x_other_features_train, data.x_other_features_test, data.x_other_features_validate), axis=0)
data.y_train = np.concatenate((data.y_train, data.y_test, data.y_validate), axis=0)

# build neural network architecture
nn_output_dim = 10
batch_size = 16
actual_epoch = 20
num_fake_epoch = 20

conv_input_shape=(23, 36)
othr_input = (140, )

conv_input = Input(shape=conv_input_shape, name='conv_input')
x_conv = Conv1D(115, (3), padding='same', activation='relu')(conv_input)
x_conv = Conv1D(46, (3), padding='same', activation='relu')(x_conv)
x_conv = MaxPooling1D(pool_size=2)(x_conv)
x_conv = Flatten()(x_conv)

othr_input = Input(shape=othr_input, name='othr_input')

x = concatenate([x_conv, othr_input])
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
x = Dense(nn_output_dim, activation='sigmoid')(x)

model = Model(inputs=[conv_input, othr_input], outputs=x)
opt = optimizers.Adam(lr=1E-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss='mean_squared_error', optimizer=opt, metrics=[actual_accuracy,])
# model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])


# start learning
history = model.fit_generator(data.generator('train', batch_size, content_name='SlovarIJS_BESEDE_utf8.lex', content_location='../accetuation/data/'),
                              data.x_train.shape[0]/(batch_size * num_fake_epoch),
                              epochs=actual_epoch*num_fake_epoch,
                              validation_data=data.generator('test', batch_size),
                              validation_steps=data.x_test.shape[0]/(batch_size * num_fake_epoch))


# save generated data
name = 'test_data/20_epoch'
model.save(name + '.h5')
output = open(name + '_history.pkl', 'wb')
pickle.dump(history.history, output)
output.close()
