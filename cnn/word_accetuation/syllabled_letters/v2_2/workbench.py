# -*- coding: utf-8 -*-
from __future__ import unicode_literals
# text in Western (Windows 1252)

import pickle
import numpy as np
from keras import optimizers
from keras.models import Model
from keras.layers import Dense, Dropout, Input
from keras.layers.merge import concatenate
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import Flatten
# from keras import backend as Input
np.random.seed(7)

# get_ipython().magic('run ../../../prepare_data.py')

# import sys
# # sys.path.insert(0, '../../../')
# sys.path.insert(0, '/home/luka/Developement/accetuation/')
from prepare_data import *


# X_train, X_other_features_train, y_train, X_validate, X_other_features_validate, y_validate = generate_full_matrix_inputs()
# save_inputs('../../internal_representations/inputs/shuffeled_matrix_train_inputs_other_features_output_11.h5', X_train, y_train, other_features = X_other_features_train)
# save_inputs('../../internal_representations/inputs/shuffeled_matrix_validate_inputs_other_features_output_11.h5', X_validate, y_validate,  other_features = X_other_features_validate)
# X_train, X_other_features_train, y_train = load_inputs('cnn/internal_representations/inputs/shuffeled_matrix_train_inputs_other_features_output_11.h5', other_features=True)
# X_validate, X_other_features_validate, y_validate = load_inputs('cnn/internal_representations/inputs/shuffeled_matrix_validate_inputs_other_features_output_11.h5', other_features=True)
# letters
# data = Data('l', save_generated_data=False, number_of_syllables=True)

# syllabled letters
data = Data('sl', save_generated_data=False)
data.generate_data('letters_word_accetuation_train',
                   'letters_word_accetuation_test',
                   'letters_word_accetuation_validate', content_name='SlovarIJS_BESEDE_utf8.lex',
                      content_shuffle_vector='content_shuffle_vector', shuffle_vector='shuffle_vector',
                      inputs_location='', content_location='')


num_examples = len(data.x_train)  # training set size
nn_output_dim = 10
nn_hdim = 516
batch_size = 16
# actual_epoch = 1
actual_epoch = 40
# num_fake_epoch = 2
num_fake_epoch = 20



# letters
# conv_input_shape=(23, 36)

# syllabled letters
conv_input_shape=(10, 252)


othr_input = (140, )

conv_input = Input(shape=conv_input_shape, name='conv_input')
# letters
# x_conv = Conv1D(115, (3), padding='same', activation='relu')(conv_input)
# x_conv = Conv1D(46, (3), padding='same', activation='relu')(x_conv)

# syllabled letters
x_conv = Conv1D(80, (2), padding='same', activation='relu')(conv_input)
x_conv = MaxPooling1D(pool_size=2)(x_conv)
x_conv = Flatten()(x_conv)

othr_input = Input(shape=othr_input, name='othr_input')

x = concatenate([x_conv, othr_input])
# x = Dense(1024, input_dim=(516 + 256), activation='relu')(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(nn_output_dim, activation='sigmoid')(x)




model = Model(inputs=[conv_input, othr_input], outputs=x)
opt = optimizers.Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=[actual_accuracy,])
# model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])


history = model.fit_generator(data.generator('train', batch_size, content_name='SlovarIJS_BESEDE_utf8.lex', content_location=''),
                              data.x_train.shape[0]/(batch_size * num_fake_epoch),
                              epochs=actual_epoch*num_fake_epoch,
                              validation_data=data.generator('test', batch_size, content_name='SlovarIJS_BESEDE_utf8.lex', content_location=''),
                              validation_steps=data.x_test.shape[0]/(batch_size * num_fake_epoch),
                              verbose=2
                              )

name = '40_epoch'
model.save(name + '.h5')
output = open(name + '_history.pkl', 'wb')
pickle.dump(history.history, output)
output.close()
