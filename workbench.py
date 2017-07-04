
# coding: utf-8

# In[1]:


# -*- coding: utf-8 -*-
from __future__ import unicode_literals
# text in Western (Windows 1252)

import numpy as np
# import StringIO
import math
from keras.models import Sequential
from keras.layers import Dense, Dropout, Merge
from keras.layers.merge import concatenate
from keras import regularizers
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.constraints import maxnorm
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.models import load_model
np.random.seed(7)

# get_ipython().magic('run ../../../prepare_data.py')

# import sys
# # sys.path.insert(0, '../../../')
# sys.path.insert(0, '/home/luka/Developement/accetuation/')
from prepare_data import *


# X_train, X_other_features_train, y_train, X_validate, X_other_features_validate, y_validate = generate_full_matrix_inputs()
# save_inputs('../../internal_representations/inputs/shuffeled_matrix_train_inputs_other_features_output_11.h5', X_train, y_train, other_features = X_other_features_train)
# save_inputs('../../internal_representations/inputs/shuffeled_matrix_validate_inputs_other_features_output_11.h5', X_validate, y_validate,  other_features = X_other_features_validate)
X_train, X_other_features_train, y_train = load_inputs('cnn/internal_representations/inputs/shuffeled_matrix_train_inputs_other_features_output_11.h5', other_features=True)
X_validate, X_other_features_validate, y_validate = load_inputs('cnn/internal_representations/inputs/shuffeled_matrix_validate_inputs_other_features_output_11.h5', other_features=True)

num_examples = len(X_train)  # training set size
nn_output_dim = 11
nn_hdim = 516

word_processor = Sequential()
word_processor.add(Conv1D(43, (3), input_shape=(23, 43), padding='same', activation='relu'))
word_processor.add(Conv1D(43, (3), padding='same', activation='relu'))
word_processor.add(MaxPooling1D(pool_size=2))
word_processor.add(Flatten())
word_processor.add(Dense(516, activation='relu', kernel_constraint=maxnorm(3)))

metadata_processor = Sequential()
metadata_processor.add(Dense(256, input_dim=167, activation='relu'))

model = Sequential()
model.add(Merge([word_processor, metadata_processor], mode='concat'))  # Merge is your sensor fusion buddy
model.add(Dense(1024, input_dim=(516 + 256), activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1024, input_dim=(516 + 256), activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(nn_output_dim, activation='sigmoid'))


# In[10]:


# epochs = 5
# lrate = 0.1
# decay = lrate/epochs
# sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model.fit([X_train, X_other_features_train], y_train, validation_data=([X_validate, X_other_features_validate], y_validate), epochs=10, batch_size=10)
model.save('v1_1.h5')
