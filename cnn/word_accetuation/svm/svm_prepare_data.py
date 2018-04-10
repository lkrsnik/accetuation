# -*- coding: utf-8 -*-
from __future__ import unicode_literals
# text in Western (Windows 1252)

import pickle
import numpy as np
from scipy import sparse

# from keras import backend as Input
np.random.seed(7)


import sys
sys.path.insert(0, '../../../')
from prepare_data import *

c_search = False
mode = 'l'


data = Data(mode)
if mode == 'l':
    data.generate_data('../../internal_representations/inputs/letters_word_accentuation_train',
                       '../../internal_representations/inputs/letters_word_accentuation_test',
                       '../../internal_representations/inputs/letters_word_accentuation_validate')
else:
    data.generate_data('../../internal_representations/inputs/syllables_word_accetuation_train',
                       '../../internal_representations/inputs/syllables_word_accetuation_test',
                       '../../internal_representations/inputs/syllables_word_accetuation_validate')

final_train = sparse.csr_matrix(np.concatenate((np.reshape(data.x_train, (data.x_train.shape[0], 828)), data.x_other_features_train), axis=1))
final_test = sparse.csr_matrix(np.concatenate((np.reshape(data.x_test, (data.x_test.shape[0], 828)), data.x_other_features_test), axis=1))

filename = 'data/letters_word_accentuation_x_train.pkl'
pickle.dump(final_train, open(filename, 'wb'))
filename = 'data/letters_word_accentuation_x_test.pkl'
pickle.dump(final_test, open(filename, 'wb'))
filename = 'data/letters_word_accentuation_y_train.pkl'
pickle.dump(data.y_train, open(filename, 'wb'))
filename = 'data/letters_word_accentuation_y_test.pkl'
pickle.dump(data.y_test, open(filename, 'wb'))
