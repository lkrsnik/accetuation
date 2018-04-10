# -*- coding: utf-8 -*-
from __future__ import unicode_literals
# text in Western (Windows 1252)

import pickle
import numpy as np
# import StringIO
import math
from sklearn.svm import LinearSVC
from skmultilearn.problem_transform.br import BinaryRelevance

# from keras import backend as Input
np.random.seed(7)

#%run ../../../prepare_data.py

import sys
sys.path.insert(0, '../../../')
from prepare_data import *

data = Data('l')
#data.generate_data('../../internal_representations/inputs/letters_word_accentuation_train_TEST',
#                   '../../internal_representations/inputs/letters_word_accentuation_test_TEST',
#                   '../../internal_representations/inputs/letters_word_accentuation_validate_TEST')
#data.generate_data('../../internal_representations/inputs/letters_word_accentuation_train_TEST2',
#                   '../../internal_representations/inputs/letters_word_accentuation_test_TEST2',
#                   '../../internal_representations/inputs/letters_word_accentuation_validate_TEST2')
#data.generate_data('../../internal_representations/inputs/letters_word_accentuation_train_TEST_UNSHUFLED',
#                   '../../internal_representations/inputs/letters_word_accentuation_test_TEST_UNSHUFLED',
#                   '../../internal_representations/inputs/letters_word_accentuation_validate_TEST_UNSHUFLED')
data.generate_data('../../internal_representations/inputs/letters_word_accentuation_train',
                   '../../internal_representations/inputs/letters_word_accentuation_test',
                   '../../internal_representations/inputs/letters_word_accentuation_validate')

reshaped_train = np.reshape(data.x_train, (data.x_train.shape[0], data.x_train.shape[1] * data.x_train.shape[2]))
concatenated_train = np.concatenate((reshaped_train, data.x_other_features_train), axis=1)
reshaped_test = np.reshape(data.x_test, (data.x_test.shape[0], data.x_test.shape[1] * data.x_test.shape[2]))
concatenated_test = np.concatenate((reshaped_test, data.x_other_features_test), axis=1)



c_search = False


if c_search:
    for i in range(-5, 5):
        c = math.pow(10, i)

        svm = LinearSVC(random_state=0, verbose=True, max_iter=10000, C=c, class_weight='balanced')
        cls = BinaryRelevance(classifier=svm)

        cls.fit(reshaped_train, data.y_train)

        filename = 'finalized_model_c_balanced_' + str(c) + '.sav'
        pickle.dump(cls, open(filename, 'wb'))

        result = cls.score(reshaped_test, data.y_test)
        print('c = ' + str(c) + ' || result = \n' + str(result))

else:
    c = 1.0

    svm = LinearSVC(random_state=0, verbose=True, max_iter=10000, C=c)
    cls = BinaryRelevance(classifier=svm)

    cls.fit(concatenated_train, data.y_train)

    filename = 'complete_input.sav'
    pickle.dump(cls, open(filename, 'wb'))

    result = cls.score(concatenated_test, data.y_test)
    print('c = ' + str(c) + ' || result = \n' + str(result))

