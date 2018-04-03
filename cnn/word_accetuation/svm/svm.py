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

reshaped_train = np.reshape(data.x_train, (data.x_train.shape[0], 828))

for i in range(-5, 5):
    c = math.pow(10, i)

    svm = LinearSVC(random_state=0, verbose=True, max_iter=10000, C=c)
    cls = BinaryRelevance(classifier=svm)

    cls.fit(reshaped_train, data.y_train)

    filename = 'finalized_model_c_' + str(c) + '.sav'
    pickle.dump(cls, open(filename, 'wb'))

    reshaped_test = np.reshape(data.x_test, (data.x_test.shape[0], 828))
    result = cls.score(reshaped_test, data.y_test)
    print('c = ' + str(c) + ' || result = ' + str(result))
