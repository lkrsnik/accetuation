# -*- coding: utf-8 -*-
from __future__ import unicode_literals
# text in Western (Windows 1252)

import pickle
import numpy as np
import math
from sklearn.svm import LinearSVC
from skmultilearn.problem_transform.br import BinaryRelevance

# from keras import backend as Input
np.random.seed(7)

#%run ../../../prepare_data.py


c_search = False

filename = 'data/letters_word_accentuation_x_train.pkl'
x_train = pickle.load(open(filename, 'rb'))
filename = 'data/letters_word_accentuation_x_test.pkl'
x_test = pickle.load(open(filename, 'rb'))
filename = 'data/letters_word_accentuation_y_train.pkl'
y_train = pickle.load(open(filename, 'rb'))
filename = 'data/letters_word_accentuation_y_test.pkl'
y_test = pickle.load(open(filename, 'rb'))

if c_search:
    for i in range(-5, 5):
        c = math.pow(10, i)

        svm = LinearSVC(random_state=0, verbose=True, max_iter=10000, C=c, class_weight='balanced')
        cls = BinaryRelevance(classifier=svm)

        cls.fit(x_train, y_train)

        filename = 'finalized_model_c_balanced_' + str(c) + '.sav'
        pickle.dump(cls, open(filename, 'wb'))

        result = cls.score(x_test, y_test)
        print('c = ' + str(c) + ' || result = \n' + str(result))

else:
    c = 1.0

    svm = LinearSVC(random_state=0, verbose=True, max_iter=10000, C=c)
    cls = BinaryRelevance(classifier=svm)

<<<<<<< HEAD
    cls.fit(x_train, y_train)
=======
    cls.fit(concatenated_train, data.y_train)
>>>>>>> e6cb2d7338e88eff7c5b9f09d3a9816e24e28a51

    filename = 'model.sav'
    pickle.dump(cls, open(filename, 'wb'))

<<<<<<< HEAD
    result = cls.score(x_test, y_test)
    print('c = ' + str(c) + ' || result = ' + str(result))
=======
    result = cls.score(concatenated_test, data.y_test)
    print('c = ' + str(c) + ' || result = \n' + str(result))
>>>>>>> e6cb2d7338e88eff7c5b9f09d3a9816e24e28a51

