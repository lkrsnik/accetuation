# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import numpy as np
from keras.models import load_model
import sys
import pickle
import time

from prepare_data import *

np.random.seed(7)

data = Data('l', shuffle_all_inputs=False)
content = data._read_content('data/SlovarIJS_BESEDE_utf8.lex')
dictionary, max_word, max_num_vowels, vowels, accented_vowels = data._create_dict(content)
feature_dictionary = data._create_slovene_feature_dictionary()
syllable_dictionary = data._create_syllables_dictionary(content, vowels)
accented_vowels = ['ŕ', 'á', 'ä', 'é', 'ë', 'ě', 'í', 'î', 'ó', 'ô', 'ö', 'ú', 'ü']

data = Data('l', shuffle_all_inputs=False)
letter_location_model, syllable_location_model, syllabled_letters_location_model = data.load_location_models(
    'cnn/word_accetuation/cnn_dictionary/v5_3/20_final_epoch.h5',
    'cnn/word_accetuation/syllables/v3_3/20_final_epoch.h5',
    'cnn/word_accetuation/syllabled_letters/v3_3/20_final_epoch.h5')

letter_location_co_model, syllable_location_co_model, syllabled_letters_location_co_model = data.load_location_models(
    'cnn/word_accetuation/cnn_dictionary/v5_2/20_final_epoch.h5',
    'cnn/word_accetuation/syllables/v3_2/20_final_epoch.h5',
    'cnn/word_accetuation/syllabled_letters/v3_2/20_final_epoch.h5')

letter_type_model, syllable_type_model, syllabled_letter_type_model = data.load_type_models(
    'cnn/accent_classification/letters/v3_1/20_final_epoch.h5',
    'cnn/accent_classification/syllables/v2_1/20_final_epoch.h5',
    'cnn/accent_classification/syllabled_letters/v2_1/20_final_epoch.h5')

letter_type_co_model, syllable_type_co_model, syllabled_letter_type_co_model = data.load_type_models(
    'cnn/accent_classification/letters/v3_0/20_final_epoch.h5',
    'cnn/accent_classification/syllables/v2_0/20_final_epoch.h5',
    'cnn/accent_classification/syllabled_letters/v2_0/20_final_epoch.h5')

data = Data('s', shuffle_all_inputs=False)
new_content = data._read_content('data/sloleks-sl_v1.2.tbl')

print('Commencing accentuator!')

rate = 100000
start_timer = time.time()
with open("data/new_sloleks/new_sloleks.tab", "a") as myfile:
    for index in range(300000, len(new_content), rate):
        if index+rate >= len(new_content):
            words = [[el[0], '', el[2], el[0]] for el in new_content][index:len(new_content)]
        else:
            words = [[el[0], '', el[2], el[0]] for el in new_content][index:index+rate]
        data = Data('l', shuffle_all_inputs=False)
        location_accented_words, accented_words = data.accentuate_word(words, letter_location_model, syllable_location_model, syllabled_letters_location_model,
                                letter_location_co_model, syllable_location_co_model, syllabled_letters_location_co_model,
                                letter_type_model, syllable_type_model, syllabled_letter_type_model,
                                letter_type_co_model, syllable_type_co_model, syllabled_letter_type_co_model,
                                dictionary, max_word, max_num_vowels, vowels, accented_vowels, feature_dictionary, syllable_dictionary)

        res = ''
        for i in range(index, index + len(words)):
            res += new_content[i][0] + '\t' + new_content[i][1] + '\t' + new_content[i][2] + '\t' \
            + new_content[i][3][:-1] + '\t' + location_accented_words[i-index] + '\t' + accented_words[i-index] + '\n'

        print('Writing data from ' + str(index) + ' onward.')
        end_timer = time.time()
        print("Elapsed time: " + "{0:.2f}".format((end_timer - start_timer)/60.0) + " minutes")
        myfile.write(res)
