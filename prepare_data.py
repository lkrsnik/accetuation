# -*- coding: utf-8 -*-
from __future__ import unicode_literals
# text in Western (Windows 1252)

import numpy as np
import h5py
import math
import keras.backend as K
import os.path
import codecs

from copy import copy

from keras import optimizers
from keras.models import Model
from keras.layers import Dense, Dropout, Input
from keras.layers.merge import concatenate
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import Flatten
from keras.models import load_model


class Data:
    def __init__(self, input_type, allow_shuffle_vector_generation=False, save_generated_data=True, shuffle_all_inputs=True,
                 additional_letter_attributes=True, reverse_inputs=True, accent_classification=False, number_of_syllables=False,
                 convert_multext=True, bidirectional_basic_input=False, bidirectional_architectural_input=False):
        self._input_type = input_type
        self._save_generated_data = save_generated_data
        self._allow_shuffle_vector_generation = allow_shuffle_vector_generation
        self._shuffle_all_inputs = shuffle_all_inputs
        self._additional_letter_attributes = additional_letter_attributes
        self._reverse_inputs = reverse_inputs
        self._accent_classification = accent_classification
        self._number_of_syllables = number_of_syllables
        self._convert_multext = convert_multext
        self._bidirectional_basic_input = bidirectional_basic_input
        self._bidirectional_architectural_input = bidirectional_architectural_input

        self.x_train = None
        # self.x2_train = None
        self.x_other_features_train = None
        self.y_train = None
        self.x_test = None
        # self.x2_test = None
        self.x_other_features_test = None
        self.y_test = None
        self.x_validate = None
        # self.x2_validate = None
        self.x_other_features_validate = None
        self.y_validate = None

    def generate_data(self, train_inputs_name, test_inputs_name, validate_inputs_name, test_and_validation_size=0.1,
                      force_override=False, content_name='SlovarIJS_BESEDE_utf8.lex',
                      content_shuffle_vector='content_shuffle_vector', shuffle_vector='shuffle_vector',
                      inputs_location='../../internal_representations/inputs/', content_location='../../../data/',
                      test_set=False, complete_set=False):
        content_path = '{}{}'.format(content_location, content_name)
        train_path = '{}{}.h5'.format(inputs_location, train_inputs_name)
        test_path = '{}{}.h5'.format(inputs_location, test_inputs_name)
        validate_path = '{}{}.h5'.format(inputs_location, validate_inputs_name)
        if not force_override and os.path.exists(train_path) and os.path.exists(test_path) and os.path.exists(validate_path):
            print('LOADING DATA...')
            self.x_train, self.x_other_features_train, self.y_train = self._load_inputs(train_path)
            self.x_test, self.x_other_features_test, self.y_test = self._load_inputs(test_path)
            self.x_validate, self.x_other_features_validate, self.y_validate = self._load_inputs(validate_path)
            print('LOAD SUCCESSFUL!')
        else:
            content_shuffle_vector_path = '{}{}.h5'.format(inputs_location, content_shuffle_vector)
            shuffle_vector_path = '{}{}'.format(inputs_location, shuffle_vector)

            # actual generation of inputs
            self._generate_inputs(content_path, content_shuffle_vector_path, shuffle_vector_path, test_and_validation_size, train_path, test_path,
                                  validate_path)
        if test_set:
            self.x_train = np.concatenate((self.x_train, self.x_test), axis=0)
            self.x_other_features_train = np.concatenate((self.x_other_features_train, self.x_other_features_test), axis=0)
            self.y_train = np.concatenate((self.y_train, self.y_test), axis=0)

            self.x_test = self.x_validate
            self.x_other_features_test = self.x_other_features_validate
            self.y_test = self.y_validate

        if complete_set:
            self.x_train = np.concatenate((self.x_train, self.x_test, self.x_validate), axis=0)
            self.x_other_features_train = np.concatenate((self.x_other_features_train, self.x_other_features_test, self.x_other_features_validate),
                                                         axis=0)
            self.y_train = np.concatenate((self.y_train, self.y_test, self.y_validate), axis=0)

            self.x_test = self.x_validate
            self.x_other_features_test = self.x_other_features_validate
            self.y_test = self.y_validate

    def _generate_inputs(self, content_location, content_shuffle_vector_location, shuffle_vector_location, test_and_validation_size, train_path,
                         test_path, validate_path):
        print('READING CONTENT...')
        content = self._read_content(content_location)
        print('CONTENT READ SUCCESSFULLY')
        print('CREATING DICTIONARY...')
        dictionary, max_word, max_num_vowels, vowels, accented_vowels = self._create_dict(content)
        if self._input_type == 's' or self._input_type == 'sl':
            dictionary = self._create_syllables_dictionary(content, vowels)
        print('DICTIONARY CREATION SUCCESSFUL!')
        # test_and_validation_size = 0.1
        train_content, test_content, validate_content = self._split_content(content, test_and_validation_size, content_shuffle_vector_location)
        feature_dictionary = self._create_feature_dictionary()

        # Generate X and y
        print('GENERATING X AND y...')
        self.x_train, self.x_other_features_train, self.y_train = self._generate_x_and_y(dictionary, max_word, max_num_vowels, train_content, vowels,
                                                                                         accented_vowels,
                                                                                         feature_dictionary, shuffle_vector_location + '_train.h5')
        self.x_test, self.x_other_features_test, self.y_test = self._generate_x_and_y(dictionary, max_word, max_num_vowels, test_content, vowels,
                                                                                      accented_vowels,
                                                                                      feature_dictionary, shuffle_vector_location + '_test.h5')
        self.x_validate, self.x_other_features_validate, self.y_validate = self._generate_x_and_y(dictionary, max_word, max_num_vowels,
                                                                                                  validate_content, vowels,
                                                                                                  accented_vowels, feature_dictionary,
                                                                                                  shuffle_vector_location + '_validate.h5')
        print('GENERATION SUCCESSFUL!')

        # save inputs
        if self._save_generated_data:
            self._save_inputs(train_path, self.x_train, self.x_other_features_train, self.y_train)
            self._save_inputs(test_path, self.x_test, self.x_other_features_test, self.y_test)
            self._save_inputs(validate_path, self.x_validate, self.x_other_features_validate, self.y_validate)

        # return X_train, X_other_features_train, y_train, X_test, X_other_features_test, y_test, X_validate, X_other_features_validate, y_validate

    # functions for creating X and y from content
    @staticmethod
    def _read_content(content_path):
        # with open(content_path) as f:
        with codecs.open(content_path, encoding='utf8') as f:
            content = f.readlines()
        return [x.split('\t') for x in content]

    def _create_dict(self, content):
        # CREATE dictionary AND max_word
        accented_vowels = self._get_accented_vowels()
        unaccented_vowels = self._get_unaccented_vowels()
        vowels = []
        vowels.extend(accented_vowels)
        vowels.extend(unaccented_vowels)

        dictionary_input = ['']
        line = 0
        max_word = 0
        # ADD 'EMPTY' VOWEL
        max_num_vowels = 0
        for el in content:
            num_vowels = 0
            try:
                if len(el[3]) > max_word:
                    max_word = len(el[3])
                if len(el[0]) > max_word:
                    max_word = len(el[0])
                for i in range(len(el[3])):
                    if self._is_vowel(list(el[3]), i, vowels):
                        num_vowels += 1
                for c in list(el[0]):
                    if c not in dictionary_input:
                        dictionary_input.append(c)
                if num_vowels > max_num_vowels:
                    max_num_vowels = num_vowels
            except Exception:
                print(line - 1)
                print(el)
                break
            line += 1
        dictionary_input = sorted(dictionary_input)
        # max_num_vowels += 1
        return dictionary_input, max_word, max_num_vowels, vowels, accented_vowels

    # split content so that there is no overfitting
    def _split_content(self, content, test_and_validation_ratio, content_shuffle_vector_location):
        expanded_content = [el[1] if el[1] != '=' else el[0] for el in content]
        # print(len(content))
        unique_content = sorted(set(expanded_content))

        s = self._load_shuffle_vector(content_shuffle_vector_location, len(unique_content))

        test_num = math.floor(len(unique_content) * (test_and_validation_ratio * 2))
        validation_num = math.floor(test_num * 0.5)
        shuffled_unique_train_content = [unique_content[i] for i in range(len(s)) if s[i] >= test_num]
        shuffled_unique_train_content_set = set(shuffled_unique_train_content)

        shuffled_unique_test_content = [unique_content[i] for i in range(len(s)) if test_num > s[i] >= validation_num]
        shuffled_unique_test_content_set = set(shuffled_unique_test_content)

        shuffled_unique_validate_content = [unique_content[i] for i in range(len(s)) if s[i] < validation_num]
        shuffled_unique_validate_content_set = set(shuffled_unique_validate_content)

        train_content = [content[i] for i in range(len(content)) if expanded_content[i] in shuffled_unique_train_content_set]
        test_content = [content[i] for i in range(len(content)) if expanded_content[i] in shuffled_unique_test_content_set]
        validate_content = [content[i] for i in range(len(content)) if expanded_content[i] in shuffled_unique_validate_content_set]
        return train_content, test_content, validate_content

    @staticmethod
    def _create_and_save_shuffle_vector(file_name, length):
        shuffle_vector = np.arange(length)
        np.random.shuffle(shuffle_vector)
        h5f = h5py.File(file_name, 'w')
        adict = dict(shuffle_vector=shuffle_vector)
        for k, v in adict.items():
            h5f.create_dataset(k, data=v)
        h5f.close()
        return shuffle_vector

    def _x_letter_input(self, content, dictionary, max_word, vowels, shuffle_vector_location):
        if self._additional_letter_attributes:
            if not self._bidirectional_basic_input:
                x = np.zeros((len(content), max_word, len(dictionary) + 6), dtype=int)
            else:
                x = np.zeros((len(content), 2 * max_word, len(dictionary) + 6), dtype=int)
            voiced_consonants = self._get_voiced_consonants()
            resonant_silent_consonants = self._get_resonant_silent_consonants()
            nonresonant_silent_consonants = self._get_nonresonant_silent_consonants()
            # print('HERE!!!')
        else:
            # print('HERE!!!')
            if not self._bidirectional_basic_input:
                x = np.zeros((len(content), max_word, len(dictionary)), dtype=int)
            else:
                x = np.zeros((len(content), 2 * max_word, len(dictionary)), dtype=int)

        if self._shuffle_all_inputs:
            s = self._load_shuffle_vector(shuffle_vector_location, len(content))
        else:
            s = None

        # i = 0
        for i in range(len(content)):
            if self._shuffle_all_inputs:
                mod_i = s[i]
            else:
                mod_i = i
            word = content[mod_i][0]
            if self._reverse_inputs:
                word = word[::-1]
            j = 0
            for c in list(word):
                if j >= max_word:
                    continue
                index = 0
                if self._bidirectional_basic_input:
                    j2 = max_word + (len(word) - j - 1)
                for d in dictionary:
                    if c == d:
                        x[i][j][index] = 1
                        if self._bidirectional_basic_input:
                            x[i][j2][index] = 1
                        break
                    index += 1
                if self._additional_letter_attributes:
                    if self._is_vowel(word, j, vowels):
                        x[i][j][len(dictionary)] = 1
                        if self._bidirectional_basic_input:
                            x[i][j2][len(dictionary)] = 1
                    else:
                        x[i][j][len(dictionary) + 1] = 1
                        if self._bidirectional_basic_input:
                            x[i][j2][len(dictionary) + 1] = 1
                        if c in voiced_consonants:
                            x[i][j][len(dictionary) + 2] = 1
                            if self._bidirectional_basic_input:
                                x[i][j2][len(dictionary) + 2] = 1
                        else:
                            x[i][j][len(dictionary) + 3] = 1
                            if self._bidirectional_basic_input:
                                x[i][j2][len(dictionary) + 3] = 1

                            if c in resonant_silent_consonants:
                                x[i][j][len(dictionary) + 4] = 1
                                if self._bidirectional_basic_input:
                                    x[i][j2][len(dictionary) + 4] = 1
                            elif c in nonresonant_silent_consonants:
                                x[i][j][len(dictionary) + 5] = 1
                                if self._bidirectional_basic_input:
                                    x[i][j2][len(dictionary) + 5] = 1
                j += 1
            #i += 1
        return x

    def _x_syllable_input(self, content, dictionary, max_num_vowels, vowels, shuffle_vector_location):
        if not self._bidirectional_basic_input:
            x = np.zeros((len(content), max_num_vowels), dtype=int)
        else:
            x = np.zeros((len(content), 2 * max_num_vowels), dtype=int)

        if self._shuffle_all_inputs:
            s = self._load_shuffle_vector(shuffle_vector_location, len(content))
        else:
            s = None

        for i in range(len(content)):
            if self._shuffle_all_inputs:
                mod_i = s[i]
            else:
                mod_i = i
            j = 0
            syllables = self._create_syllables(content[mod_i][0], vowels)
            if self._reverse_inputs:
                syllables = syllables[::-1]
            for syllable in syllables:
                if j >= max_num_vowels:
                    continue
                if syllable in dictionary:
                    x[i][j] = dictionary.index(syllable)
                    if self._bidirectional_basic_input:
                        x[i][max_num_vowels + (len(syllables) - j - 1)] = dictionary.index(syllable)
                else:
                    x[i][j] = 0
                j += 1
            #i += 1
        return x

    def _y_output(self, content, max_num_vowels, vowels, accentuated_vowels, shuffle_vector_location):
        y = np.zeros((len(content), max_num_vowels))
        i = 0
        if self._shuffle_all_inputs:
            s = self._load_shuffle_vector(shuffle_vector_location, len(content))
        else:
            s = None
        for i in range(len(content)):
            if self._shuffle_all_inputs:
                mod_i = s[i]
            else:
                mod_i = i
            el = content[mod_i]
            word = el[3]
            if self._reverse_inputs:
                word = word[::-1]

            j = 0
            # word_accentuations = []
            num_vowels = 0
            for c in list(word):
                index = 0
                for d in accentuated_vowels:
                    if c == d:
                        if not self._accent_classification:
                            y[i][num_vowels] = 1
                        else:
                            y[i][num_vowels] = index
                        # word_accentuations.append(num_vowels)
                        break
                    index += 1
                if self._is_vowel(word, j, vowels):
                    num_vowels += 1
                j += 1
        return y

    # Generate each y as an array of 11 numbers (with possible values between 0 and 1)
    def _generate_x_and_y(self, dictionary, max_word, max_num_vowels, content, vowels, accentuated_vowels, feature_dictionary,
                          shuffle_vector_location):
        if self._input_type == 'l':
            x = self._x_letter_input(content, dictionary, max_word, vowels, shuffle_vector_location)
        elif self._input_type == 's' or self._input_type == 'sl':
            x = self._x_syllable_input(content, dictionary, max_num_vowels, vowels, shuffle_vector_location)
        else:
            raise ValueError('No input_type provided. It could be \'l\', \'s\' or \'sl\'.')
        y = self._y_output(content, max_num_vowels, vowels, accentuated_vowels, shuffle_vector_location)

        # print('CREATING OTHER FEATURES...')
        x_other_features = self._create_x_features(content, feature_dictionary, vowels, shuffle_vector_location)
        # print('OTHER FEATURES CREATED!')

        if self._shuffle_all_inputs:
            print('SHUFFELING INPUTS...')
            #x, x_other_features, y = self._shuffle_inputs(x, x_other_features, y, shuffle_vector_location)
            print('INPUTS SHUFFELED!')
        return x, x_other_features, y

    def _create_syllables_dictionary(self, content, vowels):
        dictionary = []
        for el in content:
            syllables = self._create_syllables(el[0], vowels)
            for syllable in syllables:
                if syllable not in dictionary:
                    dictionary.append(syllable)
        dictionary.append('')
        return sorted(dictionary)

    def _create_syllables(self, word, vowels):
        word_list = list(word)
        consonants = []
        syllables = []
        for i in range(len(word_list)):
            if self._is_vowel(word_list, i, vowels):
                if syllables == []:
                    consonants.append(word_list[i])
                    syllables.append(''.join(consonants))
                else:
                    left_consonants, right_consonants = self._split_consonants(list(''.join(consonants).lower()))
                    syllables[-1] += ''.join(left_consonants)
                    right_consonants.append(word_list[i])
                    syllables.append(''.join(right_consonants))
                consonants = []
            else:
                consonants.append(word_list[i])
        if len(syllables) < 1:
            return word
        syllables[-1] += ''.join(consonants)

        return syllables

    def _is_vowel(self, word_list, position, vowels):
        if word_list[position] in vowels:
            return True
        if (word_list[position] == u'r' or word_list[position] == u'R') and (position - 1 < 0 or word_list[position - 1] not in vowels) and (
                            position + 1 >= len(word_list) or word_list[position + 1] not in vowels):
            return True
        return False

    def _split_consonants(self, consonants):
        voiced_consonants = self._get_voiced_consonants()
        resonant_silent_consonants = self._get_resonant_silent_consonants()
        unresonant_silent_consonants = self._get_nonresonant_silent_consonants()
        if len(consonants) == 0:
            return [''], ['']
        elif len(consonants) == 1:
            return [''], consonants
        else:
            split_options = []
            for i in range(len(consonants) - 1):
                if consonants[i] == '-' or consonants[i] == '_':
                    split_options.append([i, -1])
                elif consonants[i] == consonants[i + 1]:
                    split_options.append([i, 0])
                elif consonants[i] in voiced_consonants:
                    if consonants[i + 1] in resonant_silent_consonants or consonants[i + 1] in unresonant_silent_consonants:
                        split_options.append([i, 2])
                elif consonants[i] in resonant_silent_consonants:
                    if consonants[i + 1] in resonant_silent_consonants:
                        split_options.append([i, 1])
                    elif consonants[i + 1] in unresonant_silent_consonants:
                        split_options.append([i, 3])
                elif consonants[i] in unresonant_silent_consonants:
                    if consonants[i + 1] in resonant_silent_consonants:
                        split_options.append([i, 4])

            if split_options == []:
                return [''], consonants
            else:
                split = min(split_options, key=lambda x: x[1])
                return consonants[:split[0] + 1], consonants[split[0] + 1:]

    def _create_x_features(self, content, feature_dictionary, vowels, shuffle_vector_location):
        content = content
        x_other_features = []
        if self._shuffle_all_inputs:
            s = self._load_shuffle_vector(shuffle_vector_location, len(content))
        else:
            s = None
        for index in range(len(content)):
            if self._shuffle_all_inputs:
                mod_i = s[index]
            else:
                mod_i = index
            el = content[mod_i]
            x_el_other_features = []
            if self._convert_multext:
                converted_el = ''.join(self._convert_to_multext_east_v4(list(el[2]), feature_dictionary))
            else:
                converted_el = el[2]
            for feature in feature_dictionary:
                if converted_el[0] == feature[1]:
                    x_el_other_features.append(1)
                    for i in range(2, len(feature)):
                        for j in range(len(feature[i])):
                            if i - 1 < len(converted_el) and feature[i][j] == converted_el[i - 1]:
                                x_el_other_features.append(1)
                            else:
                                x_el_other_features.append(0)
                else:
                    x_el_other_features.extend([0] * feature[0])
            if self._number_of_syllables:
                list_of_letters = list(el[0])
                num_of_vowels = 0
                for i in range(len(list_of_letters)):
                    if self._is_vowel(list(el[0]), i, vowels):
                        num_of_vowels += 1
                x_el_other_features.append(num_of_vowels)

            x_other_features.append(x_el_other_features)
        return np.array(x_other_features)

    def _shuffle_inputs(self, x, x_other_features, y, shuffle_vector_location):
        s = self._load_shuffle_vector(shuffle_vector_location, x.shape[0])
        x = x[s]
        y = y[s]
        x_other_features = x_other_features[s]
        return x, x_other_features, y

    # functions for saving, loading and shuffling whole arrays to ram
    @staticmethod
    def _save_inputs(file_name, x, x_other_features, y):
        h5f = h5py.File(file_name, 'w')
        a_dict = dict(X=x, X_other_features=x_other_features, y=y)
        for k, v in a_dict.items():
            h5f.create_dataset(k, data=v)
        h5f.close()

    @staticmethod
    def _load_inputs(file_name):
        h5f = h5py.File(file_name, 'r')
        x = h5f['X'][:]
        y = h5f['y'][:]
        x_other_features = h5f['X_other_features'][:]
        h5f.close()
        return x, x_other_features, y

    def _load_shuffle_vector(self, file_path, length=0):
        if os.path.exists(file_path):
            h5f = h5py.File(file_path, 'r')
            shuffle_vector = h5f['shuffle_vector'][:]
            h5f.close()
        else:
            if self._allow_shuffle_vector_generation:
                shuffle_vector = self._create_and_save_shuffle_vector(file_path, length)
            else:
                raise ValueError('Shuffle vector on path: \'{}\' does not exist! Either generate new vector (with initializing new Data object with '
                                 'parameter allow_shuffle_vector_generation=True or paste one that is already generated!'.format(file_path))
        return shuffle_vector

    @staticmethod
    def _convert_to_multext_east_v4(old_features, feature_dictionary):
        new_features = ['-'] * 9
        new_features[:len(old_features)] = old_features
        if old_features[0] == 'A':
            if old_features[1] == 'f' or old_features[1] == 'o':
                new_features[1] = 'g'
            return new_features[:len(feature_dictionary[0]) - 1]
        if old_features[0] == 'C':
            return new_features[:len(feature_dictionary[1]) - 1]
        if old_features[0] == 'I':
            return new_features[:len(feature_dictionary[2]) - 1]
        if old_features[0] == 'M':
            new_features[2:6] = old_features[1:5]
            new_features[1] = old_features[5]
            if new_features[2] == 'm':
                new_features[2] = '-'
            return new_features[:len(feature_dictionary[3]) - 1]
        if old_features[0] == 'N':
            if len(old_features) >= 7:
                new_features[5] = old_features[7]
            return new_features[:len(feature_dictionary[4]) - 1]
        if old_features[0] == 'P':
            if new_features[8] == 'n':
                new_features[8] = 'b'
            return new_features[:len(feature_dictionary[5]) - 1]
        if old_features[0] == 'Q':
            return new_features[:len(feature_dictionary[6]) - 1]
        if old_features[0] == 'R':
            return new_features[:len(feature_dictionary[7]) - 1]
        if old_features[0] == 'S':
            if len(old_features) == 4:
                new_features[1] = old_features[3]
            else:
                new_features[1] = '-'
            return new_features[:len(feature_dictionary[8]) - 1]
        if old_features[0] == 'V':
            if old_features[1] == 'o' or old_features[1] == 'c':
                new_features[1] = 'm'
            new_features[3] = old_features[2]
            new_features[2] = '-'
            if old_features[2] == 'i':
                new_features[3] = 'r'
            if len(old_features) > 3 and old_features[3] == 'p':
                new_features[3] = 'r'
            elif len(old_features) > 3 and old_features[3] == 'f':
                new_features[3] = 'f'
            if len(old_features) >= 9:
                new_features[7] = old_features[8]
            else:
                new_features[7] = '-'
            return new_features[:len(feature_dictionary[9]) - 1]
        return ''

    # generator for inputs for tracking of data fitting
    def generator(self, data_type, batch_size, x=None, x_other_features_validate=None, y_validate=None, content_name='SlovarIJS_BESEDE_utf8.lex',
                  content_location='../../../data/'):
        content_path = '{}{}'.format(content_location, content_name)
        if data_type == 'train':
            return self._generator_instance(self.x_train, self.x_other_features_train, self.y_train, batch_size, content_path)
        elif data_type == 'test':
            return self._generator_instance(self.x_test, self.x_other_features_test, self.y_test, batch_size, content_path)
        elif data_type == 'validate':
            return self._generator_instance(self.x_validate, self.x_other_features_validate, self.y_validate, batch_size, content_path)
        else:
            return self._generator_instance(x, x_other_features_validate, y_validate, batch_size)

            # if self._input_type

    def _generator_instance(self, orig_x, orig_x_additional, orig_y, batch_size, content_path):
        if self._input_type == 'l':
            content = self._read_content(content_path)
            dictionary, max_word, max_num_vowels, vowels, accented_vowels = self._create_dict(content)
            return self._letter_generator(orig_x, orig_x_additional, orig_y, batch_size, accented_vowels)
        elif self._input_type == 's':
            content = self._read_content(content_path)
            dictionary, max_word, max_num_vowels, vowels, accented_vowels = self._create_dict(content)
            syllable_dictionary = self._create_syllables_dictionary(content, vowels)
            eye = np.eye(len(syllable_dictionary), dtype=int)
            return self._syllable_generator(orig_x, orig_x_additional, orig_y, batch_size, eye, accented_vowels)
        elif self._input_type == 'sl':
            content = self._read_content(content_path)
            dictionary, max_word, max_num_vowels, vowels, accented_vowels = self._create_dict(content)
            syllable_dictionary = self._create_syllables_dictionary(content, vowels)
            max_syllable = self._get_max_syllable(syllable_dictionary)
            syllable_letters_translator = self._create_syllable_letters_translator(max_syllable, syllable_dictionary, dictionary, vowels)
            return self._syllable_generator(orig_x, orig_x_additional, orig_y, batch_size, syllable_letters_translator, accented_vowels)

    # generator for inputs for tracking of data fitting
    def _letter_generator(self, orig_x, orig_x_additional, orig_y, batch_size, accented_vowels):
        size = orig_x.shape[0]
        while 1:
            loc = 0
            if self._accent_classification:
                eye = np.eye(len(accented_vowels), dtype=int)
                eye_input_accent = np.eye(len(orig_y[0]), dtype=int)
                input_x_stack = []
                input_x_other_features_stack = []
                input_y_stack = []
                while loc < size:
                    while len(input_x_stack) < batch_size and loc < size:
                        accent_loc = 0
                        for accent in orig_y[loc]:
                            if accent > 0:
                                new_orig_x_additional = orig_x_additional[loc]
                                new_orig_x_additional = np.concatenate((new_orig_x_additional, eye_input_accent[accent_loc]))
                                input_x_stack.append(orig_x[loc])
                                input_x_other_features_stack.append(new_orig_x_additional)
                                input_y_stack.append(eye[int(accent)])
                            accent_loc += 1
                        loc += 1
                    if len(input_x_stack) > batch_size:
                        yield ([np.array(input_x_stack[:batch_size]),
                                np.array(input_x_other_features_stack[:batch_size])], np.array(input_y_stack)[:batch_size])
                        input_x_stack = input_x_stack[batch_size:]
                        input_x_other_features_stack = input_x_other_features_stack[batch_size:]
                        input_y_stack = input_y_stack[batch_size:]
                    else:
                        # print('BBB')
                        # print(np.array(input_stack))
                        # yield (np.array(input_stack))
                        yield ([np.array(input_x_stack), np.array(input_x_other_features_stack)], np.array(input_y_stack))
                        input_x_stack = []
                        input_x_other_features_stack = []
                        input_y_stack = []
            else:
                while loc < size:
                    if loc + batch_size >= size:
                        if self._bidirectional_architectural_input:
                            split_orig_x = np.hsplit(orig_x[loc:size], 2)
                            yield ([split_orig_x[0], split_orig_x[1], orig_x_additional[loc:size]], orig_y[loc:size])
                        else:
                            yield ([orig_x[loc:size], orig_x_additional[loc:size]], orig_y[loc:size])
                    else:
                        if self._bidirectional_architectural_input:
                            split_orig_x = np.hsplit(orig_x[loc:loc + batch_size], 2)
                            yield ([split_orig_x[0], split_orig_x[1], orig_x_additional[loc:loc + batch_size]], orig_y[loc:loc + batch_size])
                        else:
                            yield ([orig_x[loc:loc + batch_size], orig_x_additional[loc:loc + batch_size]], orig_y[loc:loc + batch_size])
                    loc += batch_size

    # generator for inputs for tracking of data fitting
    def _syllable_generator(self, orig_x, orig_x_additional, orig_y, batch_size, translator, accented_vowels):
        size = orig_x.shape[0]
        while 1:
            loc = 0
            if self._accent_classification:
                eye = np.eye(len(accented_vowels), dtype=int)
                eye_input_accent = np.eye(len(orig_y[0]), dtype=int)
                input_x_stack = []
                input_x_other_features_stack = []
                input_y_stack = []
                while loc < size:
                    while len(input_x_stack) < batch_size and loc < size:
                        accent_loc = 0
                        for accent in orig_y[loc]:
                            if accent > 0:
                                new_orig_x_additional = orig_x_additional[loc]
                                new_orig_x_additional = np.concatenate((new_orig_x_additional, eye_input_accent[accent_loc]))
                                input_x_stack.append(orig_x[loc])
                                input_x_other_features_stack.append(new_orig_x_additional)
                                input_y_stack.append(eye[int(accent)])
                            accent_loc += 1
                        loc += 1
                    if len(input_x_stack) > batch_size:
                        gen_orig_x = translator[np.array(input_x_stack[:batch_size])]

                        if self._bidirectional_architectural_input:
                            split_orig_x = np.hsplit(gen_orig_x, 2)
                            yield ([split_orig_x[0], split_orig_x[1], np.array(input_x_other_features_stack[:batch_size])],
                                   np.array(input_y_stack)[:batch_size])
                        else:
                            yield ([gen_orig_x, np.array(input_x_other_features_stack[:batch_size])], np.array(input_y_stack)[:batch_size])

                        # yield ([gen_orig_x, np.array(input_x_other_features_stack[:batch_size])], np.array(input_y_stack)[:batch_size])
                        input_x_stack = input_x_stack[batch_size:]
                        input_x_other_features_stack = input_x_other_features_stack[batch_size:]
                        input_y_stack = input_y_stack[batch_size:]
                    else:
                        #print('-------------------------------------------------------------------------------------------')
                        #if dictionary is not None:
                        #    print(self.decode_x(word_encoded, dictionary))
                        #print(input_x_stack)
                        #print(input_x_other_features_stack)
                        #print(input_y_stack)
                        #print(loc)
                        if len(input_x_stack) == 0:
                            continue
                        gen_orig_x = translator[np.array(input_x_stack)]

                        if self._bidirectional_architectural_input:
                            split_orig_x = np.hsplit(gen_orig_x, 2)
                            yield ([split_orig_x[0], split_orig_x[1], np.array(input_x_other_features_stack)],
                                   np.array(input_y_stack))
                        else:
                            yield ([gen_orig_x, np.array(input_x_other_features_stack)], np.array(input_y_stack))

                        # yield ([gen_orig_x, np.array(input_x_other_features_stack)], np.array(input_y_stack))
                        input_x_stack = []
                        input_x_other_features_stack = []
                        input_y_stack = []
            else:
                while loc < size:
                    if loc + batch_size >= size:
                        gen_orig_x = translator[orig_x[loc:size]]

                        if self._bidirectional_architectural_input:
                            split_orig_x = np.hsplit(gen_orig_x, 2)
                            yield ([split_orig_x[0], split_orig_x[1], orig_x_additional[loc:size]], orig_y[loc:size])
                        else:
                            yield ([gen_orig_x, orig_x_additional[loc:size]], orig_y[loc:size])

                        #yield ([gen_orig_x, orig_x_additional[loc:size]], orig_y[loc:size])
                    else:
                        gen_orig_x = translator[orig_x[loc:loc + batch_size]]

                        if self._bidirectional_architectural_input:
                            split_orig_x = np.hsplit(gen_orig_x, 2)
                            yield ([split_orig_x[0], split_orig_x[1], orig_x_additional[loc:loc + batch_size]], orig_y[loc:loc + batch_size])
                        else:
                            yield ([gen_orig_x, orig_x_additional[loc:loc + batch_size]], orig_y[loc:loc + batch_size])

                        #yield ([gen_orig_x, orig_x_additional[loc:loc + batch_size]], orig_y[loc:loc + batch_size])
                    loc += batch_size

    def _get_max_syllable(self, syllable_dictionary):
        max_len = 0
        for el in syllable_dictionary:
            if len(el) > max_len:
                max_len = len(el)
        return max_len

    def _create_syllable_letters_translator(self, max_syllable, syllable_dictionary, dictionary, vowels, aditional_letter_attributes=True):
        if aditional_letter_attributes:
            voiced_consonants = self._get_voiced_consonants()
            resonant_silent_consonants = self._get_resonant_silent_consonants()
            nonresonant_silent_consonants = self._get_nonresonant_silent_consonants()

        syllable_letters_translator = []
        for syllable in syllable_dictionary:
            di_syllable = []
            for let in range(max_syllable):
                # di_let = []
                for a in dictionary:
                    if let < len(syllable) and a == list(syllable)[let]:
                        di_syllable.append(1)
                    else:
                        di_syllable.append(0)

                if aditional_letter_attributes:
                    if let >= len(syllable):
                        di_syllable.extend([0, 0, 0, 0, 0, 0])
                    elif self._is_vowel(list(syllable), let, vowels):
                        di_syllable.extend([1, 0, 0, 0, 0, 0])
                    else:
                        # X[i][j][len(dictionary) + 1] = 1
                        if list(syllable)[let] in voiced_consonants:
                            # X[i][j][len(dictionary) + 2] = 1
                            di_syllable.extend([0, 1, 1, 0, 0, 0])
                        else:
                            # X[i][j][len(dictionary) + 3] = 1
                            if list(syllable)[let] in resonant_silent_consonants:
                                # X[i][j][len(dictionary) + 4] = 1
                                di_syllable.extend([0, 1, 0, 1, 1, 0])
                            elif list(syllable)[let] in nonresonant_silent_consonants:
                                # X[i][j][len(dictionary) + 5] = 1
                                di_syllable.extend([0, 1, 0, 1, 0, 1])
                            else:
                                di_syllable.extend([0, 0, 0, 0, 0, 0])
                                # di_syllable.append(di_let)
            syllable_letters_translator.append(di_syllable)
        syllable_letters_translator = np.array(syllable_letters_translator, dtype=int)
        return syllable_letters_translator

    @staticmethod
    def _get_accented_vowels():
        return [u'à', u'á', u'ä', u'é', u'ë', u'ì', u'í', u'î', u'ó', u'ô', u'ö', u'ú', u'ü']

    @staticmethod
    def _get_unaccented_vowels():
        return [u'a', u'e', u'i', u'o', u'u']

    @staticmethod
    def _get_voiced_consonants():
        return ['m', 'n', 'v', 'l', 'r', 'j', 'y', 'w']

    @staticmethod
    def _get_resonant_silent_consonants():
        return ['b', 'd', 'z', 'ž', 'g']

    @staticmethod
    def _get_nonresonant_silent_consonants():
        return ['p', 't', 's', 'š', 'č', 'k', 'f', 'h', 'c']

    @staticmethod
    def _create_slovene_feature_dictionary():
        # old: http://nl.ijs.si/ME/Vault/V3/msd/html/
        # new: http://nl.ijs.si/ME/V4/msd/html/
        # changes: http://nl.ijs.si/jos/msd/html-en/msd.diffs.html
        return [[21,
                 'P',
                 ['p', 's'],
                 ['n', 'p', 's'],
                 ['m', 'z', 's'],
                 ['e', 'd', 'm'],
                 ['i', 'r', 'd', 't', 'm', 'o'],
                 ['-', 'n', 'd']],
                [3, 'V', ['p', 'd']],
                [1, 'M'],
                [21,
                 'K',
                 ['b'],
                 ['-', 'g', 'v', 'd'],
                 ['m', 'z', 's'],
                 ['e', 'd', 'm'],
                 ['i', 'r', 'd', 't', 'm', 'o'],
                 ['-', 'n', 'd']],
                [17,
                 'S',
                 ['o'],
                 ['m', 'z', 's'],
                 ['e', 'd', 'm'],
                 ['i', 'r', 'd', 't', 'm', 'o'],
                 ['-', 'n', 'd']],
                [40,
                 'Z',
                 ['o', 's', 'k', 'z', 'p', 'c', 'v', 'n', 'l'],
                 ['-', 'p', 'd', 't'],
                 ['-', 'm', 'z', 's'],
                 ['-', 'e', 'd', 'm'],
                 ['-', 'i', 'r', 'd', 't', 'm', 'o'],
                 ['-', 'e', 'd', 'm'],
                 ['-', 'm', 'z', 's'],
                 ['-', 'k', 'z']],
                [1, 'L'],
                [5, 'R', ['s'], ['n', 'r', 's']],
                [7, 'D', ['-', 'r', 'd', 't', 'm', 'o']],
                [24,
                 'G',
                 ['g'],
                 ['-'],
                 ['n', 'm', 'd', 's', 'p', 'g'],
                 ['-', 'p', 'd', 't'],
                 ['-', 'e', 'm', 'd'],
                 ['-', 'm', 'z', 's'],
                 ['-', 'n', 'd']]
                ]

    @staticmethod
    def _create_feature_dictionary():
        # old: http://nl.ijs.si/ME/Vault/V3/msd/html/
        # new: http://nl.ijs.si/ME/V4/msd/html/
        # changes: http://nl.ijs.si/jos/msd/html-en/msd.diffs.html
        return [[21,
                 'A',
                 ['g', 's'],
                 ['p', 'c', 's'],
                 ['m', 'f', 'n'],
                 ['s', 'd', 'p'],
                 ['n', 'g', 'd', 'a', 'l', 'i'],
                 ['-', 'n', 'y']],
                [3, 'C', ['c', 's']],
                [1, 'I'],
                [21,
                 'M',
                 ['l'],
                 ['-', 'c', 'o', 's'],
                 ['m', 'f', 'n'],
                 ['s', 'd', 'p'],
                 ['n', 'g', 'd', 'a', 'l', 'i'],
                 ['-', 'n', 'y']],
                [17,
                 'N',
                 ['c'],
                 ['m', 'f', 'n'],
                 ['s', 'd', 'p'],
                 ['n', 'g', 'd', 'a', 'l', 'i'],
                 ['-', 'n', 'y']],
                [40,
                 'P',
                 ['p', 's', 'd', 'r', 'x', 'g', 'q', 'i', 'z'],
                 ['-', '1', '2', '3'],
                 ['-', 'm', 'f', 'n'],
                 ['-', 's', 'd', 'p'],
                 ['-', 'n', 'g', 'd', 'a', 'l', 'i'],
                 ['-', 's', 'd', 'p'],
                 ['-', 'm', 'f', 'n'],
                 ['-', 'y', 'b']],
                [1, 'Q'],
                [5, 'R', ['g'], ['p', 'c', 's']],
                [7, 'S', ['-', 'g', 'd', 'a', 'l', 'i']],
                [24,
                 'V',
                 ['m'],
                 ['-'],
                 ['n', 'u', 'p', 'r', 'f', 'c'],
                 ['-', '1', '2', '3'],
                 ['-', 's', 'p', 'd'],
                 ['-', 'm', 'f', 'n'],
                 ['-', 'n', 'y']]
                ]

    # Decoders for inputs and outputs
    @staticmethod
    def decode_x(word_encoded, dictionary):
        word = ''
        for el in word_encoded:
            i = 0
            for num in el:
                if num == 1:
                    word += dictionary[i]
                    break
                i += 1
        return word

    @staticmethod
    def decode_x_other_features(feature_dictionary, x_other_features):
        final_word = []
        for word in x_other_features:
            final_word = []
            i = 0
            for z in range(len(feature_dictionary)):
                for j in range(1, len(feature_dictionary[z])):
                    if j == 1:
                        if word[i] == 1:
                            final_word.append(feature_dictionary[z][1])
                        i += 1
                    else:
                        for k in range(len(feature_dictionary[z][j])):
                            if word[i] == 1:
                                final_word.append(feature_dictionary[z][j][k])
                            i += 1
                            # print(u''.join(final_word))
        return u''.join(final_word)

    @staticmethod
    def decode_y(y):
        i = 0
        res = []
        for el in y:
            if el >= 0.5:
                res.append(i)
            i += 1
        return res

    def test_accuracy(self, predictions, x, x_other_features, y, dictionary, feature_dictionary, vowels, syllable_dictionary=None,
                      threshold=0.4999955, patterns=None):
        errors = []
        num_of_pred = len(predictions)
        num_of_correct_pred = 0

        # wrong_patterns = 0
        # wrong_pattern_prediction = 0
        for i in range(predictions.shape[0]):
            correct_prediction = True

            round_predictions = np.zeros(predictions[i].shape)
            for j in range(len(y[i])):
                if predictions[i][j] < threshold:
                    round_predictions[j] = 0.0
                else:
                    round_predictions[j] = 1.0
                if (predictions[i][j] < threshold and y[i][j] == 1.0) or (predictions[i][j] >= threshold and y[i][j] == 0.0):
                    correct_prediction = False

            # in_pattern = False
            # if patterns is not None:
            #     test_predictions = copy(predictions[i])
            #     l = self.get_word_length(x[i])
            #     round_predictions = np.zeros(test_predictions.shape)
            #     for j in range(len(y[i])):
            #         if test_predictions[j] < threshold:
            #             round_predictions[j] = 0.0
            #         else:
            #             round_predictions[j] = 1.0
            #
            #     in_pattern = False
            #     for pattern in patterns[l]:
            #         if (pattern == round_predictions).all():
            #             in_pattern = True
            #     if not in_pattern:
            #         wrong_patterns += 1
            #
            # for j in range(len(y[i])):
            #     if (predictions[i][j] < threshold and y[i][j] == 1.0) or (predictions[i][j] >= threshold and y[i][j] == 0.0):
            #         correct_prediction = False
            #
            # if not in_pattern and not correct_prediction:
            #     wrong_pattern_prediction += 1
            # if (np.around(predictions[i]) == y[i]).all():
            if correct_prediction:
                num_of_correct_pred += 1
            else:
                if self._input_type == 'l':
                    decoded_x = self.decode_x(x[i], dictionary)
                else:
                    decoded_x = self.decode_syllable_x(x[i], syllable_dictionary)
                if self._bidirectional_basic_input:
                    decoded_x = decoded_x[:int(len(decoded_x)/2)]
                errors.append([i,
                               decoded_x,
                               self.decode_x_other_features(feature_dictionary, [x_other_features[i]]),
                               self.assign_stress_locations(decoded_x, round_predictions, vowels, syllables=self._input_type != 'l'),
                               self.assign_stress_locations(decoded_x, y[i], vowels, syllables=self._input_type != 'l')
                               ])

        # print(wrong_patterns)
        # print(wrong_pattern_prediction)
        return (num_of_correct_pred / float(num_of_pred)) * 100, errors

    # def get_word_length(self, x_el):
    #     i = 0
    #     for el in x_el:
    #         if el == 0:
    #             return i
    #         i += 1
    #     return 10

    @staticmethod
    def decode_syllable_x(word_encoded, syllable_dictionary):
        word = []
        for i in range(len(word_encoded)):
            word.append(syllable_dictionary[word_encoded[i]])
        return ''.join(word[::-1])

    def assign_stress_locations(self, word, y, vowels, syllables=False):
        if not syllables:
            word_list = list(word)
        else:
            if self._reverse_inputs:
                word_list = list(word)[::-1]
            else:
                word_list = list(word)
        vowel_num = 0
        for i in range(len(word_list)):
            if self._is_vowel(word_list, i, vowels):
                if word_list[i] == 'a' and y[vowel_num] == 1:
                    word_list[i] = 'á'
                elif word_list[i] == 'e' and y[vowel_num] == 1:
                    word_list[i] = 'é'
                elif word_list[i] == 'i' and y[vowel_num] == 1:
                    word_list[i] = 'í'
                elif word_list[i] == 'o' and y[vowel_num] == 1:
                    word_list[i] = 'ó'
                elif word_list[i] == 'u' and y[vowel_num] == 1:
                    word_list[i] = 'ú'
                elif word_list[i] == 'r' and y[vowel_num] == 1:
                    word_list[i] = 'ŕ'
                elif word_list[i] == 'A' and y[vowel_num] == 1:
                    word_list[i] = 'Á'
                elif word_list[i] == 'E' and y[vowel_num] == 1:
                    word_list[i] = 'É'
                elif word_list[i] == 'I' and y[vowel_num] == 1:
                    word_list[i] = 'Í'
                elif word_list[i] == 'O' and y[vowel_num] == 1:
                    word_list[i] = 'Ó'
                elif word_list[i] == 'U' and y[vowel_num] == 1:
                    word_list[i] = 'Ú'
                elif word_list[i] == 'R' and y[vowel_num] == 1:
                    word_list[i] = 'Ŕ'
                vowel_num += 1
        if not syllables:
            return ''.join(word_list)
        else:
            return ''.join(word_list[::-1])

    def test_type_accuracy(self, predictions, x, x_other_features, y, dictionary, feature_dictionary, vowels, accented_vowels,
                      syllable_dictionary=None):
        errors = []
        num_of_pred = len(predictions)
        num_of_correct_pred = 0
        num_of_correct_pred_words = 0
        accentuation_index = 0
        eye = np.eye(len(accented_vowels), dtype=int)
        for i in range(len(y)):
            correct_prediction = True
            if self._input_type == 'l':
                decoded_x = self.decode_x(x[i], dictionary)
            else:
                decoded_x = self.decode_syllable_x(x[i], syllable_dictionary)
            wrong_word = decoded_x
            correct_word = decoded_x

            for j in range(len(y[i])):
                if y[i][j] > 0:
                    # ERROR AS IT IS CALCULATED
                    # arounded_predictions = np.around(predictions[accentuation_index]).astype(int)

                    # MAX ELEMENT ONLY
                    # arounded_predictions = np.zeros(len(predictions[accentuation_index]))
                    # arounded_predictions[np.argmax(predictions[accentuation_index]).astype(int)] = 1

                    # MAX ELEMENT AMONGT POSSIBLE ONES
                    # if i == 313:
                    #    print(decoded_x)
                    stressed_letter = self.get_accentuated_letter(decoded_x, j, vowels, syllables=self._input_type != 'l')
                    possible_places = np.zeros(len(predictions[accentuation_index]))
                    if stressed_letter == 'r':
                        possible_places[0] = 1
                    elif stressed_letter == 'a':
                        possible_places[1] = 1
                        possible_places[2] = 1
                    elif stressed_letter == 'e':
                        possible_places[3] = 1
                        possible_places[4] = 1
                        possible_places[5] = 1
                    elif stressed_letter == 'i':
                        possible_places[6] = 1
                        possible_places[7] = 1
                    elif stressed_letter == 'o':
                        possible_places[8] = 1
                        possible_places[9] = 1
                        possible_places[10] = 1
                    elif stressed_letter == 'u':
                        possible_places[11] = 1
                        possible_places[12] = 1
                    possible_predictions = predictions[accentuation_index] * possible_places

                    arounded_predictions = np.zeros(len(predictions[accentuation_index]), dtype=int)
                    arounded_predictions[np.argmax(possible_predictions).astype(int)] = 1

                    wrong_word = self.assign_word_accentuation_type(wrong_word, j, arounded_predictions, vowels, accented_vowels,
                                                               syllables=self._input_type != 'l', debug=i == 313)
                    correct_word = self.assign_word_accentuation_type(correct_word, j, eye[int(y[i][j])], vowels, accented_vowels,
                                                                 syllables=self._input_type != 'l', debug=i == 313)

                    if (eye[int(y[i][j])] == arounded_predictions).all():
                        num_of_correct_pred += 1
                    else:
                        correct_prediction = False

                    accentuation_index += 1

            if correct_prediction:
                num_of_correct_pred_words += 1
            else:
                if self._input_type == 'l':
                    errors.append([i,
                                   decoded_x[::-1],
                                   self.decode_x_other_features(feature_dictionary, [x_other_features[i]]),
                                   wrong_word[::-1],
                                   correct_word[::-1]
                                   ])
                else:
                    errors.append([i,
                                   decoded_x,
                                   self.decode_x_other_features(feature_dictionary, [x_other_features[i]]),
                                   wrong_word,
                                   correct_word
                                   ])
        print(num_of_pred)
        print(len(y))
        print(num_of_correct_pred_words)
        print(len(errors))
        print(num_of_correct_pred_words + len(errors))
        return (num_of_correct_pred / float(num_of_pred)) * 100, (num_of_correct_pred_words / float(len(y))) * 100, errors

    def get_accentuated_letter(self, word, location, vowels, syllables=False, debug=False):
        # print(location)
        vowel_index = 0
        word_list = list(word)
        if not syllables:
            word_list = list(word)
        else:
            word_list = list(word[::-1])
        for i in range(len(word_list)):
            if self._is_vowel(word_list, i, vowels):
                if location == vowel_index:
                    return word_list[i]
                vowel_index += 1

    def assign_word_accentuation_type(self, word, location, y, vowels, accented_vowels, syllables=False, debug=False):
        vowel_index = 0
        if not syllables:
            word_list = list(word)
        else:
            word_list = list(word[::-1])
        for i in range(len(word_list)):
            if self._is_vowel(word_list, i, vowels):
                if location == vowel_index:
                    if len(np.where(y == 1)[0]) == 1:
                        word_list[i] = accented_vowels[np.where(y == 1)[0][0]]
                vowel_index += 1
        if not syllables:
            return ''.join(word_list)
        else:
            return ''.join(word_list[::-1])

    def assign_stress_types(self, predictions, word, y, vowels, accented_vowels):
        words = []
        accentuation_index = 0
        for i in range(len(y)):
            wrong_word = word[i][::-1]

            for j in range(len(y[i])):
                if y[i][j] > 0:
                    stressed_letter = self.get_accentuated_letter(word[i][::-1], j, vowels, syllables=self._input_type != 'l')
                    possible_places = np.zeros(len(predictions[accentuation_index]))
                    if stressed_letter == 'r':
                        possible_places[0] = 1
                    elif stressed_letter == 'a':
                        possible_places[1] = 1
                        possible_places[2] = 1
                    elif stressed_letter == 'e':
                        possible_places[3] = 1
                        possible_places[4] = 1
                        possible_places[5] = 1
                    elif stressed_letter == 'i':
                        possible_places[6] = 1
                        possible_places[7] = 1
                    elif stressed_letter == 'o':
                        possible_places[8] = 1
                        possible_places[9] = 1
                        possible_places[10] = 1
                    elif stressed_letter == 'u':
                        possible_places[11] = 1
                        possible_places[12] = 1
                    possible_predictions = predictions[accentuation_index] * possible_places

                    arounded_predictions = np.zeros(len(predictions[accentuation_index]), dtype=int)

                    arounded_predictions[np.argmax(possible_predictions).astype(int)] = 1

                    if np.max(possible_predictions) != 0:
                        wrong_word = self.assign_word_accentuation_type(wrong_word, j, arounded_predictions, vowels, accented_vowels,
                                                                    syllables=self._input_type != 'l', debug=i == 313)

                    accentuation_index += 1

            words.append(wrong_word[::-1])
        return words

    @staticmethod
    def load_location_models(letters_path, syllables_path, syllabled_letters_path):
        ############################ LOCATION ########################
        nn_output_dim = 10

        conv_input_shape = (23, 36)
        othr_input = (140,)

        conv_input = Input(shape=conv_input_shape, name='conv_input')
        x_conv = Conv1D(115, (3), padding='same', activation='relu')(conv_input)
        x_conv = Conv1D(46, (3), padding='same', activation='relu')(x_conv)
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
        x = Dropout(0.3)(x)
        x = Dense(nn_output_dim, activation='sigmoid')(x)

        letter_location_model = Model(inputs=[conv_input, othr_input], outputs=x)
        opt = optimizers.Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        letter_location_model.compile(loss='binary_crossentropy', optimizer=opt, metrics=[actual_accuracy, ])


        letter_location_model.load_weights(letters_path)

        ##############################################################
        # num_examples = len(data.x_train)  # training set size
        nn_output_dim = 10

        conv_input_shape = (10, 5168)
        othr_input = (140,)
        conv_input = Input(shape=conv_input_shape, name='conv_input')

        # syllabled letters
        x_conv = Conv1D(200, (2), padding='same', activation='relu')(conv_input)
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

        syllable_location_model = Model(inputs=[conv_input, othr_input], outputs=x)
        opt = optimizers.Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        syllable_location_model.compile(loss='binary_crossentropy', optimizer=opt, metrics=[actual_accuracy, ])
        syllable_location_model.load_weights(syllables_path)


        #####################################################
        conv_input_shape = (10, 252)

        othr_input = (140,)

        conv_input = Input(shape=conv_input_shape, name='conv_input')

        # syllabled letters
        x_conv = Conv1D(200, (2), padding='same', activation='relu')(conv_input)
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

        syllabled_letters_location_model = Model(inputs=[conv_input, othr_input], outputs=x)
        opt = optimizers.Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        syllabled_letters_location_model.compile(loss='binary_crossentropy', optimizer=opt, metrics=[actual_accuracy, ])
        syllabled_letters_location_model.load_weights(syllabled_letters_path)

        return letter_location_model, syllable_location_model, syllabled_letters_location_model

    @staticmethod
    def load_type_models(letters_path, syllables_path, syllabled_letters_path):
        nn_output_dim = 13

        # letters
        conv_input_shape = (23, 36)
        othr_input = (150,)
        conv_input = Input(shape=conv_input_shape, name='conv_input')
        # letters
        x_conv = Conv1D(115, (3), padding='same', activation='relu')(conv_input)
        x_conv = Conv1D(46, (3), padding='same', activation='relu')(x_conv)

        # syllabled letters
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

        letter_type_model = Model(inputs=[conv_input, othr_input], outputs=x)
        opt = optimizers.Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        letter_type_model.compile(loss='binary_crossentropy', optimizer=opt, metrics=[actual_accuracy, ])
        letter_type_model.load_weights(letters_path)

        conv_input_shape = (10, 5168)
        othr_input = (150,)
        conv_input = Input(shape=conv_input_shape, name='conv_input')

        x_conv = Conv1D(200, (2), padding='same', activation='relu')(conv_input)
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

        syllable_type_model = Model(inputs=[conv_input, othr_input], outputs=x)
        opt = optimizers.Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        syllable_type_model.compile(loss='binary_crossentropy', optimizer=opt, metrics=[actual_accuracy, ])
        syllable_type_model.load_weights(syllables_path)

        # syllabled letters
        conv_input_shape = (10, 252)
        othr_input = (150,)
        conv_input = Input(shape=conv_input_shape, name='conv_input')

        x_conv = Conv1D(200, (2), padding='same', activation='relu')(conv_input)
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

        syllabled_letter_type_model = Model(inputs=[conv_input, othr_input], outputs=x)
        opt = optimizers.Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        syllabled_letter_type_model.compile(loss='binary_crossentropy', optimizer=opt, metrics=[actual_accuracy, ])
        syllabled_letter_type_model.load_weights(syllabled_letters_path)

        return letter_type_model, syllable_type_model, syllabled_letter_type_model

    @staticmethod
    def get_ensemble_location_predictions(input_words, letter_location_model, syllable_location_model, syllabled_letters_location_model,
                                          letter_location_co_model, syllable_location_co_model, syllabled_letters_location_co_model,
                                          dictionary, max_word, max_num_vowels, vowels, accented_vowels, feature_dictionary, syllable_dictionary):
        batch_size = 16
        # print(tagged_input_words[pos])

        data = Data('l', shuffle_all_inputs=False, convert_multext=False)
        x, x_other_features, fake_y = data._generate_x_and_y(dictionary, max_word, max_num_vowels, input_words, vowels, accented_vowels,
                                                             feature_dictionary, 'who cares')
        generator = data._letter_generator(x, x_other_features, fake_y, batch_size, accented_vowels)
        letter_location_predictions = letter_location_model.predict_generator(generator, len(x) / (batch_size))

        data = Data('s', shuffle_all_inputs=False, convert_multext=False)
        x, x_other_features, fake_y = data._generate_x_and_y(syllable_dictionary, max_word, max_num_vowels, input_words, vowels,
                                                             accented_vowels, feature_dictionary, 'who cares')
        eye = np.eye(len(syllable_dictionary), dtype=int)
        generator = data._syllable_generator(x, x_other_features, fake_y, batch_size, eye, accented_vowels)
        syllable_location_predictions = syllable_location_model.predict_generator(generator, len(x) / (batch_size))

        data = Data('sl', shuffle_all_inputs=False, convert_multext=False)
        x, x_other_features, fake_y = data._generate_x_and_y(syllable_dictionary, max_word, max_num_vowels, input_words, vowels,
                                                             accented_vowels, feature_dictionary, 'who cares')
        max_syllable = data._get_max_syllable(syllable_dictionary)
        syllable_letters_translator = data._create_syllable_letters_translator(max_syllable, syllable_dictionary, dictionary, vowels)
        generator = data._syllable_generator(x, x_other_features, fake_y, batch_size, syllable_letters_translator, accented_vowels)
        syllabled_letters_location_predictions = syllabled_letters_location_model.predict_generator(generator, len(x) / (batch_size))

        ############## CORRECT ORDER INPUT ##############
        data = Data('l', shuffle_all_inputs=False, convert_multext=False, reverse_inputs=False)
        x, x_other_features, fake_y = data._generate_x_and_y(dictionary, max_word, max_num_vowels, input_words, vowels, accented_vowels,
                                                             feature_dictionary, 'who cares')
        generator = data._letter_generator(x, x_other_features, fake_y, batch_size, accented_vowels)
        letter_location_co_predictions = letter_location_co_model.predict_generator(generator, len(x) / (batch_size))

        letter_location_co_predictions = data.reverse_predictions(letter_location_co_predictions, input_words, vowels)

        data = Data('s', shuffle_all_inputs=False, convert_multext=False, reverse_inputs=False)
        x, x_other_features, fake_y = data._generate_x_and_y(syllable_dictionary, max_word, max_num_vowels, input_words, vowels,
                                                                 accented_vowels, feature_dictionary, 'who cares')
        eye = np.eye(len(syllable_dictionary), dtype=int)
        generator = data._syllable_generator(x, x_other_features, fake_y, batch_size, eye, accented_vowels)
        syllable_location_co_predictions = syllable_location_co_model.predict_generator(generator, len(x) / (batch_size))

        syllable_location_co_predictions = data.reverse_predictions(syllable_location_co_predictions, input_words, vowels)

        data = Data('sl', shuffle_all_inputs=False, convert_multext=False, reverse_inputs=False)
        x, x_other_features, fake_y = data._generate_x_and_y(syllable_dictionary, max_word, max_num_vowels, input_words, vowels,
                                                             accented_vowels, feature_dictionary, 'who cares')
        max_syllable = data._get_max_syllable(syllable_dictionary)
        syllable_letters_translator = data._create_syllable_letters_translator(max_syllable, syllable_dictionary, dictionary, vowels)
        generator = data._syllable_generator(x, x_other_features, fake_y, batch_size, syllable_letters_translator, accented_vowels)
        syllabled_letters_location_co_predictions = syllabled_letters_location_co_model.predict_generator(generator, len(x) / (batch_size))

        syllabled_letters_location_co_predictions = data.reverse_predictions(syllabled_letters_location_co_predictions, input_words, vowels)

        return np.mean(np.array([letter_location_predictions, syllable_location_predictions, syllabled_letters_location_predictions,
                                 letter_location_co_predictions, syllable_location_co_predictions, syllabled_letters_location_co_predictions]), axis=0)

    def count_syllables(self, word, vowels):
        j = 0
        num_vowels = 0
        for j in range(len(word)):
            if self._is_vowel(word, j, vowels):
                num_vowels += 1
        return num_vowels

    def reverse_predictions(self, predictions, words, vowels):
        new_predictions = np.zeros(predictions.shape, dtype='float32')
        for i in range(len(predictions)):
            word_len = self.count_syllables(words[i][0], vowels)

            for k in range(word_len):
                new_predictions[i][k] += predictions[i][word_len - 1 - k]

        return new_predictions

    @staticmethod
    def get_ensemble_type_predictions(input_words, location_y, letter_type_model, syllable_type_model, syllabled_letter_type_model,
                                      letter_type_co_model, syllable_type_co_model, syllabled_letter_type_co_model,
                                      dictionary, max_word, max_num_vowels, vowels, accented_vowels, feature_dictionary, syllable_dictionary):
        batch_size = 16
        y_array = np.asarray(location_y)
        accentuation_length = (y_array > 0).sum()

        data = Data('l', shuffle_all_inputs=False, accent_classification=True, convert_multext=False)
        x, x_other_features, fake_y = data._generate_x_and_y(dictionary, max_word, max_num_vowels, input_words, vowels, accented_vowels,
                                                             feature_dictionary, 'who cares')
        generator = data._letter_generator(x, x_other_features, location_y, batch_size, accented_vowels)
        letter_type_predictions = letter_type_model.predict_generator(generator, accentuation_length / (batch_size))

        data = Data('s', shuffle_all_inputs=False, accent_classification=True, convert_multext=False)
        x, x_other_features, fake_y = data._generate_x_and_y(syllable_dictionary, max_word, max_num_vowels, input_words, vowels,
                                                             accented_vowels, feature_dictionary, 'who cares')
        eye = np.eye(len(syllable_dictionary), dtype=int)
        generator = data._syllable_generator(x, x_other_features, location_y, batch_size, eye, accented_vowels)
        syllable_type_predictions = syllable_type_model.predict_generator(generator, accentuation_length / (batch_size))

        data = Data('sl', shuffle_all_inputs=False, accent_classification=True, convert_multext=False)
        x, x_other_features, fake_y = data._generate_x_and_y(syllable_dictionary, max_word, max_num_vowels, input_words, vowels,
                                                             accented_vowels, feature_dictionary, 'who cares')
        max_syllable = data._get_max_syllable(syllable_dictionary)
        syllable_letters_translator = data._create_syllable_letters_translator(max_syllable, syllable_dictionary, dictionary, vowels)
        generator = data._syllable_generator(x, x_other_features, location_y, batch_size, syllable_letters_translator, accented_vowels)
        syllabled_letter_type_predictions = syllabled_letter_type_model.predict_generator(generator, accentuation_length / batch_size)

        ############## CORRECT ORDER INPUT ##############
        location_y = data.reverse_predictions(location_y, input_words, vowels)

        data = Data('l', shuffle_all_inputs=False, accent_classification=True, convert_multext=False, reverse_inputs=False)
        x, x_other_features, fake_y = data._generate_x_and_y(dictionary, max_word, max_num_vowels, input_words, vowels, accented_vowels,
                                                             feature_dictionary, 'who cares')
        generator = data._letter_generator(x, x_other_features, location_y, batch_size, accented_vowels)
        letter_type_co_predictions = letter_type_co_model.predict_generator(generator, accentuation_length / (batch_size))

        data.reorder_correct_direction_inputs(letter_type_co_predictions, location_y)

        data = Data('s', shuffle_all_inputs=False, accent_classification=True, convert_multext=False, reverse_inputs=False)
        x, x_other_features, fake_y = data._generate_x_and_y(syllable_dictionary, max_word, max_num_vowels, input_words, vowels,
                                                             accented_vowels, feature_dictionary, 'who cares')
        eye = np.eye(len(syllable_dictionary), dtype=int)
        generator = data._syllable_generator(x, x_other_features, location_y, batch_size, eye, accented_vowels)
        syllable_type_co_predictions = syllable_type_co_model.predict_generator(generator, accentuation_length / (batch_size))

        data.reorder_correct_direction_inputs(syllable_type_co_predictions, location_y)

        data = Data('sl', shuffle_all_inputs=False, accent_classification=True, convert_multext=False, reverse_inputs=False)
        x, x_other_features, fake_y = data._generate_x_and_y(syllable_dictionary, max_word, max_num_vowels, input_words, vowels,
                                                             accented_vowels, feature_dictionary, 'who cares')
        max_syllable = data._get_max_syllable(syllable_dictionary)
        syllable_letters_translator = data._create_syllable_letters_translator(max_syllable, syllable_dictionary, dictionary, vowels)
        generator = data._syllable_generator(x, x_other_features, location_y, batch_size, syllable_letters_translator, accented_vowels)
        syllabled_letter_type_co_predictions = syllabled_letter_type_co_model.predict_generator(generator, accentuation_length / batch_size)

        data.reorder_correct_direction_inputs(syllabled_letter_type_co_predictions, location_y)

        return np.mean(np.array([letter_type_predictions, syllable_type_predictions, syllabled_letter_type_predictions,
                                 letter_type_co_predictions, syllable_type_co_predictions, syllabled_letter_type_co_predictions]), axis=0)

    def reorder_correct_direction_inputs(self, predictions, y):
        pred_i = 0
        for i in range(len(y)):
            num_accented_syllables = 0
            for el in y[i]:
                if el > 0:
                    num_accented_syllables += 1
            if num_accented_syllables > 1:
                min_i = pred_i
                max_i = pred_i + num_accented_syllables - 1
                while (max_i > min_i):
                    min_pred = copy(predictions[min_i])
                    max_pred = copy(predictions[max_i])
                    predictions[min_i] = max_pred
                    predictions[max_i] = min_pred
                    min_i += 1
                    max_i -= 1
            pred_i += num_accented_syllables

    def assign_location_stress(self, word, locations, vowels):
            #     word = list(word)
        word_list = list(word)
        for loc in locations:
            vowel_num = 0
            # if loc == 0:
            #    return word
            for i in range(len(word_list)):
                if self._is_vowel(word_list, i, vowels):
                    if word_list[i] == 'a' and vowel_num == loc:
                        word_list[i] = 'á'
                    elif word_list[i] == 'e' and vowel_num == loc:
                        word_list[i] = 'é'
                    elif word_list[i] == 'i' and vowel_num == loc:
                        word_list[i] = 'í'
                    elif word_list[i] == 'o' and vowel_num == loc:
                        word_list[i] = 'ó'
                    elif word_list[i] == 'u' and vowel_num == loc:
                        word_list[i] = 'ú'
                    elif word_list[i] == 'r' and vowel_num == loc:
                        word_list[i] = 'ŕ'
                    elif word_list[i] == 'A' and vowel_num == loc:
                        word_list[i] = 'Á'
                    elif word_list[i] == 'E' and vowel_num == loc:
                        word_list[i] = 'É'
                    elif word_list[i] == 'I' and vowel_num == loc:
                        word_list[i] = 'Í'
                    elif word_list[i] == 'O' and vowel_num == loc:
                        word_list[i] = 'Ó'
                    elif word_list[i] == 'U' and vowel_num == loc:
                        word_list[i] = 'Ú'
                    elif word_list[i] == 'R' and vowel_num == loc:
                        word_list[i] = 'Ŕ'
                    vowel_num += 1
                    #     print(word_list)
        return ''.join(word_list)

    def accentuate_word(self, input_words, letter_location_model, syllable_location_model, syllabled_letters_location_model,
                        letter_location_co_model, syllable_location_co_model, syllabled_letters_location_co_model,
                        letter_type_model, syllable_type_model, syllabled_letter_type_model,
                        letter_type_co_model, syllable_type_co_model, syllabled_letter_type_co_model,
                        dictionary, max_word, max_num_vowels, vowels, accented_vowels, feature_dictionary, syllable_dictionary):
        predictions = self.get_ensemble_location_predictions(input_words, letter_location_model, syllable_location_model,
                                                             syllabled_letters_location_model,
                                                             letter_location_co_model, syllable_location_co_model,
                                                             syllabled_letters_location_co_model,
                                                             dictionary, max_word, max_num_vowels, vowels, accented_vowels, feature_dictionary,
                                                             syllable_dictionary)
        #print(predictions)
        if 'A' not in vowels:
            vowels.extend(['A', 'E', 'I', 'O', 'U'])
        location_accented_words = [self.assign_location_stress(input_words[i][0][::-1], self.decode_y(predictions[i]), vowels)[::-1] for i in
                          range(len(input_words))]

        location_y = np.around(predictions)
        type_predictions = self.get_ensemble_type_predictions(input_words, location_y, letter_type_model, syllable_type_model,
                                                              syllabled_letter_type_model,
                                                              letter_type_co_model, syllable_type_co_model, syllabled_letter_type_co_model,
                                                              dictionary, max_word, max_num_vowels, vowels, accented_vowels, feature_dictionary,
                                                              syllable_dictionary)

        only_words = [el[0] for el in input_words]
        accented_words = self.assign_stress_types(type_predictions, only_words, location_y, vowels, accented_vowels)

        return location_accented_words, accented_words

# def count_vowels(content, vowels):
#     num_all_vowels = 0
#     for el in content:
#         for m in range(len(el[0])):
#             if is_vowel(list(el[0]), m, vowels):
#                 num_all_vowels += 1
#     return num_all_vowels



# metric for calculation of correct results
# test with:
# print(mean_pred(y_validate[pos], predictions[pos]).eval())
# print(mean_pred(np.array([[ 0.,  1.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
#                           [ 0.,  1.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]]),
#                 np.array([[ 0.,  0.51,  0.,  0.51,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
#                           [ 0.,  0.92,  0.,  0.51,  0.,  0.,  0.,  0.,  0.,  0.,  0.]])).eval())
def actual_accuracy(y_true, y_pred):
    return K.mean(K.equal(K.mean(K.equal(K.round(y_true), K.round(y_pred)), axis=-1), 1.0))
