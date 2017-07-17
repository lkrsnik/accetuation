# -*- coding: utf-8 -*-
from __future__ import unicode_literals
# text in Western (Windows 1252)

import numpy as np
import h5py
import gc
import math
import keras.backend as K
import os.path


# functions for saving, loading and shuffling whole arrays to ram
def save_inputs(file_name, X, y, other_features=[]):
    h5f = h5py.File(file_name, 'w')
    if other_features == []:
        adict = dict(X=X, y=y)
    else:
        adict = dict(X=X, X_other_features=other_features, y=y)
    for k, v in adict.items():
        h5f.create_dataset(k, data=v)
    h5f.close()


def load_inputs(file_name, other_features=False):
    h5f = h5py.File(file_name,'r')
    X = h5f['X'][:]
    y = h5f['y'][:]
    if other_features:
        X_other_features = h5f['X_other_features'][:]
        h5f.close()
        return X, X_other_features, y

    h5f.close()
    return X, y


def shuffle_inputs(X, y, shuffle_vector_location, X_pure=[]):
    if os.path.exists(shuffle_vector_location):
        s = load_shuffle_vector(shuffle_vector_location)
    else:
        s = np.arange(X.shape[0])
        np.random.shuffle(s)
        create_and_save_shuffle_vector(shuffle_vector_location, s)
    # s = np.arange(X.shape[0])
    # np.random.shuffle(s)
    X = X[s]
    y = y[s]
    if X_pure != []:
        X_pure = X_pure[s]
        return X, y, X_pure
    else:
        return X, y


# functions for saving and loading partial arrays to ram
def create_and_save_inputs(file_name, part, X, y, X_pure):
    # X, y, X_pure = generate_full_vowel_matrix_inputs()
    h5f = h5py.File(file_name + part + '.h5', 'w')
    adict=dict(X=X, y=y, X_pure=X_pure)
    for k, v in adict.items():
        h5f.create_dataset(k,data=v)
    h5f.close()


def load_extended_inputs(file_name, obtain_range):
    h5f = h5py.File(file_name, 'r')
    X = h5f['X'][obtain_range[0]:obtain_range[1]]
    y = h5f['y'][obtain_range[0]:obtain_range[1]]
    X_pure = h5f['X_pure'][obtain_range[0]:obtain_range[1]]

    h5f.close()
    return X, y, X_pure


# functions for creating and loading shuffle vector
def create_and_save_shuffle_vector(file_name, shuffle_vector):
    # X, y, X_pure = generate_full_vowel_matrix_inputs()
    h5f = h5py.File(file_name, 'w')
    adict = dict(shuffle_vector=shuffle_vector)
    for k, v in adict.items():
        h5f.create_dataset(k, data=v)
    h5f.close()


def load_shuffle_vector(file_name):
    h5f = h5py.File(file_name, 'r')
    # shuffle_vector = h5f['shuffle_vector'][[179859, 385513, 893430]]
    shuffle_vector = h5f['shuffle_vector'][:]

    h5f.close()
    return shuffle_vector


# functions for saving and loading model - ONLY WHERE KERAS IS NOT NEEDED
# def save_model(model, file_name):
#     h5f = h5py.File(file_name, 'w')
#     adict = dict(W1=model['W1'], b1=model['b1'], W2=model['W2'], b2=model['b2'])
#     for k,v in adict.items():
#         h5f.create_dataset(k,data=v)
#
#     h5f.close()
#
#
# def load_model(file_name):
#     h5f = h5py.File(file_name,'r')
#     model = {}
#     W1.set_value(h5f['W1'][:])
#     b1.set_value(h5f['b1'][:])
#     W2.set_value(h5f['W2'][:])
#     b2.set_value(h5f['b2'][:])
#     h5f.close()
#     return model

# functions for creating X and y from content
def read_content():
    print('READING CONTENT...')
    with open('../../../data/SlovarIJS_BESEDE_utf8.lex') as f:
        content = f.readlines()
    print('CONTENT READ SUCCESSFULY')
    return [x.split('\t') for x in content]


def is_vowel(word_list, position, vowels):
    if word_list[position] in vowels:
        return True
    if word_list[position] == u'r' and     (position - 1 < 0 or word_list[position - 1] not in vowels) and     (position + 1 >= len(word_list) or word_list[position + 1] not in vowels):
        return True
    return False


def is_accetuated_vowel(word_list, position, accetuated_vowels):
    if word_list[position] in accetuated_vowels:
        return True
    return False


def create_dict():
    content = read_content()
    print('CREATING DICTIONARY...')

    # CREATE dictionary AND max_word
    accetuated_vowels = [u'à', u'á', u'ä', u'é', u'ë', u'ì', u'í', u'î', u'ó', u'ô', u'ö', u'ú', u'ü']
    default_vowels = [u'a', u'e', u'i', u'o', u'u']
    vowels = []
    vowels.extend(accetuated_vowels)
    vowels.extend(default_vowels)

    dictionary_output = ['']
    dictionary_input = ['']
    line = 0
    max_word = 0
    # ADD 'EMPTY' VOWEL
    max_num_vowels = 0
    for el in content:
        num_vowels = 0
        i = 0
        try: 
            if len(el[3]) > max_word:
                max_word = len(el[3])
            if len(el[0]) > max_word:
                max_word = len(el[0])
            for c in list(el[3]):
                if is_vowel(list(el[3]), i, vowels):
                    num_vowels += 1
                if c not in dictionary_output:
                    dictionary_output.append(c)
                i += 1
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
    max_num_vowels += 1
    print('DICTIONARY CREATION SUCCESSFUL!')
    return dictionary_input, max_word, max_num_vowels, content, vowels, accetuated_vowels


# GENERATE X and y
def generate_presentable_y(accetuations_list, word_list, max_num_vowels):
    while len(accetuations_list) < 2:
        accetuations_list.append(0)
    if len(accetuations_list) > 2:
        accetuations_list = accetuations_list[:2]
    accetuations_list = np.array(accetuations_list)
    final_position = accetuations_list[0] + max_num_vowels * accetuations_list[1]
    return final_position


# def generate_inputs():
#     dictionary, max_word, max_num_vowels, content, vowels, accetuated_vowels = create_dict()
#
#     print('GENERATING X AND y...')
#     X = np.zeros((len(content), max_word*len(dictionary)))
#     y = np.zeros((len(content), max_num_vowels * max_num_vowels ))
#
#     i = 0
#     for el in content:
#         j = 0
#         for c in list(el[0]):
#             index = 0
#             for d in dictionary:
#                 if c == d:
#                     X[i][index + j * max_word] = 1
#                     break
#                 index += 1
#             j += 1
#         j = 0
#         word_accetuations = []
#         num_vowels = 0
#         for c in list(el[3]):
#             index = 0
#             if is_vowel(el[3], j, vowels):
#                 num_vowels += 1
#             for d in accetuated_vowels:
#                 if c == d:
#                     word_accetuations.append(num_vowels)
#                     break
#                 index += 1
#             j += 1
#         y[i][generate_presentable_y(word_accetuations, list(el[3]), max_num_vowels)] = 1
#         i += 1
#     print('GENERATION SUCCESSFUL!')
#     print('SHUFFELING INPUTS...')
#     X, y = shuffle_inputs(X, y)
#     print('INPUTS SHUFFELED!')
#     return X, y
#
#
# def generate_matrix_inputs():
#     dictionary, max_word, max_num_vowels, content, vowels, accetuated_vowels = create_dict()
#
#     print('GENERATING X AND y...')
#     # X = np.zeros((len(content), max_word*len(dictionary)))
#     y = np.zeros((len(content), max_num_vowels * max_num_vowels ))
#
#     X = []
#
#     i = 0
#     for el in content:
#         # j = 0
#         word = []
#         for c in list(el[0]):
#             index = 0
#             character = np.zeros(len(dictionary))
#             for d in dictionary:
#                 if c == d:
#                     # X[i][index + j * max_word] = 1
#                     character[index] = 1
#                     break
#                 index += 1
#             word.append(character)
#             # j += 1
#         j = 0
#         X.append(word)
#         word_accetuations = []
#         num_vowels = 0
#         for c in list(el[3]):
#             index = 0
#             if is_vowel(el[3], j, vowels):
#                 num_vowels += 1
#             for d in accetuated_vowels:
#                 if c == d:
#                     word_accetuations.append(num_vowels)
#                     break
#                 index += 1
#             j += 1
#         y[i][generate_presentable_y(word_accetuations, list(el[3]), max_num_vowels)] = 1
#         i += 1
#     X = np.array(X)
#     print('GENERATION SUCCESSFUL!')
#     print('SHUFFELING INPUTS...')
#     X, y = shuffle_inputs(X, y)
#     print('INPUTS SHUFFELED!')
#     return X, y


def generate_full_matrix_inputs(content_shuffle_vector_location, shuffle_vector_location):
    dictionary, max_word, max_num_vowels, content, vowels, accetuated_vowels = create_dict()
    train_content, test_content, validate_content = split_content(content, 0.2, content_shuffle_vector_location)
    feature_dictionary = create_feature_dictionary()

    # Generate X and y
    print('GENERATING X AND y...')
    X_train, X_other_features_train, y_train = generate_X_and_y(dictionary, max_word, max_num_vowels, train_content, vowels, accetuated_vowels, feature_dictionary, shuffle_vector_location + '_train.h5')
    X_test, X_other_features_test, y_test = generate_X_and_y(dictionary, max_word, max_num_vowels, test_content, vowels, accetuated_vowels, feature_dictionary, shuffle_vector_location + '_test.h5')
    X_validate, X_other_features_validate, y_validate = generate_X_and_y(dictionary, max_word, max_num_vowels, validate_content, vowels, accetuated_vowels, feature_dictionary, shuffle_vector_location + '_validate.h5')
    print('GENERATION SUCCESSFUL!')
    return X_train, X_other_features_train, y_train, X_test, X_other_features_test, y_test, X_validate, X_other_features_validate, y_validate


# generate full matrix, with old features
def old_generate_full_matrix_inputs():
    dictionary, max_word, max_num_vowels, content, vowels, accetuated_vowels = create_dict()
    train_content, validate_content = split_content(content, 0.2)
    feature_dictionary = create_feature_dictionary(content)

    # Generate X and y
    print('GENERATING X AND y...')
    X_train, X_other_features_train, y_train = generate_X_and_y(dictionary, max_word, max_num_vowels, train_content, vowels, accetuated_vowels, feature_dictionary)
    X_validate, X_other_features_validate, y_validate = generate_X_and_y(dictionary, max_word, max_num_vowels, validate_content, vowels, accetuated_vowels, feature_dictionary)
    print('GENERATION SUCCESSFUL!')
    return X_train, X_other_features_train, y_train, X_validate, X_other_features_validate, y_validate


# Generate each y as an array of 11 numbers (with possible values between 0 and 1)
def generate_X_and_y(dictionary, max_word, max_num_vowels, content, vowels, accetuated_vowels, feature_dictionary, shuffle_vector_location):
    y = np.zeros((len(content), max_num_vowels))
    X = np.zeros((len(content), max_word, len(dictionary)))
    print('CREATING OTHER FEATURES...')
    X_other_features = create_X_features(content, feature_dictionary)
    print('OTHER FEATURES CREATED!')

    i = 0
    for el in content:
        j = 0
        for c in list(el[0]):
            index = 0
            for d in dictionary:
                if c == d:
                    X[i][j][index] = 1
                    break
                index += 1
            j += 1
        j = 0
        word_accetuations = []
        num_vowels = 0
        for c in list(el[3]):
            index = 0
            if is_vowel(el[3], j, vowels):
                num_vowels += 1
            for d in accetuated_vowels:
                if c == d:
                    word_accetuations.append(num_vowels)
                    break
                index += 1
            j += 1
        if len(word_accetuations) > 0:
            y_value = 1/len(word_accetuations)
            for el in word_accetuations:
                # y[i][el] = y_value
                y[i][el] = 1
        else:
            y[i][0] = 1
        # y[i][generate_presentable_y(word_accetuations, list(el[3]), max_num_vowels)] = 1
        i += 1

    print('SHUFFELING INPUTS...')
    X, y, X_other_features = shuffle_inputs(X, y, shuffle_vector_location, X_pure=X_other_features)
    print('INPUTS SHUFFELED!')
    return X, X_other_features, y


# Generate each y as an array of 121 numbers (with one 1 per line and the rest zeros)
def generate_X_and_y_one_classification(dictionary, max_word, max_num_vowels, content, vowels, accetuated_vowels, feature_dictionary):
    y = np.zeros((len(content), max_num_vowels * max_num_vowels ))
    X = np.zeros((len(content), max_word, len(dictionary)))
    print('CREATING OTHER FEATURES...')
    X_other_features = create_X_features(content, feature_dictionary)
    print('OTHER FEATURES CREATED!')

    i = 0
    for el in content:
        j = 0
        for c in list(el[0]):
            index = 0
            for d in dictionary:
                if c == d:
                    X[i][j][index] = 1
                    break
                index += 1
            j += 1
        j = 0
        word_accetuations = []
        num_vowels = 0
        for c in list(el[3]):
            index = 0
            if is_vowel(el[3], j, vowels):
                num_vowels += 1
            for d in accetuated_vowels:
                if c == d:
                    word_accetuations.append(num_vowels)
                    break
                index += 1
            j += 1
        y[i][generate_presentable_y(word_accetuations, list(el[3]), max_num_vowels)] = 1
        i += 1

    print('SHUFFELING INPUTS...')
    X, y, X_other_features = shuffle_inputs(X, y, X_pure=X_other_features)
    print('INPUTS SHUFFELED!')
    return X, X_other_features, y


def count_vowels(content, vowels):
    num_all_vowels = 0
    for el in content:
        for m in range(len(el[0])):
            if is_vowel(list(el[0]), m, vowels):
                num_all_vowels += 1
    return num_all_vowels


# Data generation for generator inputs
def generate_X_and_y_RAM_efficient(name, split_number):
    h5f = h5py.File(name + '.h5', 'w')
    dictionary, max_word, max_num_vowels, content, vowels, accetuated_vowels = create_dict()
    num_all_vowels = count_vowels(content, vowels)
    data_X = h5f.create_dataset('X', (num_all_vowels, max_word, len(dictionary)),
                                maxshape=(num_all_vowels, max_word, len(dictionary)),
                                dtype=np.uint8)
    data_y = h5f.create_dataset('y', (num_all_vowels,),
                                maxshape=(num_all_vowels,),
                                dtype=np.uint8)
    data_X_pure = h5f.create_dataset('X_pure', (num_all_vowels,),
                                     maxshape=(num_all_vowels,),
                                     dtype=np.uint8)

    gc.collect()
    print('GENERATING X AND y...')
    X_pure = []
    X = []
    y = []
    part_len = len(content)/float(split_number)
    current_part_generation = 1

    i = 0
    num_all_vowels = 0
    old_num_all_vowels = 0
    for el in content:
        j = 0
        X_el = np.zeros((max_word, len(dictionary)))
        for c in list(el[0]):
            index = 0
            for d in dictionary:
                if c == d:
                    X_el[j][index] = 1
                    break
                index += 1
            j += 1
        vowel_i = 0
        for m in range(len(el[0])):
            if is_vowel(list(el[0]), m, vowels):
                X.append(X_el)
                X_pure.append(vowel_i)
                vowel_i += 1
                if is_accetuated_vowel(list(el[3]), m, accetuated_vowels):
                    y.append(1)
                else:
                    y.append(0)

                if current_part_generation * part_len <= i:
                    print('Saving part '+ str(current_part_generation))
                    data_X[old_num_all_vowels:num_all_vowels + 1] = np.array(X)
                    data_y[old_num_all_vowels:num_all_vowels + 1] = np.array(y)
                    data_X_pure[old_num_all_vowels:num_all_vowels + 1] = np.array(X_pure)


                    old_num_all_vowels = num_all_vowels + 1


                    X_pure = []
                    X = []
                    y = []
                    current_part_generation += 1
                num_all_vowels += 1
        if i%10000 == 0:
            print(i)
        i += 1

    print('Saving part ' + str(current_part_generation))

    data_X[old_num_all_vowels:num_all_vowels] = np.array(X)
    data_y[old_num_all_vowels:num_all_vowels] = np.array(y)
    data_X_pure[old_num_all_vowels:num_all_vowels] = np.array(X_pure)

    h5f.close()


# metric for calculation of correct results
def actual_accuracy(y_true, y_pred):
    return K.mean(K.equal(K.mean(K.equal(K.round(y_true), K.round(y_pred)), axis=-1), 1.0))


# generator for inputs for tracking of data fitting
def generate_fake_epoch(orig_X, orig_X_additional, orig_y, batch_size):
    size = orig_X.shape[0]
    while 1:
        loc = 0
        while loc < size:
            if loc + batch_size >= size:
                yield([orig_X[loc:size], orig_X_additional[loc:size]], orig_y[loc:size])
            else:
                yield([orig_X[loc:loc + batch_size], orig_X_additional[loc:loc + batch_size]], orig_y[loc:loc + batch_size])
            loc += batch_size


# generator for inputs
def generate_arrays_from_file(path, batch_size):
    h5f = h5py.File(path, 'r')

    X = h5f['X'][:]
    y = h5f['y'][:]
    X_pure = h5f['X_pure'][:]
    yield (X, y, X_pure)
    # while 1:
    #     f = open(path)
    #     for line in f:
    #         # create Numpy arrays of input data
    #         # and labels, from each line in the file
    #         x, y = process_line(line)
    #         yield (x, y)
    #         # f.close()

    h5f.close()


# shuffle inputs for generator
def shuffle_full_vowel_inputs(name, orderd_name, parts):
    dictionary, max_word, max_num_vowels, content, vowels, accetuated_vowels = create_dict()
    num_all_vowels = count_vowels(content, vowels)
    # num_all_vowels = 12


    s = np.arange(num_all_vowels)
    np.random.shuffle(s)

    h5f = h5py.File(name, 'w')
    data_X = h5f.create_dataset('X', (num_all_vowels, max_word, len(dictionary)),
                                maxshape=(num_all_vowels, max_word, len(dictionary)),
                                dtype=np.uint8)
    data_y = h5f.create_dataset('y', (num_all_vowels,),
                                maxshape=(num_all_vowels,),
                                dtype=np.uint8)
    data_X_pure = h5f.create_dataset('X_pure', (num_all_vowels,),
                                     maxshape=(num_all_vowels,),
                                     dtype=np.uint8)


    gc.collect()

    print('Shuffled vector loaded!')
    section_range = [0, (num_all_vowels + 1)/parts]
    for h in range(1, parts+1):
        gc.collect()
        new_X = np.zeros((section_range[1] - section_range[0], max_word, len(dictionary)))
        new_X_pure = np.zeros(section_range[1] - section_range[0])
        new_y = np.zeros(section_range[1] - section_range[0])
        targeted_range = [0, (num_all_vowels + 1)/parts]
        for i in range(1, parts+1):
            X, y, X_pure = load_extended_inputs(orderd_name, targeted_range)
            for j in range(X.shape[0]):
                if s[j + targeted_range[0]] >= section_range[0] and s[j + targeted_range[0]] < section_range[1]:
                    # print 's[j] ' + str(s[j + targeted_range[0]]) + ' section_range[0] ' + str(section_range[0]) + ' section_range[1] ' + str(section_range[1])
                    new_X[s[j + targeted_range[0]] - section_range[0]] = X[j]
                    new_y[s[j + targeted_range[0]] - section_range[0]] = y[j]
                    new_X_pure[s[j + targeted_range[0]] - section_range[0]] = X_pure[j]
            targeted_range[0] = targeted_range[1]
            if targeted_range[1] + (num_all_vowels + 1) / parts < num_all_vowels:
                targeted_range[1] += (num_all_vowels + 1) / parts
            else:
                targeted_range[1] = num_all_vowels
            del X, y, X_pure
        print('CREATED ' + str(h) + '. PART OF SHUFFLED MATRIX')
        data_X[section_range[0]:section_range[1]] = new_X
        data_y[section_range[0]:section_range[1]] = new_y
        data_X_pure[section_range[0]:section_range[1]] = new_X_pure
        section_range[0] = section_range[1]
        if section_range[1] + (num_all_vowels + 1)/parts < num_all_vowels:
            section_range[1] += (num_all_vowels + 1)/parts
        else:
            section_range[1] = num_all_vowels
        del new_X, new_X_pure, new_y

    h5f.close()


# Decoders for inputs and outputs
def decode_X_features(feature_dictionary, X_other_features):
    final_word = []
    for word in X_other_features:
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
        print(u''.join(final_word))
    return u''.join(final_word)


def decode_position(y, max_num_vowels):
    max_el = 0
    i = 0
    pos = -1
    for el in y:
        if el > max_el:
            max_el = el
            pos = i
        i += 1
    return [pos % max_num_vowels, pos / max_num_vowels]


def decode_input(word_encoded, dictionary):
    word = ''
    for el in word_encoded:
        i = 0
        for num in el:
            if num == 1:
                word += dictionary[i]
                break
            i += 1
    return word


def decode_position_from_number(y, max_num_vowels):
    return [y % max_num_vowels, y / max_num_vowels]
    

def generate_input_from_word(word, max_word, dictionary):
    x = np.zeros(max_word*len(dictionary))
    j = 0
    for c in list(word):
        index = 0
        for d in dictionary:
            if c == d:
                x[index + j * max_word] = 1
                break
            index += 1
        j += 1
    return x


def generate_input_per_vowel_from_word(word, max_word, dictionary, vowels):
    X_el = np.zeros((max_word, len(dictionary)))
    j = 0
    for c in list(word):
        index = 0
        for d in dictionary:
            if c == d:
                X_el[j][index] = 1
                break
            index += 1
        j += 1

    X = []
    X_pure = []
    vowel_i = 0
    for i in range(len(word)):
        if is_vowel(list(word), i, vowels):
            X.append(X_el)
            X_pure.append(vowel_i)
            vowel_i += 1
    return np.array(X), np.array(X_pure)


def decode_position_from_vowel_to_final_number(y):
    res = []
    for i in range(len(y)):
        if y[i][0] > 0.5:
            res.append(i + 1)
    return res


# split content so that there is no overfitting
def split_content(content, test_and_validation_ratio, content_shuffle_vector_location, validation_ratio=0.5):
    expanded_content = [el[1] if el[1] != '=' else el[0] for el in content]
    # print(len(content))
    unique_content = sorted(set(expanded_content))

    if os.path.exists(content_shuffle_vector_location):
        s = load_shuffle_vector(content_shuffle_vector_location)
    else:
        s = np.arange(len(unique_content))
        np.random.shuffle(s)
        create_and_save_shuffle_vector(content_shuffle_vector_location, s)

    split_num = math.floor(len(unique_content) * test_and_validation_ratio)
    validation_num = math.floor(split_num * validation_ratio)
    shuffled_unique_train_content = [unique_content[i] for i in range(len(s)) if s[i] >= split_num]
    shuffled_unique_train_content_set = set(shuffled_unique_train_content)

    shuffled_unique_test_content = [unique_content[i] for i in range(len(s)) if split_num > s[i] >= validation_num]
    shuffled_unique_test_content_set = set(shuffled_unique_test_content)

    shuffled_unique_validate_content = [unique_content[i] for i in range(len(s)) if s[i] < validation_num]
    shuffled_unique_validate_content_set = set(shuffled_unique_validate_content)

    train_content = [content[i] for i in range(len(content)) if expanded_content[i] in shuffled_unique_train_content_set]
    test_content = [content[i] for i in range(len(content)) if expanded_content[i] in shuffled_unique_test_content_set]
    validate_content = [content[i] for i in range(len(content)) if expanded_content[i] in shuffled_unique_validate_content_set]
    return train_content, test_content, validate_content


# split content so that there is no overfitting with out split of validation and test data
def old_split_content(content, ratio):
    expanded_content = [el[1] if el[1] != '=' else el[0] for el in content]
    # print(len(content))
    unique_content = sorted(set(expanded_content))

    s = np.arange(len(unique_content))
    np.random.shuffle(s)

    split_num = math.floor(len(unique_content) * ratio)
    shuffled_unique_train_content = [unique_content[i] for i in range(len(s)) if s[i] >= split_num]

    shuffled_unique_train_content_set = set(shuffled_unique_train_content)
    shuffled_unique_validate_content = [unique_content[i] for i in range(len(s)) if s[i] < split_num]

    shuffled_unique_validate_content_set = set(shuffled_unique_validate_content)

    train_content = [content[i] for i in range(len(content)) if expanded_content[i] in shuffled_unique_train_content_set]
    validate_content = [content[i] for i in range(len(content)) if expanded_content[i] in shuffled_unique_validate_content_set]
    return train_content, validate_content


# X features that use MULTEX v3 as their encoding
def create_old_feature_dictionary(content):
    additional_data = [el[2] for el in content]
    possible_variants = sorted(set(additional_data))
    categories = sorted(set([el[0] for el in possible_variants]))

    feature_dictionary = []
    for category in categories:
        category_features = [1, category]
        examples_per_category = [el for el in possible_variants if el[0] == category]
        longest_element = max(examples_per_category, key=len)
        for i in range(1, len(longest_element)):
            possibilities_per_el = sorted(set([el[i] for el in examples_per_category if i < len(el)]))
            category_features[0] += len(possibilities_per_el)
            category_features.append(possibilities_per_el)
        feature_dictionary.append(category_features)
    return feature_dictionary


# X features that use MULTEX v3 as their encoding
def create_old_X_features(content, feature_dictionary):
    content = content
    X_other_features = []
    for el in content:
        X_el_other_features = []
        for feature in feature_dictionary:
            if el[2][0] == feature[1]:
                X_el_other_features.append(1)
                for i in range(2, len(feature)):
                    for j in range(len(feature[i])):
                        if i-1 < len(el[2]) and feature[i][j] == el[2][i-1]:
                            X_el_other_features.append(1)
                        else:
                            X_el_other_features.append(0)
            else:
                X_el_other_features.extend([0] * feature[0])
        X_other_features.append(X_el_other_features)
    return np.array(X_other_features)


def convert_to_MULTEXT_east_v4(old_features, feature_dictionary):
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
        if len(old_features) > 5:
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


def create_X_features(content, feature_dictionary):
    content = content
    X_other_features = []
    for el in content:
        X_el_other_features = []
        converted_el = ''.join(convert_to_MULTEXT_east_v4(list(el[2]), feature_dictionary))
#         converted_el = el[2]
        for feature in feature_dictionary:
            if converted_el[0] == feature[1]:
                X_el_other_features.append(1)
                for i in range(2, len(feature)):
                    for j in range(len(feature[i])):
                        if i-1 < len(converted_el) and feature[i][j] == converted_el[i-1]:
                            X_el_other_features.append(1)
                        else:
                            X_el_other_features.append(0)
            else:
                X_el_other_features.extend([0] * feature[0])
        X_other_features.append(X_el_other_features)
    return np.array(X_other_features)


def create_feature_dictionary():
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


def complete_feature_dict():
    # old: http://nl.ijs.si/ME/Vault/V3/msd/html/
    # new: http://nl.ijs.si/ME/V4/msd/html/
    # changes: http://nl.ijs.si/jos/msd/html-en/msd.diffs.html
    return [[27,
             'A',
             ['-', 'g', 's', 'p'],
             ['-', 'p', 'c', 's'],
             ['-', 'm', 'f', 'n'],
             ['-', 's', 'd', 'p'],
             ['-', 'n', 'g', 'd', 'a', 'l', 'i'],
             ['-', 'n', 'y']],
            [4, 'C', ['-', 'c', 's']],
            [1, 'I'],
            [28,
             'M',
             ['-', 'd', 'r', 'l'],
             ['-', 'c', 'o', 'p', 's'],
             ['-', 'm', 'f', 'n'],
             ['-', 's', 'd', 'p'],
             ['-', 'n', 'g', 'd', 'a', 'l', 'i'],
             ['-', 'n', 'y']],
            [22,
             'N',
             ['-', 'c', 'p'],
             ['-', 'm', 'f', 'n'],
             ['-', 's', 'd', 'p'],
             ['-', 'n', 'g', 'd', 'a', 'l', 'i'],
             ['-', 'n', 'y']],
            [41,
             'P',
             ['-', 'p', 's', 'd', 'r', 'x', 'g', 'q', 'i', 'z'],
             ['-', '1', '2', '3'],
             ['-', 'm', 'f', 'n'],
             ['-', 's', 'd', 'p'],
             ['-', 'n', 'g', 'd', 'a', 'l', 'i'],
             ['-', 's', 'd', 'p'],
             ['-', 'm', 'f', 'n'],
             ['-', 'y', 'b']],
            [1, 'Q'],
            [8, 'R', ['-', 'g', 'r'], ['-', 'p', 'c', 's']],
            [8, 'S', ['-', 'n', 'g', 'd', 'a', 'l', 'i']],
            [31,
             'V',
             ['-', 'm', 'a'],
             ['-', 'e', 'p', 'b'],
             ['-', 'n', 'u', 'p', 'r', 'f', 'c', 'm'],
             ['-', '1', '2', '3'],
             ['-', 's', 'p', 'd'],
             ['-', 'm', 'f', 'n'],
             ['-', 'n', 'y']]
            ]


def check_feature_letter_usage(X_other_features, feature_dictionary):
    case_numbers = np.sum(X_other_features, axis=0)
    arrays = [1] * 164
    letters = list(decode_X_features(feature_dictionary, [arrays]))
    print(sum(case_numbers))
    for i in range(len(letters)):
        print(letters[i] + ': ' + str(case_numbers[i]))


def dict_occurances_in_dataset_rate(content):
    feature_dictionary = complete_feature_dict()
    # case = 3107
    # print(content[case])
    # print(feature_dictionary)
    # X_other_features = create_X_features([content[case]], feature_dictionary)
    X_other_features = create_X_features(content, feature_dictionary)
    # print(X_other_features)
    # print(decode_X_features(feature_dictionary, X_other_features))
    X_other_features = np.array(X_other_features)

    case_numbers = np.sum(X_other_features, axis=0)
    print(case_numbers)
