# -*- coding: utf-8 -*-
from __future__ import unicode_literals
# text in Western (Windows 1252)

import numpy as np
import h5py
import gc

def save_inputs(file_name, X, y):
    h5f = h5py.File(file_name, 'w')
    adict=dict(X=X, y=y)
    for k,v in adict.items():
        h5f.create_dataset(k,data=v)
    h5f.close()

def create_and_save_inputs(file_name):
    X, y, X_pure = generate_full_vowel_matrix_inputs()
    h5f = h5py.File(file_name, 'w')
    adict=dict(X=X, y=y, X_pure=X_pure)
    for k,v in adict.items():
        h5f.create_dataset(k,data=v)
    h5f.close()

def load_inputs(file_name):
    h5f = h5py.File(file_name,'r')
    X = h5f['X'][:]
    y = h5f['y'][:]

    h5f.close()
    return X, y

def save_model(model, file_name):
    h5f = h5py.File(file_name, 'w')
    adict=dict(W1=model['W1'], b1=model['b1'], W2=model['W2'], b2=model['b2'])
    for k,v in adict.items():
        h5f.create_dataset(k,data=v)

    h5f.close()

def load_model(file_name):
    h5f = h5py.File(file_name,'r')
    model = {}
    W1.set_value(h5f['W1'][:])
    b1.set_value(h5f['b1'][:])
    W2.set_value(h5f['W2'][:])
    b2.set_value(h5f['b2'][:])
    h5f.close()
    return model

def read_content():
    print('READING CONTENT...')
    with open('../../data/SlovarIJS_BESEDE_utf8.lex') as f:
        content = f.readlines()
    print('CONTENT READ SUCCESSFULY')
    return [x.decode('utf8').split('\t') for x in content]


def is_vowel(word_list, position, vowels):
    if word_list[position] in vowels:
        return True
    if word_list[position] == u'r' and     (position - 1 < 0 or word_list[position - 1] not in vowels) and     (position + 1 >= len(word_list) or word_list[position + 1] not in vowels):
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

    dictionary = ['']
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
                if c not in dictionary:
                    dictionary.append(c)
                i += 1
            for c in list(el[0]):
                if c not in dictionary:
                    dictionary.append(c)
            if num_vowels > max_num_vowels:
                max_num_vowels = num_vowels
        except Exception, e:
            print line - 1
            print el
            break
        line += 1
    dictionary = sorted(dictionary)
    max_num_vowels += 1
    print('DICTIONARY CREATION SUCCESSFUL!')
    return dictionary, max_word, max_num_vowels, content, vowels, accetuated_vowels


# GENERATE X and y
def generate_presentable_y(accetuations_list, word_list, max_num_vowels):
    while len(accetuations_list) < 2:
        accetuations_list.append(0)
    if len(accetuations_list) > 2:
        accetuations_list = accetuations_list[:2]
    accetuations_list = np.array(accetuations_list)
    final_position = accetuations_list[0] + max_num_vowels * accetuations_list[1]
    return final_position
    
def shuffle_inputs(X, y, X_pure):
    s = np.arange(X.shape[0])
    np.random.shuffle(s)
    X = X[s]
    y = y[s]
    X_pure = X_pure[s]
    return X, y, X_pure

def generate_inputs():
    dictionary, max_word, max_num_vowels, content, vowels, accetuated_vowels = create_dict()
    
    print('GENERATING X AND y...')
    X = np.zeros((len(content), max_word*len(dictionary)))
    y = np.zeros((len(content), max_num_vowels * max_num_vowels ))

    i = 0
    for el in content:
        j = 0
        for c in list(el[0]):
            index = 0
            for d in dictionary:
                if c == d:
                    X[i][index + j * max_word] = 1
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
    print('GENERATION SUCCESSFUL!')
    print('SHUFFELING INPUTS...')
    X, y = shuffle_inputs(X, y)
    print('INPUTS SHUFFELED!')
    return X, y


def generate_matrix_inputs():
    dictionary, max_word, max_num_vowels, content, vowels, accetuated_vowels = create_dict()
    
    print('GENERATING X AND y...')
    # X = np.zeros((len(content), max_word*len(dictionary)))
    y = np.zeros((len(content), max_num_vowels * max_num_vowels ))

    X = []

    i = 0
    for el in content:
        # j = 0
        word = []
        for c in list(el[0]):
            index = 0
            character = np.zeros(len(dictionary))
            for d in dictionary:
                if c == d:
                    # X[i][index + j * max_word] = 1
                    character[index] = 1
                    break
                index += 1
            word.append(character)
            # j += 1
        j = 0
        X.append(word)
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
    X = np.array(X)
    print('GENERATION SUCCESSFUL!')
    print('SHUFFELING INPUTS...')
    X, y = shuffle_inputs(X, y)
    print('INPUTS SHUFFELED!')
    return X, y


def generate_full_matrix_inputs():
    dictionary, max_word, max_num_vowels, content, vowels, accetuated_vowels = create_dict()
    
    print('GENERATING X AND y...')
    # X = np.zeros((len(content), max_word*len(dictionary)))
    y = np.zeros((len(content), max_num_vowels * max_num_vowels ))
    X = np.zeros((len(content), max_word, len(dictionary)))

    i = 0
    for el in content:
        j = 0
        # word = []
        for c in list(el[0]):
            index = 0
            # character = np.zeros(len(dictionary))
            for d in dictionary:
                if c == d:
                    X[i][j][index] = 1
                    # character[index] = 1
                    break
                index += 1
            # word.append(character)
            j += 1
        j = 0
        # X.append(word)
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
    # X = np.array(X)
    print('GENERATION SUCCESSFUL!')
    print('SHUFFELING INPUTS...')
    X, y = shuffle_inputs(X, y)
    print('INPUTS SHUFFELED!')
    return X, y

def count_vowels(content, vowels):
    num_all_vowels = 0
    for el in content:
        for m in range(len(el[0])):
            if is_vowel(list(el[0]), m, vowels):
                num_all_vowels += 1
    return num_all_vowels


def generate_full_vowel_matrix_inputs():
    dictionary, max_word, max_num_vowels, content, vowels, accetuated_vowels = create_dict()
    gc.collect()
    # print (2018553 * max_word * len(dictionary) / (2**30.0))
    print('GENERATING X AND y...')
    # X = np.zeros((len(content), max_word*len(dictionary)))
    y = np.zeros((len(content), max_num_vowels * max_num_vowels ))
    # X = np.zeros((2018553, max_word, len(dictionary)))
    X_pure = []
    X = []

    i = 0
    for el in content:
        j = 0
        # word = []
        X_el = np.zeros((max_word, len(dictionary)))
        for c in list(el[0]):
            index = 0
            # character = np.zeros(len(dictionary))
            for d in dictionary:
                if c == d:
                    X_el[j][index] = 1
                    # character[index] = 1
                    break
                index += 1
            # word.append(character)
            j += 1
        # for c in list(el[0]):
        vowel_i = 0
        for m in range(len(el[0])):
            if is_vowel(list(el[0]), m, vowels):
                X.append(X_el)
                X_pure.append(vowel_i)
                vowel_i += 1
        j = 0
        # X.append(word)
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
    # print(len(X))
    # del X_pure
    # del dictionary, max_word, max_num_vowels, content, vowels, accetuated_vowels
    
    X = np.array(X)
    X_pure = np.array(X_pure)
    print('GENERATION SUCCESSFUL!')
    print('SHUFFELING INPUTS...')
    X, y, X_pure = shuffle_inputs(X, y, X_pure)
    print('INPUTS SHUFFELED!')
    return X, y, X_pure


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
