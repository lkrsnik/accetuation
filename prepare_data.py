# -*- coding: utf-8 -*-
from __future__ import unicode_literals
# text in Western (Windows 1252)

import numpy as np
import h5py
import gc
import StringIO
import copy

def save_inputs(file_name, X, y):
    h5f = h5py.File(file_name, 'w')
    adict=dict(X=X, y=y)
    for k,v in adict.items():
        h5f.create_dataset(k,data=v)
    h5f.close()

def create_and_save_inputs(file_name, part, X, y, X_pure):
    # X, y, X_pure = generate_full_vowel_matrix_inputs()
    h5f = h5py.File(file_name + part + '.h5', 'w')
    adict=dict(X=X, y=y, X_pure=X_pure)
    for k,v in adict.items():
        h5f.create_dataset(k,data=v)
    h5f.close()

def create_and_save_shuffle_vector(file_name, shuffle_vector):
    # X, y, X_pure = generate_full_vowel_matrix_inputs()
    h5f = h5py.File(file_name + '_shuffle_vector.h5', 'w')
    adict=dict(shuffle_vector=shuffle_vector)
    for k,v in adict.items():
        h5f.create_dataset(k,data=v)
    h5f.close()

def load_shuffle_vector(file_name):
    h5f = h5py.File(file_name,'r')
    shuffle_vector = h5f['shuffle_vector'][[179859, 385513, 893430]]

    h5f.close()
    return shuffle_vector

def load_inputs(file_name):
    h5f = h5py.File(file_name,'r')
    X = h5f['X'][:]
    y = h5f['y'][:]

    h5f.close()
    return X, y

def load_extended_inputs(file_name, obtain_range):
    h5f = h5py.File(file_name,'r')
    X = h5f['X'][obtain_range[0]:obtain_range[1]]
    y = h5f['y'][obtain_range[0]:obtain_range[1]]
    X_pure = h5f['X_pure'][obtain_range[0]:obtain_range[1]]

    h5f.close()
    return X, y, X_pure

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
    with open('../data/SlovarIJS_BESEDE_utf8.lex') as f:
        content = f.readlines()
    print('CONTENT READ SUCCESSFULY')
    return [x.decode('utf8').split('\t') for x in content]


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


def generate_full_vowel_matrix_inputs(name, split_number):
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
    # print (2018553 * max_word * len(dictionary) / (2**30.0))
    print('GENERATING X AND y...')
    # X = np.zeros((len(content), max_word*len(dictionary)))
    # y = np.zeros((len(content), max_num_vowels * max_num_vowels))
    # X = np.zeros((2018553, max_word, len(dictionary)))
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
                    # create_and_save_inputs(name, str(current_part_generation), np.array(X), np.zeros(len(X)), np.array(X_pure))

                    # adict = dict(X=np.array(X), y=np.zeros(len(X)), X_pure=np.array(X_pure))
                    # for k, v in adict.items():
                    #     h5f.create_dataset(k, data=v)
                    # print (len(np.array(X)))
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
            print i
        # text_file.write("Purchase Amount: %s" % TotalAmount)
        j = 0
        # X.append(word)
        # word_accetuations = []
        # num_vowels = 0
        # for c in list(el[3]):
        #     index = 0
        #     if is_vowel(el[3], j, vowels):
        #         num_vowels += 1
        #     for d in accetuated_vowels:
        #         if c == d:
        #             word_accetuations.append(num_vowels)
        #             break
        #         index += 1
        #     j += 1
        # y[i][generate_presentable_y(word_accetuations, list(el[3]), max_num_vowels)] = 1
        i += 1

    print('Saving part ' + str(current_part_generation))
    # create_and_save_inputs(name, str(current_part_generation), np.array(X), np.zeros(len(X)), np.array(X_pure))

    data_X[old_num_all_vowels:num_all_vowels] = np.array(X)
    data_y[old_num_all_vowels:num_all_vowels] = np.array(y)
    data_X_pure[old_num_all_vowels:num_all_vowels] = np.array(X_pure)

    # adict = dict(X=X, y=y, X_pure=X_pure)
    # for k, v in adict.items():
    #     h5f.create_dataset(k, data=v)


    h5f.close()


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




def shuffle_full_vowel_inputs(name, orderd_name, parts):
#     internal_representations/inputs/X_ordered_part
    dictionary, max_word, max_num_vowels, content, vowels, accetuated_vowels = create_dict()
    num_all_vowels = count_vowels(content, vowels)
    num_all_vowels = 12


    s = np.arange(num_all_vowels)
    np.random.shuffle(s)
    # create_and_save_shuffle_vector(name, s)

    # s = load_shuffle_vector('internal_representations/inputs/X_shuffled_part_shuffle_vector.h5')

# try:
    #     h5f.close()
    # except Exception, e:
    #     pass

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
                # print targeted_range[0]
                # print targeted_range[1]
                # print s[j]
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
        # create_and_save_inputs(name, str(h), new_X, new_y, new_X_pure)
        # a =
        # print (a.shape)
        # print s
        # for el in np.array(new_X):
        #     print el
        # print 'new_X ' + str(new_X) + ' section_range[0] ' + str(section_range[0]) + ' section_range[1] ' + str(section_range[1])
        # print new_X.shape
        # print type(new_X)
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
