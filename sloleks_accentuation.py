# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import numpy as np
from keras.models import load_model
import sys

from prepare_data import *

np.random.seed(7)

data = Data('l', shuffle_all_inputs=False)
content = data._read_content('data/SlovarIJS_BESEDE_utf8.lex')
dictionary, max_word, max_num_vowels, vowels, accented_vowels = data._create_dict(content)
feature_dictionary = data._create_slovene_feature_dictionary()
syllable_dictionary = data._create_syllables_dictionary(content, vowels)
accented_vowels = ['ŕ', 'á', 'ä', 'é', 'ë', 'ě', 'í', 'î', 'ó', 'ô', 'ö', 'ú', 'ü']


letter_location_model, syllable_location_model, syllabled_letters_location_model = data.load_location_models(
    'cnn/word_accetuation/cnn_dictionary/v3_10/20_test_epoch.h5',
    'cnn/word_accetuation/syllables/v2_4/20_test_epoch.h5',
    'cnn/word_accetuation/syllabled_letters/v2_5_3/20_test_epoch.h5')

letter_type_model, syllable_type_model, syllabled_letter_type_model = data.load_type_models(
    'cnn/accent_classification/letters/v2_1/20_test_epoch.h5',
    'cnn/accent_classification/syllables/v1_0/20_test_epoch.h5',
    'cnn/accent_classification/syllabled_letters/v1_0/20_test_epoch.h5')

from lxml import etree


def xml_words_generator(xml_path):
    for event, element in etree.iterparse(xml_path, tag="LexicalEntry", encoding="UTF-8"):
        words = []
        for child in element:
            if child.tag == 'WordForm':
                msd = None
                word = None
                for wf in child:
                    if 'att' in wf.attrib and wf.attrib['att'] == 'msd':
                        msd = wf.attrib['val']
                    elif wf.tag == 'FormRepresentation':
                        for form_rep in wf:
                            if form_rep.attrib['att'] == 'zapis_oblike':
                                word = form_rep.attrib['val']
                        # if msd is not None and word is not None:
                        #    pass
                        # else:
                        #    print('NOOOOO')
                        words.append([word, '', msd, word])
        yield words


gen = xml_words_generator('data/Sloleks_v1.2.xml')

# Words proccesed: 650250
# Word indeks: 50023
# Word number: 50023

from lxml import etree
import time

gen = xml_words_generator('data/Sloleks_v1.2.xml')
word_glob_num = 0
word_limit = 0
iter_num = 50000
word_index = 0
start_timer = time.time()
iter_index = 0
words = []

lexical_entries_load_number = 0
lexical_entries_save_number = 0

# INSIDE
word_glob_num = 1500686

word_limit = 50000
iter_index = 30

done_lexical_entries = 33522

import gc

with open("data/new_sloleks/new_sloleks.xml", "ab") as myfile:
    myfile2 = open('data/new_sloleks/p' + str(iter_index) + '.xml', 'ab')
    for event, element in etree.iterparse('data/Sloleks_v1.2.xml', tag="LexicalEntry", encoding="UTF-8", remove_blank_text=True):
        # LOAD NEW WORDS AND ACCENTUATE THEM
        # print("HERE")

        if lexical_entries_save_number < done_lexical_entries:
            g = next(gen)
            # print(lexical_entries_save_number)
            lexical_entries_save_number += 1
            lexical_entries_load_number += 1
            print(lexical_entries_save_number)
            del g
            gc.collect()
            continue

        if word_glob_num >= word_limit:
            myfile2.close()
            myfile2 = open('data/new_sloleks/p' + str(iter_index) + '.xml', 'ab')
            iter_index += 1
            print("Words proccesed: " + str(word_glob_num))

            print("Word indeks: " + str(word_index))
            print("Word number: " + str(len(words)))

            print("lexical_entries_load_number: " + str(lexical_entries_load_number))
            print("lexical_entries_save_number: " + str(lexical_entries_save_number))

            end_timer = time.time()
            print("Elapsed time: " + "{0:.2f}".format((end_timer - start_timer) / 60.0) + " minutes")

            word_index = 0
            words = []

            while len(words) < iter_num:
                try:
                    words.extend(next(gen))
                    lexical_entries_load_number += 1
                except:
                    break
            # if word_glob_num > 1:
            #    break

            data = Data('l', shuffle_all_inputs=False)
            location_accented_words, accented_words = data.accentuate_word(words, letter_location_model, syllable_location_model,
                                                                           syllabled_letters_location_model,
                                                                           letter_type_model, syllable_type_model, syllabled_letter_type_model,
                                                                           dictionary, max_word, max_num_vowels, vowels, accented_vowels,
                                                                           feature_dictionary, syllable_dictionary)

            word_limit += len(words)

        # READ DATA
        for child in element:
            if child.tag == 'WordForm':
                msd = None
                word = None
                for wf in child:
                    if wf.tag == 'FormRepresentation':
                        new_element = etree.Element('feat')
                        new_element.attrib['att'] = 'naglasna_mesta_oblike'
                        new_element.attrib['val'] = location_accented_words[word_index]
                        wf.append(new_element)

                        new_element = etree.Element('feat')
                        new_element.attrib['att'] = 'naglašena_oblika'
                        new_element.attrib['val'] = accented_words[word_index]
                        wf.append(new_element)
                        word_glob_num += 1
                        word_index += 1

        # print(etree.tostring(element, encoding="UTF-8"))
        myfile2.write(etree.tostring(element, encoding="UTF-8", pretty_print=True))
        myfile.write(etree.tostring(element, encoding="UTF-8", pretty_print=True))
        element.clear()
        lexical_entries_save_number += 1
