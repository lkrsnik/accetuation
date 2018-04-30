# Words proccesed: 650250
# Word indeks: 50023
# Word number: 50023

from lxml import etree
import time
from prepare_data import *

# def xml_words_generator(xml_path):
#     for event, element in etree.iterparse(xml_path, tag="LexicalEntry", encoding="UTF-8"):
#         words = []
#         for child in element:
#             if child.tag == 'WordForm':
#                 msd = None
#                 word = None
#                 for wf in child:
#                     if 'att' in wf.attrib and wf.attrib['att'] == 'msd':
#                         msd = wf.attrib['val']
#                     elif wf.tag == 'FormRepresentation':
#                         for form_rep in wf:
#                             if form_rep.attrib['att'] == 'zapis_oblike':
#                                 word = form_rep.attrib['val']
#                         #if msd is not None and word is not None:
#                         #    pass
#                         #else:
#                         #    print('NOOOOO')
#                         words.append([word, '', msd, word])
#         yield words
#
#
# gen = xml_words_generator('data/Sloleks_v1.2_p2.xml')
word_glob_num = 0
word_limit = 50000
iter_num = 50000
word_index = 0

# iter_index = 0
# words = []
#
# lexical_entries_load_number = 0
# lexical_entries_save_number = 0
#
# # INSIDE
# # word_glob_num = 1500686
# word_glob_num = 1550705
#
# # word_limit = 1500686
# word_limit = 1550705
#
# iter_index = 31

# done_lexical_entries = 33522
data = Data('s', shuffle_all_inputs=False)
accentuated_content = data._read_content('data/new_sloleks/new_sloleks.tab')

start_timer = time.time()

print('Copy initialization complete')
with open("data/new_sloleks/final_sloleks.xml", "ab") as myfile:
    # myfile2 = open('data/new_sloleks/p' + str(iter_index) + '.xml', 'ab')
    for event, element in etree.iterparse('data/Sloleks_v1.2.xml', tag="LexicalEntry", encoding="UTF-8", remove_blank_text=True):
        # if word_glob_num >= word_limit:
        #     myfile2.close()
        #     myfile2 = open('data/new_sloleks/p' + str(iter_index) + '.xml', 'ab')
        #     iter_index += 1
        #     print("Words proccesed: " + str(word_glob_num))
        #
        #     print("Word indeks: " + str(word_index))
        #     print("Word number: " + str(len(words)))
        #
        #     # print("lexical_entries_load_number: " + str(lexical_entries_load_number))
        #     # print("lexical_entries_save_number: " + str(lexical_entries_save_number))
        #
        #     end_timer = time.time()
        #     print("Elapsed time: " + "{0:.2f}".format((end_timer - start_timer) / 60.0) + " minutes")
        lemma = ''
        accentuated_word_location = ''
        accentuated_word = ''
        for child in element:
            if child.tag == 'Lemma':
                for wf in child:
                    if 'att' in wf.attrib and wf.attrib['att'] == 'zapis_oblike':
                        lemma = wf.attrib['val']
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

                        word_index = (word_index - 500) % len(accentuated_content)
                        word_index_sp = (word_index - 1) % len(accentuated_content)
                        while word_index != word_index_sp:
                            if word == accentuated_content[word_index][0] and msd == accentuated_content[word_index][2] and \
                               lemma == accentuated_content[word_index][1]:
                                accentuated_word_location = accentuated_content[word_index][4]
                                accentuated_word = accentuated_content[word_index][5][:-1]
                                del(accentuated_content[word_index])
                                break
                            word_index = (word_index + 1) % len(accentuated_content)

                        error = word_index == word_index_sp
                        if word_index == word_index_sp and word == accentuated_content[word_index][0] and msd == accentuated_content[word_index][2] \
                            and lemma == accentuated_content[word_index][1]:
                            accentuated_word_location = accentuated_content[word_index][4]
                            accentuated_word = accentuated_content[word_index][5][:-1]
                            error = False
                            del(accentuated_content[word_index])

                        if error:
                            print('ERROR IN ' + word + ' : ' + lemma + ' : ' + msd)
                            # print('ERROR IN ' + word + ' : ' + accentuated_content[word_index][0] + ' OR ' + msd + ' : '
                            #       + accentuated_content[word_index][2])
                        # words.append([word, '', msd, word])

                        new_element = etree.Element('feat')
                        new_element.attrib['att'] = 'naglasna_mesta_besede'
                        new_element.attrib['val'] = accentuated_word_location
                        wf.append(new_element)

                        new_element = etree.Element('feat')
                        new_element.attrib['att'] = 'naglašena_beseda'
                        new_element.attrib['val'] = accentuated_word
                        wf.append(new_element)
                        word_glob_num += 1
                        # word_index += 1

        # print(etree.tostring(element, encoding="UTF-8"))
        # myfile2.write(etree.tostring(element, encoding="UTF-8", pretty_print=True))
        if word_glob_num > word_limit:
            # print('Proccessed ' + str(word_glob_num) + ' words')
            end_timer = time.time()
            # print("Elapsed time: " + "{0:.2f}".format((end_timer - start_timer) / 60.0) + " minutes")
            word_limit += iter_num
        myfile.write(etree.tostring(element, encoding="UTF-8", pretty_print=True))
        element.clear()
