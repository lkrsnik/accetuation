# -*- coding: utf-8 -*-
from __future__ import unicode_literals

# Words proccesed: 650250
# Word indeks: 50023
# Word number: 50023

from lxml import etree

word_glob_num = 0
word_limit = 50000
iter_num = 50000
word_index = 0
accented_places = 0
accented_words = 0
enters = 0

for event, element in etree.iterparse('data/new_sloleks/final_sloleks.xml', tag="LexicalEntry", encoding="UTF-8", remove_blank_text=True):
    for child in element:
        for wf in child:
            if wf.tag == 'FormRepresentation':
                for form_rep in wf:
                    if form_rep.attrib['att'] == 'naglasna_mesta_besede':
                        accented_places += 1
                        if '\n' in list(form_rep.attrib['val']):
                            enters += 1
                    if form_rep.attrib['att'] == 'naglašena_beseda':
                        accented_words += 1
                        if '\n' in list(form_rep.attrib['val']):
                            enters += 1

    element.clear()

print(accented_places)
print(accented_words)
print(enters)
