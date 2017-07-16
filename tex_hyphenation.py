import sys
sys.path.insert(0, '../../../')
from prepare_data import *
dictionary, max_word, max_num_vowels, content, vowels, accetuated_vowels = create_dict()
feature_dictionary = create_feature_dictionary(content)


def read_hyphenation_pattern():
    with open('../../../hyphenation') as f:
        content = f.readlines()
    return [x[:-1] for x in content]


def find_hyphenation_patterns_in_text(text, pattern):
    res = []
    index = 0
    while index < len(text):
        index = text.find(pattern, index)
        if index == -1:
            break
        res.append(index)
        index += 1  # +2 because len('ll') == 2

    return res


def create_hyphenation_dictionary(hyphenation_pattern):
    dictionary = []
    for el in hyphenation_pattern:
        substring = ''
        anomalies_indices = []
        digit_location = 0
        for let in list(el):
            if let.isdigit():
                anomalies_indices.append([digit_location, int(let)])
            else:
                substring += let
                digit_location += 1
        dictionary.append([substring, anomalies_indices])
    return dictionary


def split_hyphenated_word(split, word):
    split = split[2:-2]
    print(split)
    word = list(word)[1:-1]
    res = []
    hyphenate = ''
    loc = 0
    for let in word:
        hyphenate += let
        if loc == len(split) or split[loc] % 2 == 1:
            res.append(hyphenate)
            hyphenate = ''
        loc += 1
    return res


def hyphenate_word(word, hyphenation_dictionary):
    word = word.replace('è', 'č')
    word = '.' + word + '.'
    split = [0] * (len(word) + 1)
    for pattern in hyphenation_dictionary:
        pattern_locations = find_hyphenation_patterns_in_text(word, pattern[0])
        for pattern_location in pattern_locations:
            for el_hyphenation_dictionary in pattern[1]:
                if split[pattern_location + el_hyphenation_dictionary[0]] < el_hyphenation_dictionary[1]:
                    split[pattern_location + el_hyphenation_dictionary[0]] = el_hyphenation_dictionary[1]
    return split_hyphenated_word(split, word)


hyphenation_pattern = read_hyphenation_pattern()
# ['zz', [{0:2},{1:1},{2:2}]]
hyphenation_dictionary = create_hyphenation_dictionary(hyphenation_pattern)
separated_word = hyphenate_word('izziv', hyphenation_dictionary)
print(separated_word)

all_words = []
i = 0
for el in content:
    separated_word = hyphenate_word(el[0], hyphenation_dictionary)
    all_words.append([el[0], separated_word])
    if i % 10000 == 0:
        print(str(i)+'/'+str(len(content)))
    i += 1

errors = []
errors2 = []
for word in all_words:
    for hyphenated_part in word[1]:
        num_vowels = 0
        for let in list(hyphenated_part):
            if let in vowels:
                num_vowels += 1
        if num_vowels == 0:
            for let in list(hyphenated_part):
                if let == 'r':
                    errors2.append(word[0])
                    num_vowels += 1
        if num_vowels != 1:
            errors.append(word)