from copy import copy
import sys


vowels = ['à', 'á', 'ä', 'é', 'ë', 'ì', 'í', 'î', 'ó', 'ô', 'ö', 'ú', 'ü', 'a', 'e', 'i', 'o', 'u', 'O', 'E']

def syllable_stressed(syllable):
    stressed_letters = [u'ŕ', u'á', u'ä', u'é', u'ë', u'ě', u'í', u'î', u'ó', u'ô', u'ö', u'ú', u'ü']
    for letter in syllable:
        if letter in stressed_letters:
            return True
    return False

def is_vowel(word_list, position, vowels):
    if word_list[position] in vowels:
        return True
    if (word_list[position] == u'r' or word_list[position] == u'R') and (position - 1 < 0 or word_list[position - 1] not in vowels) and (
                        position + 1 >= len(word_list) or word_list[position + 1] not in vowels):
        return True
    return False

def get_voiced_consonants():
    return ['m', 'n', 'v', 'l', 'r', 'j', 'y', 'w', 'F', 'N']

def get_resonant_silent_consonants():
    return ['b', 'd', 'z', 'ž', 'g']

def get_nonresonant_silent_consonants():
    return ['p', 't', 's', 'š', 'č', 'k', 'f', 'h', 'c', 'x']

def split_consonants(consonants):
    voiced_consonants = get_voiced_consonants()
    resonant_silent_consonants = get_resonant_silent_consonants()
    unresonant_silent_consonants = get_nonresonant_silent_consonants()
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

def create_syllables(word, vowels):
    word_list = list(word)
    consonants = []
    syllables = []
    for i in range(len(word_list)):
        if is_vowel(word_list, i, vowels):
            if syllables == []:
                consonants.append(word_list[i])
                syllables.append(''.join(consonants))
            else:
                left_consonants, right_consonants = split_consonants(list(''.join(consonants).lower()))
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


def convert_to_SAMPA(word):
    syllables = create_syllables(word, vowels)
    letters_in_stressed_syllable = [False] * len(word)
    # print(syllables)
    l = 0
    for syllable in syllables:
        if syllable_stressed(syllable):
            for i in range(len(syllable)):
                letters_in_stressed_syllable[l + i] = True
        # print(l)
        l += len(syllable)
    previous_letter = ''
    word = list(word)
    for i in range(len(word)):
        if word[i] == 'e':
            word[i] = 'E'
        elif word[i] == 'o':
            word[i] = 'O'
        elif word[i] == 'š':
            word[i] = 'S'
        elif word[i] == 'ž':
            word[i] = 'Z'
        elif word[i] == 'h':
            word[i] = 'x'
        elif word[i] == 'c':
            word[i] = 'ts'
        elif word[i] == 'č':
            word[i] = 'tS'
        elif word[i] == 'á':
            word[i] = 'a:'
        elif word[i] == 'ä':
            word[i] = 'a'
        elif word[i] == 'é':
            word[i] = 'e:'
        elif word[i] == 'ë':
            word[i] = 'E'
        elif word[i] == 'ě':
            word[i] = 'E:'
        elif word[i] == 'í':
            word[i] = 'i:'
        elif word[i] == 'î':
            word[i] = 'i'
        elif word[i] == 'ó':
            word[i] = 'o:'
        elif word[i] == 'ô':
            word[i] = 'O:'
        elif word[i] == 'ö':
            word[i] = 'O'
        elif word[i] == 'ú':
            word[i] = 'u:'
        elif word[i] == 'ü':
            word[i] = 'u'
        elif word[i] == 'ŕ':
            word[i] = '@r'

    if letters_in_stressed_syllable[0]:
        word[0] = '\"' + word[0]
    for i in range(1, len(letters_in_stressed_syllable)):
        if not letters_in_stressed_syllable[i - 1] and letters_in_stressed_syllable[i]:
            word[i] = '\"' + word[i]
            # if letters_in_stressed_syllable[i - 1] and not letters_in_stressed_syllable[i]:
            #    word[i - 1] = word[i - 1] + ':'
    # if letters_in_stressed_syllable[-1]:
    #    word[-1] = word[-1] + ':'

    word = list(''.join(word))

    previous_letter_i = -1
    letter_i = 0
    next_letter_i = 1
    if word[0] == '\"':
        letter_i = 1
        if word[2] == ':':
            if len(word) > 3:
                next_letter_i = 3
            else:
                #if word[next_letter_i] == 'l':
                #    word[next_letter_i] = 'l\''
                #elif word[next_letter_i] == 'n':
                #    word[next_letter_i] = 'n\''
                return ''.join(word)
        else:
            next_letter_i = 2
    elif len(word) > 1 and word[1] == '\"':
        next_letter_i = 2
    # {('m', 'f'): 'F'}

    new_word = copy(word)
    while True:
        if word[letter_i] == 'm' and (word[next_letter_i] == 'f' or word[next_letter_i] == 'v'):
            new_word[letter_i] = 'F'
        elif word[letter_i] == 'n' and (word[next_letter_i] == 'k' or word[next_letter_i] == 'g' or word[next_letter_i] == 'x'):
            new_word[letter_i] = 'N'
        elif word[letter_i] == 'n' and (word[next_letter_i] == 'f' or word[next_letter_i] == 'v'):
            new_word[letter_i] = 'F'
        elif word[letter_i] == 'n' and not word[next_letter_i] in vowels and letter_i == len(word) - 2:
            new_word[letter_i] = 'n\''
        elif word[letter_i] == 'l' and not word[next_letter_i] in vowels and letter_i == len(word) - 2:
            new_word[letter_i] = 'l\''
        elif previous_letter_i >= 0 and word[letter_i] == 'v' and not word[previous_letter_i] in vowels and word[
            next_letter_i] in get_voiced_consonants():
            new_word[letter_i] = 'w'
        elif previous_letter_i >= 0 and word[letter_i] == 'v' and not word[previous_letter_i] in vowels and word[
            next_letter_i] in get_nonresonant_silent_consonants():
            new_word[letter_i] = 'W'
        elif word[letter_i] == 'p' and word[next_letter_i] == 'm':
            new_word[letter_i] = 'p_n'
        elif word[letter_i] == 'p' and (word[next_letter_i] == 'f' or word[next_letter_i] == 'v'):
            new_word[letter_i] = 'p_f'
        elif word[letter_i] == 'b' and word[next_letter_i] == 'm':
            new_word[letter_i] = 'b_n'
        elif word[letter_i] == 'b' and (word[next_letter_i] == 'f' or word[next_letter_i] == 'v'):
            new_word[letter_i] = 'b_f'
        elif word[letter_i] == 't' and word[next_letter_i] == 'l':
            new_word[letter_i] = 't_l'
        elif word[letter_i] == 't' and word[next_letter_i] == 'n':
            new_word[letter_i] = 't_n'
        elif word[letter_i] == 'd' and word[next_letter_i] == 'l':
            new_word[letter_i] = 'd_l'
        elif word[letter_i] == 'd' and word[next_letter_i] == 'n':
            new_word[letter_i] = 'd_n'

        if len(word) > next_letter_i + 1:
            if word[next_letter_i + 1] == ':' or word[next_letter_i + 1] == '\"':
                if len(word) > next_letter_i + 2:
                    previous_letter_i = letter_i
                    letter_i = next_letter_i
                    next_letter_i = next_letter_i + 2
                else:
                    #if word[next_letter_i] == 'l':
                    #    new_word[next_letter_i] = 'l\''
                    #elif word[next_letter_i] == 'n':
                    #    new_word[next_letter_i] = 'n\''
                    return ''.join(new_word)
            else:
                previous_letter_i = letter_i
                letter_i = next_letter_i
                next_letter_i = next_letter_i + 1
        else:
            #if word[next_letter_i] == 'l':
            #    new_word[next_letter_i] = 'l\''
            #elif word[next_letter_i] == 'n':
            #    new_word[next_letter_i] = 'n\''
            return ''.join(new_word)
            # print(word)

result = convert_to_SAMPA(sys.argv[1])
final_result = result.replace('\"', '\'')
print(final_result)
#return final_result
