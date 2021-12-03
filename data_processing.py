import os
import re
import numpy as np
import spacy
from tqdm import tqdm

# TODO: implement sanity check, to check whether words in labelfile actually correspond to the same word in parsed text

class DataProcessor():
    def __init__(self):
        pass

    def read_file(self, file):
        with open(file, 'r') as f:
            data = f.read()
        return data

    def extract_labelinfo(self, label_text):
        '''
        Create list of label information
        Extract label and start and end positions of the characters,
        only for lines starting with 'T' (ignore 'AnnotatorNotes')
        '''
        label_text = [line.split('\t') for line in label_text.split('\n')[:-1]]
        label_data = []
        for info in label_text:
            if info[0][0] == 'T':
                # Extract all start and end char positions, separated by ' ' and ';'
                label_data.append(re.split(' |;', info[1]))
        return label_data

    def check_email(self, word):
        # Check whether both @ and . occur in word
        if '@' in word and '.' in word:
            return True
        return False

    def check_multiple_dots(self, word, char):
        # Check whether '.' occurs (at least) twice
        # To catch websites and abbreviations, e.g. i.e.
        if word.count(char) > 1:
            return True
        return False

    # TODO: "(this)." will for now turn into ['(', 'this', ').']
            # Might want to separate ').'?
            # It is possible to treat (), [], " ", ' ', < >, {} combinations different from rest
    def separate_alnumgroup(self, word, pos, end_pos):
        skip_next_word = False
        # Only non-alnums
        if len(word) == end_pos-pos:
            subwords = [word]
        # Alnumgroup at beginning
        elif pos == 0:
            subwords = [word[pos:end_pos], word[end_pos:]]
        # Alnumgroup at end
        elif end_pos == len(word):
            subwords = [word[:pos], word[pos:]]
            skip_next_word = True
        # Alnumgroup between alnums
        else:
            subwords = [word[:pos], word[pos:end_pos], word[end_pos:]]
            skip_next_word = True
        return subwords, skip_next_word

    def separate_first_char(self, word):
        # Separate first character from rest of string
        subword1 = word[0]
        subword2 = word[1:]
        return [subword1, subword2]

    def separate_last_char(self, word):
        # Separate last character from rest of string
        subword1 = word[:-1]
        subword2 = word[-1]
        return [subword1, subword2]

    def separate_apostrophe_char(self, word, pos):
        # n't: haven't -> have n't
        # 've: you've -> you 've
        # I'm: I'm -> I 'm
        # 's: he's -> he 's
        skip_next_word = False
        subwords = []
        if not word[pos+1].isalnum():
            subwords = [word[:pos], word[pos], word[pos+1:]]
            return subwords, skip_next_word
        # Check "n't" ( ' was already checked to not be last char)
        elif word[pos-1].lower() == 'n' and word[pos+1].lower() == 't':
            subwords = [word[:pos-1], word[pos-1:pos+2]]
            if len(word) > pos+2:
                subwords.append(word[pos+2:])
            skip_next_word = True    
            return subwords, skip_next_word

        elif word[pos+1].lower() == 'm' or word[pos+1].lower() == 's':
            subwords = [word[:pos], word[pos:pos+2]]
            if len(word) > pos+2:
                subwords.append(word[pos+2:])
            skip_next_word = True
            return subwords, skip_next_word

        elif len(word) > pos+2: 
            if word[pos+1].lower() == 'v' and word[pos+2].lower() == 'e':
                subwords = [word[:pos], word[pos:pos+3]]
                if len(word) > pos+3:
                    subwords.append(word[pos+2:])
                skip_next_word = True
                return subwords, skip_next_word
        return [word], skip_next_word
                
    def separate_word(self, word, pos):
        subwords = [word[:pos], word[pos], word[pos+1:]]
        return subwords

    def create_subwords(self, word):
        non_alnum = re.search("[^0-9a-zA-Z]+", word)
        pos = non_alnum.start()
        end_pos = non_alnum.end()
        char = non_alnum.group(0)
        skip_next_word = False
        # Neighbouring non-alnums
        if end_pos-pos > 1:
            subwords, skip = self.separate_alnumgroup(word, pos, end_pos)
            skip_next_word = skip
        # First char to be separated
        elif pos == 0:
            subwords = self.separate_first_char(word)
        # Last char to be separated
        elif pos == len(word)-1:
            subwords = self.separate_last_char(word)
        # Apostrophe (not at start/end): special case
        elif char =='\'':
            subwords, skip = self.separate_apostrophe_char(word, pos)
            skip_next_word = skip

        elif self.check_email(word):
            subwords = [word]

        elif char == '.' and self.check_multiple_dots(word, char):
            subwords = [word]

        else:
            subwords = self.separate_word(word, pos)
            # Skip next always true, because always single non-alnum within word
            skip_next_word = True

        return subwords, skip_next_word

    def get_words_and_positions(self, text_file):
        text = self.read_file(text_file)
        text = text.replace('\n', ' ')
        text = text.split(' ')[:-1] # Last character will be '', discarced here
        char_positions = []
        i, position = 0, 0
        while i<len(text):
            word = text[i]
            if not word.isalnum() and len(word) > 1:
                subwords, skip_next = self.create_subwords(word)
                # Only need to update when word was separated into subwords
                if len(subwords) > 1:
                    j = i
                    del text[i]
                    for subword in subwords:
                        text.insert(j, subword)
                        j += 1
                    char_positions.append(position)
                    position += len(subwords[0])
                    if skip_next:
                        char_positions.append(position)
                        position += len(subwords[1])
                        # Account for space
                        if len(subwords) == 2:
                            position += 1
                        i += 1
                # Current char position: + wordlength + space
                else:
                    char_positions.append(position)
                    position += (len(word) + 1)
            else:
                char_positions.append(position)
                position += (len(word) + 1)
            i += 1
        return text, char_positions

    def get_labels(self, label_file, text, char_positions):
        label_text = self.read_file(label_file)
        label_data = self.extract_labelinfo(label_text)
        labels = ['O']*len(char_positions)
        # For all lines with labeldata for word(s)
        for i in range(len(label_data)):
            label = label_data[i][0]
            # For all start positions belonging to the label
            for j in range(1, len(label_data[i])-1, 2):
                start_pos = int(label_data[i][j])
                end_pos = int(label_data[i][j+1])
                # Get index of first word to be labeled
                word_index = char_positions.index(start_pos)
                if j == 1:
                    labels[word_index] = f'B-{label}'
                else:
                    labels[word_index] = f'I-{label}'
                # TODO: Maybe need to check whether i+1 is in range...! What to do if not?
                current_pos = start_pos + char_positions[word_index+1] - char_positions[word_index]
                while current_pos < end_pos:
                    word_index = char_positions.index(current_pos)
                    labels[word_index] = f'I-{label}'
                    current_pos += (char_positions[word_index+1] - char_positions[word_index])
        return labels

    def get_data(self, directory):
        text_dir = os.path.join(directory, 'text')
        label_dir = os.path.join(directory, 'original')

        # TODO: checks for path existence

        text, labels = [], []
        print(f'\nReading messages and constructing word-label pairs...')
        for filename in tqdm(os.listdir(text_dir)):
            if filename != "DICLOFENAC-SODIUM.7.txt":
                text_file = os.path.join(text_dir, filename)
                label_file = os.path.join(label_dir, f'{filename[:-3]}ann')
                try:
                    message_text, message_char_positions = self.get_words_and_positions(text_file)
                except:
                    print(f'Error while parsing TEXT from file: {text_file}')
                    return
                try:
                    message_labels = self.get_labels(label_file, message_text, message_char_positions)
                except:
                    print(f'Error while parsing LABELS from file: {label_file}')
                    return
                text.append(message_text)
                labels.append(message_labels)
        return [text, labels]

    def write_data(self, filename, data):
        # Write words and corresponding labels to txt file in discussed format
        n_messages = len(data[0])
        print(f'\nWriting words and corresponding labels of {n_messages} messages to {filename}...')
        with open(filename, 'w') as f:
            for i in tqdm(range(n_messages)):
                message_txt = data[0][i]
                message_labels = data[1][i]
                for word, label in zip(message_txt, message_labels):
                    f.write(f'{word}\t{label}\n')
                f.write('\n')


p = DataProcessor()
# Try arthrotec 137 and lipitor file 662 / 842

test=0

if test:
    text_file = 'cadec/text/DICLOFENAC-SODIUM.7.txt'
    label_file = 'cadec/original/DICLOFENAC-SODIUM.7.ann'
    txt, chars = p.get_words_and_positions(text_file)

    for i in range(len(txt)):
        print(f'{txt[i]} / {chars[i]}')

    labels = p.get_labels(label_file, txt, chars)

    for i in range(len(txt)):
        print(f'{txt[i]} / {labels[i]}')

else:
    dir = 'cadec'
    writefile = 'testwrite.txt'
    data = p.get_data(dir)
    testdata = [[['a', 'b', 'c'], ['d', 'e']], [['1', '2', '3'], ['4', '5']]]
    p.write_data(writefile, data)

