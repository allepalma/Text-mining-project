import os
import re
import numpy as np
from tqdm import tqdm

class DatasetCreator():
    def __init__(self, max_length=512, split_messages=True, doublecheck_labels=False, discard=["DICLOFENAC-SODIUM.7.txt"]):
        self.max_length = max_length
        self.split_messages = split_messages
        self.doublecheck_labels = doublecheck_labels
        self.discarded_files = [file for file in discard]

    def create_dataset(self, dir='cadec', writefile='dataset'):
        '''
        Create dataset and write to file
        '''
        data = self.get_data(dir)
        self.write_data(writefile, data)

    def read_file(self, file):
        '''
        Read file
        '''
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
        words_to_label = []
        for info in label_text:
            if info[0][0] == 'T':
                # Extract all start and end char positions, separated by ' ' and ';'
                label_data.append(re.split(' |;', info[1]))
                # Save words as denoted in labelfile for doublecheck
                words_to_label.append(re.split(' ', info[2]))
        return label_data, words_to_label

    def check_email(self, word):
        '''
        Check whether both @ and . occur in word
        '''
        if '@' in word and '.' in word:
            return True
        return False

    def check_multiple_dots(self, word, char):
        '''
        Check whether '.' occurs (at least) twice
        To catch websites and abbreviations, e.g. i.e.
        '''
        if word.count(char) > 1:
            return True
        return False

    def separate_nonalnumgroup(self, word, pos, end_pos):
        '''
        Separate group of neighbouring non-alphanumeric characters
        into subwords
        '''
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
        '''
        Separate first character from rest of string
        '''
        subword1 = word[0]
        subword2 = word[1:]
        return [subword1, subword2]

    def separate_last_char(self, word):
        '''
        Separate last character from rest of string
        '''
        subword1 = word[:-1]
        subword2 = word[-1]
        return [subword1, subword2]

    def separate_apostrophe_char(self, word, pos):
        '''
        Handle apostrophes specifically
        n't: haven't -> have n't. But can't wil become: ca, n't
        've: you've -> you 've
        I'm: I'm -> I 'm
        's: he's -> he 's
        '''
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
        '''
        Separate word into three subwords,
        i.e. char(s) before pos, char at pos, char(s) after pos
        '''
        subwords = [word[:pos], word[pos], word[pos+1:]]
        return subwords

    def create_subwords(self, word):
        '''
        Check if word contains a non-alphanumeric character,
        and then whether it needs to be separated into subwords
        '''
        non_alnum = re.search("[^0-9a-zA-Z]+", word)
        pos = non_alnum.start()
        end_pos = non_alnum.end()
        char = non_alnum.group(0)
        skip_next_word = False
        # Neighbouring non-alnums
        if end_pos-pos > 1:
            subwords, skip = self.separate_nonalnumgroup(word, pos, end_pos)
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
        # Check if word is email address
        elif self.check_email(word):
            subwords = [word]
        # Check if word contains multiple dots
        elif char == '.' and self.check_multiple_dots(word, char):
            subwords = [word]
        # In all other cases: separate non-alnum from the rest
        else:
            subwords = self.separate_word(word, pos)
            # Skip next always true, because always single non-alnum within word
            skip_next_word = True
        return subwords, skip_next_word

    def split_message(self, text, labels):
        '''
        Recursively split a message that is longer than max_length,
        on basis of separation of sentences with a '.'
        Find the '.' that is closest to the middle of the message,
        split message there.
        return: splitted messages and corresponding labels
        '''
        length = len(text)
        half_length = length/2
        eos_indexes = []
        for i in range(len(text)):
            if text[i].endswith('.'):
                eos_indexes.append(i)
        # Check if at least one '.' was found to split the message into sentences
        if not eos_indexes:
            print(f'Warning: no "." was found to split message longer than max_length')
            return [text, labels]
        # Find end-of-sequence indicator (.) that is closest to middle of message
        eos_np = np.asarray(eos_indexes)
        idx = (np.abs(eos_np - half_length)).argmin()
        split_i = eos_np[idx]

        text_submessages, labels_submessages = [], []
        text_1 = text[:split_i+1]
        labels_1 = labels[:split_i+1]
        text_2 = text[split_i+1:]
        labels_2 = labels[split_i+1:]
        if len(text_1) > self.max_length:
            split_text_1, split_labels_1 = self.split_message(text_1, labels_1)
            for i in range(len(split_text_1)):
                text_submessages.append(split_text_1[i])
                labels_submessages.append(split_labels_1[i])
        else:
            text_submessages.append(text_1)
            labels_submessages.append(labels_1)
        if len(text_2) > self.max_length:
            split_text_2, split_labels_2 = self.split_message(text_2, labels_2)
            for i in range(len(split_text_2)):
                text_submessages.append(split_text_2[i])
                labels_submessages.append(split_labels_2[i])
        else:
            text_submessages.append(text_2)
            labels_submessages.append(labels_2)
        return [text_submessages, labels_submessages]

    def double_check(self, i, text, word_index, words_to_label, to_label_index, file):
        '''
        Double check if words in text with special label correspond
        to words in the labelfile (saved in [words_to_label]).
        Create subwords where necessary for words in labelfile and 
        check if they are the same words as in text
            (this is not exactly identical to how subwords are created for [text],
            therefore print mismatches to also check manually)
        '''
        word = text[word_index]
        check_word = words_to_label[i][to_label_index]
        if word != check_word:
            sub_check_words, skip_next = self.create_subwords(check_word)
            if len(sub_check_words) > 2 and skip_next == False:
                pass # TODO
            del words_to_label[i][to_label_index]
            new_index = to_label_index
            for new_word in sub_check_words:
                try:
                    words_to_label[i].insert(new_index, new_word)
                except:
                    words_to_label[i].append(new_word)
                new_index += 1
        if word != words_to_label[i][to_label_index]:
            print(f'Error: word to label in text doesn\'t correspond to word in labelfile:')
            print(f'Text: {text[word_index]}\tlabelfile: {words_to_label[i][to_label_index]}')
            print(f'Filename: {file}\n')
        return words_to_label

    def get_words_and_positions(self, text_file):
        '''
        Read text from textfile,
        replace enters by spaces and split on spaces.
        Iterate over words to check for non-alhpanumeric characters.
        If this is the case, (potentially) separate the words into subwords,
        insert the subwords into [text],
        correctly save the position of the first character of each word
        into [char_positions]
        '''
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
        '''
        For all words to be labeled as denoted in label_file:
            find their index in [text] on basis of their starting character
            (which is stored in [label_data]),
            store the label in [labels] at the same index
        Potentially do the double check
        '''
        label_text = self.read_file(label_file)
        label_data, words_to_label = self.extract_labelinfo(label_text)
        labels = ['O']*len(char_positions)
        # For all lines with labeldata for word(s)
        for i in range(len(label_data)):
            label = label_data[i][0]
            to_label_index = 0
            # For all start positions belonging to the label
            for j in range(1, len(label_data[i])-1, 2):
                start_pos = int(label_data[i][j])
                end_pos = int(label_data[i][j+1])
                # Get index of first word to be labeled
                word_index = char_positions.index(start_pos)
                if j == 1:
                    labels[word_index] = f'B-{label}'
                    if self.doublecheck_labels:
                        words_to_label = self.double_check(i, text, word_index, words_to_label, to_label_index, label_file)
                    to_label_index += 1
                else:
                    labels[word_index] = f'I-{label}'
                    if self.doublecheck_labels:
                        words_to_label = self.double_check(i, text, word_index, words_to_label, to_label_index, label_file)
                    to_label_index += 1
                # TODO: Maybe need to check whether i+1 is in range...! What to do if not?
                current_pos = start_pos + char_positions[word_index+1] - char_positions[word_index]
                # Iterate over all words that have this label
                while current_pos < end_pos:
                    word_index = char_positions.index(current_pos)
                    labels[word_index] = f'I-{label}'
                    if self.doublecheck_labels:
                        words_to_label = self.double_check(i, text, word_index, words_to_label, to_label_index, label_file)
                    to_label_index += 1
                    current_pos += (char_positions[word_index+1] - char_positions[word_index])
        return labels

    def get_data(self, directory):
        '''
        Extract text and corresponding labels from all files
        in specified directory
        '''
        text_dir = os.path.join(directory, 'text')
        label_dir = os.path.join(directory, 'original')
        if not os.path.isdir(text_dir):
            print(f'\nError: directory to textfiles does not exist.')
            return [[],[]]
        if not os.path.isdir(label_dir):
            print(f'\nError: directory to labelfiles does not exist.')
            return [[],[]]

        text, labels = [], []
        print(f'\nReading messages and constructing word-label pairs...')
        for filename in tqdm(os.listdir(text_dir)):
            if filename not in self.discarded_files:
                text_file = os.path.join(text_dir, filename)
                label_file = os.path.join(label_dir, f'{filename[:-3]}ann')
                try:
                    message_text, message_char_positions = self.get_words_and_positions(text_file)
                except:
                    print(f'Error while parsing test from file: {text_file}')
                    return
                try:
                    message_labels = self.get_labels(label_file, message_text, message_char_positions)
                except:
                    print(f'Error while parsing labels from file: {label_file}')
                    return
                if self.split_messages and len(message_text) > self.max_length:
                    message_text, message_labels = self.split_message(message_text, message_labels)
                    for i in range(len(message_text)):
                        text.append(message_text[i])
                        labels.append(message_labels[i])
                else:
                    text.append(message_text)
                    labels.append(message_labels)
        return [text, labels]

    def write_data(self, filename, data):
        '''
        Write words and corresponding labels to txt file
        '''
        n_messages = len(data[0])
        print(f'\nWriting words and corresponding labels of {n_messages} messages to {filename}...')
        print(f'\t(Discarded files: {self.discarded_files})')
        with open(filename, 'w') as f:
            for i in tqdm(range(n_messages)):
                message_txt = data[0][i]
                message_labels = data[1][i]
                for i, (word, label) in enumerate(zip(message_txt, message_labels)):
                    if i==len(message_txt)-1:
                        f.write(f'{word}\t{label}\n\n')
                    else:
                        f.write(f'{word}\t{label}\n')
