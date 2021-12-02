import os
import re
import numpy as np
import spacy
from tqdm import tqdm

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

    
    def get_initial_charpositions(self, text):
        current_pos = 0
        for word in text:
            char_positions.append(current_pos)
            current_pos += (word + 1)
        return char_positions
    
    def check_email(self, word):
        # Check whether both @ and . occur in word
        return False

    def check_website_or_abbr(self, word):
        # Check whether '.' occurs (at least) twice
        # To catch websites and abbreviations, e.g. i.e.
        return False

    def separate_first_char(self, word):
        # Separate first character from rest of string
        return [word]

    def separate_last_char(self, word):
        # Separate last character from rest of string
        return [word]

    def separate_word(self, word):
        # Check for apostrophe: special case
        return [word]

    def create_subwords(self, word):
        alnum = re.search("[0-9a-zA-Z]+", word)
        # Word contains only non-alphanumeric chars: don't separate
        if alnum is None:
            return word
        
        non_alnum = re.search("[^0-9a-zA-Z]+", word)
        pos = non_alnum.start()
        # First char to be separated
        if pos == 0:
            return(self.separate_first_char(word))
        # Last char to be separated
        elif pos == len(word)-1:
            return(self.separate_last_char(word))
        elif self.check_email(word):
            return [word]
        elif self.check_website_or_abbr(word):
            return [word]
        else:
            return(self.separate_word(word))

    def get_text_and_positions(self, text_file):
        text = self.read_file(text_file)
        text = text.replace('\n', ' ')
        text = text.split(' ')[:-1] # Last character will be '', discarced here
        
        char_positions = []
        # char_positions = self.get_initial_charpositions(text)

        i = 0
        while i<len(text):
            word = text[i]
            if not word.isalnum():
                subwords = self.create_subwords(word)
                # Only need to update when word was separated into subwords
                if len(subwords) > 1:
                    j = i
                    del text[i]
                    for subword in subwords:
                        text.insert(j, subword)
                        j += 1
                    # TODO: function to update char_positions
                    # Important to take into account for where there are spaces etc.
                    # So that char positions in labelfile correspond to char_positions
            i += 1
        return text, char_positions

    # TODO: OLD VERSION. Need to update for new parsing probably
    def get_labels(self, label_file, text, char_positions):
        label_text = self.read_file(label_file)
        label_data = self.extract_labelinfo(label_text)
        labels = ['O']*len(char_positions)

        for i in range(len(label_data)):
            label = label_data[i][0]
            for j in range(1, len(label_data[i])-1, 2):
                start_pos = int(label_data[i][j])
                end_pos = int(label_data[i][j+1])
                word_index = char_positions.index(start_pos) # Gives index of first word
                if j == 1:
                    labels[word_index] = f'B-{label}'
                else:
                    labels[word_index] = f'I-{label}'
                current_pos = start_pos + len(text[word_index]) + 1
                while current_pos < end_pos:
                    word_index = char_positions.index(current_pos)
                    labels[word_index] = f'I-{label}'
                    current_pos += (len(text[word_index]) + 1)
        return labels

    # TODO: OLD VERSION. Needs update
    def get_data(self, directory):
        text_dir = os.path.join(directory, 'text')
        label_dir = os.path.join(directory, 'original')

        # TODO: checks for path existence

        text, labels = [], []
        for filename in tqdm(os.listdir(text_dir)):
            text_file = os.path.join(text_dir, filename)
            label_file = os.path.join(label_dir, f'{filename[:-3]}ann')
            message_text, message_char_positions = self.get_text_and_positions(text_file)
            if message_text == 1:
                print(filename)
            # message_labels = self.get_labels(label_file, message_text, message_char_positions)
            text.append(message_text)
            # labels.append(message_labels)
        return [text, labels]

    def write_data(self, text, labels):
        # Write words and corresponding labels to txt file in discussed format
        pass


p = DataProcessor()

# text_file = 'data/text/ARTHROTEC.137.txt'
# label_file = 'data/original/ARTHROTEC.137.ann'

text_file = 'test.txt'
text, char_positions = p.get_text_and_positions(text_file)
print(text)
