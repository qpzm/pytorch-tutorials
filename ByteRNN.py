from __future__ import unicode_literals, print_function, division
from io import open
import os
import glob
import unicodedata
import string


def findFiles(path):
    return glob.glob(path)


print(findFiles('/data1/qpzm/data-char-rnn/names/*.txt'))

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)
print(all_letters)


# Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )


print(unicodeToAscii('Ślusàrski'))

category_lines = {}
all_categories = []

def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]

for filename in findFiles('/data1/qpzm/data-char-rnn/names/*.txt'):
    category = os.path.splitext(os.path.basename(filename))[0]
    print(os.path.splitext(os.path.basename(filename)))
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines

n_categories = len(all_categories)
