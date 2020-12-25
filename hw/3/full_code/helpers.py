import numpy as np
import unidecode
import string
import random
import time
import math
import torch


all_characters = string.printable
n_characters = len(all_characters)

def read_file(filename):
    file = unidecode.unidecode(open(filename).read())
    '''vocab = set(file)
    vocab_to_int = {c:i for i,c in enumerate(vocab)}
    int_to_vocab = dict(enumerate(vocab))
    file = np.array([vocab_to_int[c] for c in file],dtype=np.int32)'''
    return file, len(file)

# Turning a string into a tensor

def char_tensor(string):
    tensor = torch.zeros(len(string)).long()
    for c in range(len(string)):
        try:
            tensor[c] = all_characters.index(string[c])
        except:
            continue
    return tensor

# Readable time elapsed

def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

