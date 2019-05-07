import numpy as np
import os
from tqdm import tqdm
import pandas as pd
import json
from mrcnn.vocabulary import Vocabulary
from nltk.tokenize import word_tokenize
data = np.load('train.npy')
vocabulary = Vocabulary(5000,'vocabulary.csv')
caps = []
train_update=[]
for dd in  data:

    word = dd['captionw']
    word_idxs = []
    masks = []
    for w in word:
        current_word_idxs_ = vocabulary.process_sentence(w)
        current_num_words = len(current_word_idxs_)
        current_word_idxs = np.zeros(15,dtype=np.int32)
        current_masks = np.zeros(15)
        current_word_idxs[:current_num_words] = np.array(current_word_idxs_)
        current_masks[:current_num_words] = 1.0
        word_idxs.append(current_word_idxs)
        masks.append(current_masks)
    word_idxs = np.array([word_idxs])
    word_masks = np.array([masks])
    dd['caption']=word_idxs
    dd['caption_mask']=word_masks
    train_update.append(dd)
np.save('./data/train_update.npy',train_update)

