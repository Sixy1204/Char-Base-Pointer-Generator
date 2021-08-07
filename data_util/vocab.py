import numpy as np
import os
import random
from collections import Counter
import config

VOCAB_SIZE = 200000

word_counter = Counter()
root = "../data/original_data/"
data_path = os.path.join(root, "train.npy")
vocab_path = os.path.join(root, "vocab")

training_data = np.load(data_path, allow_pickle=True)
for news in training_data:
    title = [t for t in news['title'] if t!=" " and t!=""]
    art = [t for t in news['content'] if t!=" " and t!=""]
    word_counter.update(title)
    word_counter.update(art)

with open(vocab_path, 'w', encoding='utf-8') as writer:
    for word, _ in word_counter.most_common(VOCAB_SIZE):
        writer.write(word+'\n')
