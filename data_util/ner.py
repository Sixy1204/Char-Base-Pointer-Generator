from tqdm import tqdm
import numpy as np
from ckipnlp.driver.ss import CkipSentenceSegmenter as SentSeg
# Make sentence segmentation obj (data structure that ckip tagger use)

from ckiptagger import construct_dictionary, WS, POS, NER

# init pipeline
ss = SentSeg(delims = {",", "。", ":", "?", "!", ";"})
ws = WS("../ckip", disable_cuda=False)
pos = POS("../ckip", disable_cuda=False)
ner = NER("../ckip", disable_cuda=False)

train_chunk = np.load('../data/test_chunk.npy', allow_pickle=True)
NER_LABLE = ['GPE', 'PERSON', 'DATE', 'EVENT']

for i, data in tqdm(enumerate(train_chunk)):
    sent_seg = ss(raw = data['content'])
    ws_lst = ws(sent_seg)
    pos_lst = pos(ws_lst)
    ner_lst = ner(ws_lst, pos_lst)
    ner_set = []
    for j, sentence in enumerate(sent_seg):
        for entity in sorted(ner_lst[j]):
            if entity[2] in NER_LABLE:
                ner_w = entity[-2:]
                if ner_w not in ner_set:
                    ner_set.append(entity[-2:])
    ner_tgt = ''.join([x[1] for x in ner_set])
    train_chunk[i]['ner'] = ner_set
    train_chunk[i]['ner_tgt'] = ner_tgt
    del sent_seg, ws_lst, pos_lst, ner_lst, ner_set

np.save('../data/test_ner.npy', train_chunk)
