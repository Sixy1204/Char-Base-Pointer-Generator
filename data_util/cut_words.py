
import os
from tqdm import tqdm
import numpy as np
from ckipnlp.driver.ss import CkipSentenceSegmenter as SentSeg
# Make sentence segmentation obj (data structure that ckip tagger use)
from ckiptagger import construct_dictionary, WS, POS, NER
# In[28]:


# init pipeline
ss = SentSeg(delims = {",", "。", ":", "?", "!", ";"})
ws = WS("../ckip", disable_cuda=False)
pos = POS("../ckip", disable_cuda=False)
ner = NER("../ckip", disable_cuda=False)

#NER_LABLE = ['GPE', 'PERSON', 'DATE', 'EVENT']


# In[34]:


data_pth = '../data/original_data'
#train = np.load(os.path.join(data_pth, 'train.npy'), allow_pickle=True)
test = np.load(os.path.join(data_pth, 'test.npy'), allow_pickle=True)


# In[31]:


def get_token_words(ws_lst):
    all_token = [' '.join(ws_sent) for ws_sent in ws_lst]
    all_sent = ' '.join(all_token).strip()
    return all_sent


# In[41]:


def tokenizer(dataset):
    for i, data in tqdm(enumerate(dataset)):
        art_seg = ss(raw = data['content'])
        art_ws_lst = ws(art_seg)
        
        title_seg = ss(raw = data['title'])
        title_ws_lst = ws(title_seg)
        
        art_pos_lst = pos(art_ws_lst)
        ner_lst = ner(art_ws_lst, art_pos_lst)
        ner_set = []
        
        dataset[i]['token_content'] = get_token_words(art_ws_lst)
        dataset[i]['token_title'] = get_token_words(title_ws_lst)
        
        for j, sentence in enumerate(art_seg):
            for entity in sorted(ner_lst[j]):
                if entity not in ner_set:
                    ner_set.append(entity[-2:])
                
                '''
                if entity[2] in NER_LABLE:
                    ner_w = entity[-2:]
                    if ner_w not in ner_set:
                        ner_set.append(entity[-2:])
                '''        

        dataset[i]['ner'] = ner_set
        #dataset[i]['ner_tgt'] = ner_tgt
        
        
        if i % 10000 == 0:
            print('Run process {:.2%}'.format(i/len(dataset)) )
        
        #dataset[i]['ner_tgt'] = ' '.join([x for x in dataset[i]['ner_tgt']]).strip()
        del art_seg, art_ws_lst, title_seg, title_ws_lst, art_pos_lst, ner_lst, ner_set
        #del art_seg, art_ws_lst, title_seg, title_ws_lst
    return dataset


# In[ ]:

#print('Process train')
#train_token = tokenizer(train)

print('Process test')
test_token = tokenizer(test)

# In[ ]:


#np.save('../data/train_all_token.npy', train_token)
np.save('../data/token_all/test_all_token.npy', test_token)


