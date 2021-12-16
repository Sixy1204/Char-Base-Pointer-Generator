import queue
import time
from random import shuffle
from threading import Thread

import numpy as np
import logging

import config
import data

import random 
random.seed(1234)


class Example(object):
    def __init__(self, article, title, vocab):
        bos_id = vocab.word2id(data.START_DECODING)
        eos_id = vocab.word2id(data.STOP_DECODING)
        '''
        ### prepare tokenized text ###        
        if config.is_tokenized:
            article_words = article.split(" ")
            article_words = [t for t in article_words if t != " "]
            
            title_words = title.split(" ")
            title_words = [t for t in title_words if t != " "]
        '''
        ### unigram ###
        article_words = [t for t in article.strip() if t!=" "]
        title_words = [t for t in title.strip() if t!=" "]
        
        # article w2id
        if len(article_words) > config.max_enc_steps:
            article_words = article_words[:config.max_enc_steps]

        self.enc_input = []
        for w in article_words:
            self.enc_input.append(vocab.word2id(w))
             
        self.enc_len = len(self.enc_input)

        # title w2id
        title_ids = [vocab.word2id(w) for w in title_words]
        self.dec_input, self.target = self.get_dec_input_target(title_ids, config.max_dec_steps, bos_id, eos_id)
        self.dec_len = len(self.dec_input)
        
        #TODO if pointer_gen=True
        if config.pointer_gen:
            self.enc_input_extend_vocab, self.article_oovs = self.article2ids(article_words, vocab)
            title_ids_extend_vocab = self.title2ids(title_words, vocab, self.article_oovs)
            _, self.target = self.get_dec_input_target(title_ids_extend_vocab, config.max_dec_steps, bos_id, eos_id)
        
        self.original_article = article
        self.original_title = title

    # dec_input has bos and target has eos
    def get_dec_input_target(self, seq, max_len, bos_id, eos_id):
        inp = [bos_id] + seq[:]
        target = seq[:]
        if len(inp) > max_len:
            inp = inp[:max_len]
            target = target[:max_len]
        else:
            target.append(eos_id)
        assert len(inp) == len(target)
        return inp, target

    def article2ids(self, article_words, vocab):
        ids, oovs = [], []
        unk_id = vocab.word2id(data.UNKNOWN_TOKEN)
        for w in article_words:
            i = vocab.word2id(w)
            if i == unk_id:
                if w not in oovs:
                    oovs.append(w)
                oov_num = oovs.index(w)
                ids.append(vocab.size() + oov_num)
            else:
                ids.append(i)
        return ids, oovs

    def title2ids(self, title_words, vocab, article_oovs):
        ids = []
        unk_id = vocab.word2id(data.UNKNOWN_TOKEN)
        for w in title_words:
            i = vocab.word2id(w)
            if i == unk_id:
                if w in article_oovs:
                    ids.append(vocab.size() + article_oovs.index(w))
                else:
                    ids.append(unk_id)
            else:
                ids.append(i)
        return ids

    def pad_decoder_inp_targ(self, max_len, pad_id):
        while len(self.dec_input) < max_len:
            self.dec_input.append(pad_id)
        while len(self.target) < max_len:
            self.target.append(pad_id)


    def pad_encoder_input(self, max_len, pad_id):
        while len(self.enc_input) < max_len:
            self.enc_input.append(pad_id)
        if config.pointer_gen:
            while len(self.enc_input_extend_vocab) < max_len:
                self.enc_input_extend_vocab.append(pad_id)

class Batch(object):
    def __init__(self, example_list, vocab, batch_size):
        self.vocab = vocab
        self.batch_size = batch_size
        self.pad_id = vocab.word2id(data.PAD_TOKEN)
        self.init_encoder_seq(example_list)
        self.init_decoder_seq(example_list)
        self.store_orig_strings(example_list)

    def init_encoder_seq(self, example_list):
        max_enc_seq_len = max([ex.enc_len for ex in example_list])
        for ex in example_list:
            ex.pad_encoder_input(max_enc_seq_len, self.pad_id)
          
        self.enc_batch = np.zeros((self.batch_size, max_enc_seq_len), dtype=np.int32)
        self.enc_lens = np.zeros((self.batch_size), dtype=np.int32)
        self.enc_padding_mask = np.zeros((self.batch_size, max_enc_seq_len), dtype=np.int32)
    
        # fill in the np arrays
        for i, ex in enumerate(example_list):
            self.enc_batch[i, :] = ex.enc_input[:]
            self.enc_lens[i] = ex.enc_len
            for j in range(ex.enc_len):
                self.enc_padding_mask[i][j] = 1
        
        if config.pointer_gen:
            # Determine the max number of in-article OOVs in this batch
            self.max_art_oovs = max([len(ex.article_oovs) for ex in example_list])
            # Store the in-article OOVs themselves
            self.art_oovs = [ex.article_oovs for ex in example_list]
            self.enc_batch_extend_vocab = np.zeros((self.batch_size, max_enc_seq_len), dtype=np.int32)
            for i, ex in enumerate(example_list):
                self.enc_batch_extend_vocab[i, :] = ex.enc_input_extend_vocab[:]            
        

    def init_decoder_seq(self, example_list):
        for ex in example_list:
            ex.pad_decoder_inp_targ(config.max_dec_steps, self.pad_id)
            
        self.dec_batch = np.zeros((self.batch_size, config.max_dec_steps), dtype=np.int32)
        self.target_batch = np.zeros((self.batch_size, config.max_dec_steps), dtype=np.int32)
        self.dec_padding_mask = np.zeros((config.batch_size, config.max_dec_steps), dtype=np.float32)
        self.dec_lens = np.zeros((self.batch_size), dtype=np.int32)
        
        # fill in the np arrays
        for i, ex in enumerate(example_list):
            self.dec_batch[i, :] = ex.dec_input[:]
            self.target_batch[i, :] = ex.target[:]
            self.dec_lens[i] = ex.dec_len
            for j in range(ex.dec_len):
                self.dec_padding_mask[i][j] = 1
            

    def store_orig_strings(self, example_list):
        self.original_articles = [ex.original_article for ex in example_list]
        self.original_titles = [ex.original_title for ex in example_list]


class Batcher(object):
    def __init__(self, data_path, vocab, mode, batch_size, single_pass):
        self.data_path = data_path
        self.vocab = vocab
        self.mode = mode 
        self.batch_size = batch_size
        self.single_pass = single_pass
        
        self.news_list = np.load(self.data_path, allow_pickle=True)
        self.count = len(self.news_list)
        
        self.all_ex_list = [Example(self.news_list[i]['content'], self.news_list[i]['title'], self.vocab) for i in range(self.count)]
                   
         
        if self.single_pass:
            self.cur = 0
            self.val_cur = 0
        
      
    def next_batch(self):
        # Testing phase
        if self.mode == 'decode':
            if self.single_pass:
                if self.cur < self.count:
                    cur_news = self.news_list[self.cur]
                    ex = Example(cur_news['content'], cur_news['title'], self.vocab)  
                    example_list = [ex for _ in range(self.batch_size)]
                    self.cur += 1
                    return Batch(example_list, self.vocab, self.batch_size)
                else:
                    return None
        
        # Training phase
        # TODO 12/15 unsort example
        else:
            #shuffle(self.all_ex_list)
            if self.single_pass:
                if self.val_cur < self.count:
                    example_list = random.sample(self.all_ex_list, self.batch_size)
                    #example_list = sorted(example_list, key=lambda ex: ex.enc_len, reverse=True)
                    self.val_cur += self.batch_size
                    return Batch(example_list, self.vocab, self.batch_size)
                else:
                    del self.val_cur
                    self.val_cur = 0
                    return None
            else:
                example_list = random.sample(self.all_ex_list, self.batch_size)
                #example_list = sorted(example_list, key=lambda ex: ex.enc_len, reverse=True)
                return Batch(example_list, self.vocab, self.batch_size)
        
