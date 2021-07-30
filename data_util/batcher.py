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
    BATCH_QUEUE_MAX = 50
    # vocab should be an instance    
    def __init__(self, data_path, vocab, mode, batch_size, single_pass):
        self._data_path = data_path
        self._vocab = vocab
        self._single_pass = single_pass
        self.mode = mode
        self.batch_size = batch_size
        self._data_set = np.load(self._data_path, allow_pickle=True).tolist()
        # Initialize a queue of Batches waiting to be used, and a queue of Examples waiting to be batched
        self._batch_queue = queue.Queue(self.BATCH_QUEUE_MAX)
        self._example_queue = queue.Queue(self.BATCH_QUEUE_MAX * self.batch_size)
        
        # Different settings depending on whether we're in single_pass mode or not
        if single_pass:
            self._num_example_q_threads = 1 # just one thread, so we read through the dataset just once
            self._num_batch_q_threads = 1  # just one thread to batch examples
            self._bucketing_cache_size = 1 # only load one batch's worth of examples before bucketing; this essentially means no bucketing
            self._finished_reading = False # this will tell us when we're finished reading the dataset
        else:
            self._num_example_q_threads = 1 #16 # num threads to fill example queue
            self._num_batch_q_threads = 1 #4  # num threads to fill batch queue
            self._bucketing_cache_size = 1 #100 # how many batches-worth of examples to load into cache before bucketing

        # Start the threads that load the queues
        self._example_q_threads = []
        for _ in range(self._num_example_q_threads):
            self._example_q_threads.append(Thread(target=self.fill_example_queue))
            self._example_q_threads[-1].daemon = True
            self._example_q_threads[-1].start()
            self._batch_q_threads = []
    
        for _ in range(self._num_batch_q_threads):
            self._batch_q_threads.append(Thread(target=self.fill_batch_queue))
            self._batch_q_threads[-1].daemon = True
            self._batch_q_threads[-1].start()

        # Start a thread that watches the other threads and restarts them if they're dead
        if not single_pass: # We don't want a watcher in single_pass mode because the threads shouldn't run forever
            self._watch_thread = Thread(target=self.watch_threads)
            self._watch_thread.daemon = True
            self._watch_thread.start()

    def next_batch(self):
    # If the batch queue is empty, print a warning
        if self._batch_queue.qsize() == 0:
            logging.warning('Bucket input queue is empty when calling next_batch. Bucket queue size: %i, Input queue size: %i', self._batch_queue.qsize(), self._example_queue.qsize())
            if self._single_pass and self._finished_reading:
                logging.info("Finished reading dataset in single_pass mode.")
                return None

        batch = self._batch_queue.get() # get the next Batch
        return batch

    def fill_example_queue(self):
        input_gen = self.text_generator(self.example_generator())

        while True:
            try:
                (article, title) = input_gen.__next__() # read the next example from file. article and abstract are both strings.
           
            except: # if there are no more examples:
                logging.info("The example generator for this example queue filling thread has exhausted data.")
                if self._single_pass:
                    logging.info("single_pass mode is on, so we've finished reading dataset. This thread is stopping.")
                    self._finished_reading = True
                    break
                else:
                    raise Exception("single_pass mode is off but the example generator is out of data; error.")

            example = Example(article, title, self._vocab) # Process into an Example.
            self._example_queue.put(example) # place the Example in the example queue.
            """ 
            time.sleep(2) 
            print("*****Example*****")
            print("="*100)
            print("Article", example.original_article, type(example.original_article))
            print("-"*80)
            print("enc_input:", example.enc_input)
            print("-"*80)
            print("Abstract Words", example.original_title, type(example.original_title))
            print("-"*80)
            print("dec_input:", example.dec_input, len(example.dec_input))
            print("-"*80)
            print("Target:", example.target, len(example.target))
            print("="*100)
            time.sleep(2) 
            """

    def fill_batch_queue(self):
        while True:
            if self.mode == 'decode':
            # beam search decode mode single example repeated in the batch
                ex = self._example_queue.get()
                b = [ex for _ in range(self.batch_size)]
                self._batch_queue.put(Batch(b, self._vocab, self.batch_size))
            else:
            # Get bucketing_cache_size-many batches of Examples into a list, then sort
                inputs = []
                for _ in range(self.batch_size * self._bucketing_cache_size):
                    inputs.append(self._example_queue.get())
                inputs = sorted(inputs, key=lambda inp: inp.enc_len, reverse=True) # sort by length of encoder sequence

                # Group the sorted Examples into batches, optionally shuffle the batches, and place in the batch queue.
                batches = []
                for i in range(0, len(inputs), self.batch_size):
                    batches.append(inputs[i:i + self.batch_size])

                if not self._single_pass:
                    shuffle(batches)
                
                for b in batches:  # each b is a list of Example objects
                    self._batch_queue.put(Batch(b, self._vocab, self.batch_size))

    def watch_threads(self):
        while True:
            logging.info('Bucket queue size: %i, Input queue size: %i',
            self._batch_queue.qsize(), self._example_queue.qsize())

            time.sleep(60)
            for idx,t in enumerate(self._example_q_threads):
                if not t.is_alive(): # if the thread is dead
                    logging.error('Found example queue thread dead. Restarting.')
                    new_t = Thread(target=self.fill_example_queue)
                    self._example_q_threads[idx] = new_t
                    new_t.daemon = True
                    new_t.start()
                    
            for idx,t in enumerate(self._batch_q_threads):
                if not t.is_alive(): # if the thread is dead
                    logging.error('Found batch queue thread dead. Restarting.')
                    new_t = Thread(target=self.fill_batch_queue)
                    self._batch_q_threads[idx] = new_t
                    new_t.daemon = True
                    new_t.start()
    
    def example_generator(self):
        while True:
            if self._single_pass:
                data_set = self._data_set

            else:
                data_set = random.sample(self._data_set, len(self._data_set))
               
            for data in data_set:
                '''
                if config.is_tokenized:
                    if config.is_ner:
                        ner = ' '.join([x[1] for x in data['ner']])
                        yield(data['token_content'], ner)
                    
                    else:
                        yield(data['token_content'], data['token_title'])
                '''
                if config.is_ner:
                    ner = ''.join([x[1] for x in data['ner']])
                    yield(data['content'], ner)
                else:
                    yield(data['content'], data['title'])
            
            if self._single_pass:
                break
   
    def text_generator(self, example_generator):
        while True:
            e = example_generator.__next__()   
            try:
                article, title = e
        
            except ValueError:
                logging.error('Failed to get article or abstract from example')
                continue

            if len(article) == 0 or len(title)==0:
                continue
        
            else:
                yield (article, title)
