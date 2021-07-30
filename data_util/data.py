PAD_TOKEN = '[PAD]'
UNKNOWN_TOKEN = '[UNK]'
#START_DECODING = '[START]'
#STOP_DECODING = '[STOP]'
START_DECODING = '[CLS]'
STOP_DECODING = '[SEP]'
class Vocab(object):
    def __init__(self, vocab_file, max_size):
        """
        Args:
            vocab_file: path to the vocab file, which is assumed to contain "<word>" on each line, sorted with most frequent word first.
            max_size: integer. The maximum size of the resulting Vocabulary.
        """
        self.word_to_id = {}
        self.id_to_word = {}
        self.count = 0
        # [PAD], [UKN], [START] and [STOP] get the ids 0,1,2,3.
        for w in [PAD_TOKEN, UNKNOWN_TOKEN, START_DECODING, STOP_DECODING]:
            self.word_to_id[w] = self.count
            self.id_to_word[self.count] = w
            self.count += 1
        # Read the vocab file and add words up to max_size
        with open(vocab_file, 'r') as vocab_f:
            c = 0
            for line in vocab_f:
                c += 1
                w = line.strip('\n')
                #if w in [START_DECODING, STOP_DECODING]:
                #   raise Exception('[UNK], [PAD], [START] and [STOP] shouldn\'t be in the vocab file, but %s is' % w)
                #if w in self.word_to_id:
                #   print(w)
                #   continue
                #   raise Exception('Duplicated word in vocabulary file: %s' % w)
                if w == '\n':
                    continue
                self.word_to_id[w] = self.count
                self.id_to_word[self.count] = w
                self.count += 1
                if max_size != 0 and self.count >= max_size:
                    print("max_size of vocab was specified as %i; we now have %i words. Stopping reading." % (max_size, self.count))
                    break
            #for w in [START_DECODING, STOP_DECODING]:
            #   self.word_to_id[w] = self.count
            #   self.id_to_word[self.count] = w
            #   self.count += 1
        print("Finished constructing vocabulary of %i total words. Last word added: %s" % (self.count, self.id_to_word[self.count-1]))
     
    def word2id(self, word):
        if word not in self.word_to_id:
            return self.word_to_id[UNKNOWN_TOKEN]
        return self.word_to_id[word]

    def id2word(self, word_id):
        if word_id not in self.id_to_word:
            raise ValueError('Id not found in vocab: %d' % word_id)
        return self.id_to_word[word_id]

    def size(self):
        return self.count
    
def outputids2words(id_list, vocab, article_oovs):
    words = []
    for i in id_list:
        try:
            w = vocab.id2word(i) # might be [UNK]
        except ValueError as e: # w is OOV
            assert article_oovs is not None, "Error: model produced a word ID that isn't in the vocabulary. This should not happen in baseline (no pointer-generator) mode"
            article_oov_idx = i - vocab.size()
            try:
                w = article_oovs[article_oov_idx]
            except ValueError as e: # i doesn't correspond to an article oov
                raise ValueError('Error: model produced word ID %i which corresponds to article OOV %i but this example only has %i article OOVs' % (i, article_oov_idx, len(article_oovs)))
        words.append(w)
    return words

    '''
    def abstract2sents(abstract):
        cur = 0
        sents = []
        while True:
            try:
                start_p = abstract.index(SENTENCE_START, cur)
                end_p = abstract.index(SENTENCE_END, start_p + 1)
                cur = end_p + len(SENTENCE_END)
                sents.append(abstract[start_p+len(SENTENCE_START):end_p])
            except ValueError as e: # no more sentences
                return sents
    '''

def show_art_oovs(article, vocab):
    unk_token = vocab.word2id(UNKNOWN_TOKEN)
    oovs = []
    for w in article:
        if vocab.word2id(w)==unk_token:
            oovs.append('__%s__' % w)
        else:
            oovs.append(w)
    out_str = ''.join(words)
    return out_str


def show_abs_oovs(abstract, vocab, article_oovs):
    unk_token = vocab.word2id(UNKNOWN_TOKEN)
    new_words = []
    for w in abstract:
        if vocab.word2id(w) == unk_token: # w is oov
            if article_oovs is None: # baseline mode
                new_words.append("__%s__" % w)
            else: # pointer-generator mode
                if w in article_oovs:
                    new_words.append("__%s__" % w)
                else:
                    new_words.append("!!__%s__!!" % w)
        else: # w is in-vocab word
            new_words.append(w)
    out_str = ''.join(new_words)
    return out_str
