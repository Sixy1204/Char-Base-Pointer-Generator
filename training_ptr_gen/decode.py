"""
decode阶段使用 beam search 算法
"""
import os
import sys
import time
import torch
from torch.autograd import Variable
sys.path.append('../')
sys.path.append('../data_util')

import data
import config
from data import Vocab
from model import Model
#from config import USE_CUDA, DEVICE
from batcher import Batcher
from train_util import get_input_from_batch
#from utils import write_for_rouge, rouge_eval, rouge_log

from rouge import Rouge
import json
use_cuda = config.use_gpu and torch.cuda.is_available()


class Beam(object):
    def __init__(self, tokens, log_probs, state, context, coverage):
        self.tokens = tokens
        self.log_probs = log_probs
        self.state = state
        self.context = context
        self.coverage = coverage

    def extend(self, token, log_prob, state, context, coverage):
        return Beam(tokens = self.tokens + [token],
                            log_probs = self.log_probs + [log_prob],
                            state = state,
                            context = context,
                            coverage = coverage)

    @property
    def latest_token(self):
        return self.tokens[-1]

    @property
    def avg_log_prob(self):
        return sum(self.log_probs) / len(self.tokens)


class BeamSearch(object):
    def __init__(self, model_file_path):
        model_name = os.path.basename(model_file_path)
        self._decode_dir = os.path.join(config.log_root, 'decode_%s' % \
                            (model_name))
        
        self._rouge_art_dir = os.path.join(self._decode_dir, 'rouge_art')
        self._rouge_ref_dir = os.path.join(self._decode_dir, 'rouge_ref')
        self._rouge_dec_dir = os.path.join(self._decode_dir, 'rouge_dec_dir')
        self._result_txt = os.path.join(self._decode_dir, 'ref_dec.txt') 
        
        # 创建3个目录
        for p in [self._decode_dir, self._rouge_art_dir, self._rouge_ref_dir, self._rouge_dec_dir]:
            if not os.path.exists(p):
                os.mkdir(p)
        """
        if config.is_tokenized:
            self.vocab = Vocab(config.vocab_token_path, config.vocab_size)
            self.batcher = Batcher(config.decode_token_path, self.vocab, mode='decode',
                               batch_size=config.beam_size, single_pass=True)
        """
        
        self.vocab = Vocab(config.vocab_path, config.vocab_size)
        self.batcher = Batcher(config.decode_data_path, self.vocab, mode='decode',
                               batch_size=config.beam_size, single_pass=True)
        time.sleep(5)

        self.model = Model(model_file_path, is_eval=True)

    def sort_beams(self, beams):
        return sorted(beams, key=lambda h: h.avg_log_prob, reverse=True)


    def decode(self):
        start = time.time()
        rouge = Rouge()
        score_json = {}
        score_json_file = os.path.join(self._decode_dir,"score.json")
        result_json = {}
        result_json_file = os.path.join(self._decode_dir,"result.json")

        counter = 0
        batch = self.batcher.next_batch()
         
        while batch is not None:
            # Run beam search to get best Hypothesis
            best_summary = self.beam_search(batch)

            # Extract the output ids from the hypothesis and convert back to words
            output_ids = [int(t) for t in best_summary.tokens[1:]]
            decoded_words = data.outputids2words(output_ids, self.vocab,
                                                 (batch.art_oovs[0] if config.pointer_gen else None))

            # Remove the [STOP] token from decoded_words, if necessary
            try:
                fst_stop_idx = decoded_words.index(data.STOP_DECODING)
                decoded_words = decoded_words[:fst_stop_idx]
            except ValueError:
                decoded_words = decoded_words

            original_title = batch.original_titles[0]
            """
            if config.is_tokenized:
                w_ref_article = ''.join(batch.original_articles[0].split()).strip()
                w_ref_title = ''.join(original_title.split()).strip() 
                w_dec_title =  ''.join(decoded_words).strip() 

                ref_title = ' '.join([x for x in w_ref_title])
                dec_title = ' '.join([x for x in w_dec_title])
            """    

            w_ref_article = batch.original_articles[0]
            w_ref_title = original_title
            w_dec_title = ''.join(decoded_words)

            ref_title = ' '.join([x for x in original_title])
            dec_title = ' '.join(decoded_words)
            
            '''    
            print('*'*40) 
            print('File: %d'%counter)
            print('art: %s'%w_ref_article)
            print('ref: %s'%ref_title)
            print('dec: %s'%dec_title)
            time.sleep(1)
            '''

            '''
            write_for_rouge(w_ref_article, w_ref_title, w_dec_title, counter,\
                                        self._rouge_art_dir, self._rouge_ref_dir, self._rouge_dec_dir)
            '''
            
            with open(self._result_txt, "a+") as f: 
                f.write(w_ref_article + "\t" + w_ref_title + "\t" + w_dec_title + "\n")
            
            rouge_score = rouge.get_scores(hyps = dec_title, refs = ref_title)
            
            score_json[counter] = rouge_score[0]
            counter += 1
           
            if counter % 1000 == 0:
                print('%d example in %d sec'%(counter, time.time() - start))
                start = time.time()
           
            batch = self.batcher.next_batch()
           
        print('End of decoding')
        with open(score_json_file, 'w') as f:
            json.dump(score_json,f)
            f.close()
        
        rouge_1_f = []
        rouge_1_p = []
        rouge_1_r = []
        
        rouge_2_f = []
        rouge_2_p = []
        rouge_2_r = []
        
        rouge_l_f = []
        rouge_l_p = []
        rouge_l_r = []

        for name,score in score_json.items():
            rouge_1_f.append(score["rouge-1"]['f'])
            rouge_1_p.append(score["rouge-1"]['p'])
            rouge_1_r.append(score["rouge-1"]['r'])
            
            rouge_2_f.append(score["rouge-2"]['f'])
            rouge_2_p.append(score["rouge-2"]['p'])
            rouge_2_r.append(score["rouge-2"]['r'])
                
            rouge_l_f.append(score["rouge-l"]['f'])                
            rouge_l_p.append(score["rouge-l"]['p'])
            rouge_l_r.append(score["rouge-l"]['r'])
        
        mean_1_f = sum(rouge_1_f) / len(rouge_1_f)
        mean_1_p = sum(rouge_1_p) / len(rouge_1_p)
        mean_1_r = sum(rouge_1_r) / len(rouge_1_r)
        mean_2_f = sum(rouge_2_f) / len(rouge_2_f)
        mean_2_p = sum(rouge_2_p) / len(rouge_2_p)
        mean_2_r = sum(rouge_2_r) / len(rouge_2_r)
        mean_l_f = sum(rouge_l_f) / len(rouge_l_f)
        mean_l_p = sum(rouge_l_p) / len(rouge_l_p)
        mean_l_r = sum(rouge_l_r) / len(rouge_l_r)

        result_json['mean_1_f'] = mean_1_f
        result_json['mean_1_p'] = mean_1_p
        result_json['mean_1_r'] = mean_1_r
        
        result_json['mean_2_f'] = mean_2_f
        result_json['mean_2_p'] = mean_2_p
        result_json['mean_2_r'] = mean_2_r
        result_json['mean_l_f'] = mean_l_f
        result_json['mean_l_p'] = mean_l_p
        result_json['mean_l_r'] = mean_l_r
        
        with open(result_json_file, 'w') as f:  
            json.dump(result_json, f)  
            f.close()


    def beam_search(self, batch):
        # batch should have only one example
        enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_t_0, coverage_t_0 = \
            get_input_from_batch(batch, use_cuda)

        encoder_outputs, encoder_feature, encoder_hidden = self.model.encoder(enc_batch, enc_lens)
        s_t_0 = self.model.reduce_state(encoder_hidden)

        dec_h, dec_c = s_t_0 # 1 x 2*hidden_size
        dec_h = dec_h.squeeze()
        dec_c = dec_c.squeeze()

        # decoder batch preparation, it has beam_size example initially everything is repeated
        beams = [Beam(tokens=[self.vocab.word2id(data.START_DECODING)],
                      log_probs=[0.0],
                      state=(dec_h[0], dec_c[0]),
                      context = c_t_0[0],
                      coverage=(coverage_t_0[0] if config.is_coverage else None))
                 for _ in range(config.beam_size)]
        results = []
        steps = 0
        while steps < config.max_dec_steps and len(results) < config.beam_size:
            latest_tokens = [h.latest_token for h in beams]
            latest_tokens = [t if t < self.vocab.size() else self.vocab.word2id(data.UNKNOWN_TOKEN) \
                             for t in latest_tokens]
            y_t_1 = Variable(torch.LongTensor(latest_tokens))
            if use_cuda:
                y_t_1 = y_t_1.cuda()
            all_state_h =[]
            all_state_c = []

            all_context = []

            for h in beams:
                state_h, state_c = h.state
                all_state_h.append(state_h)
                all_state_c.append(state_c)

                all_context.append(h.context)

            s_t_1 = (torch.stack(all_state_h, 0).unsqueeze(0), torch.stack(all_state_c, 0).unsqueeze(0))
            c_t_1 = torch.stack(all_context, 0)

            coverage_t_1 = None
            if config.is_coverage:
                all_coverage = []
                for h in beams:
                    all_coverage.append(h.coverage)
                coverage_t_1 = torch.stack(all_coverage, 0)

            final_dist, s_t, c_t, attn_dist, p_gen, coverage_t = self.model.decoder(y_t_1, s_t_1,
                                                        encoder_outputs, encoder_feature, enc_padding_mask, c_t_1,
                                                        extra_zeros, enc_batch_extend_vocab, coverage_t_1, steps)
            log_probs = torch.log(final_dist)
            topk_log_probs, topk_ids = torch.topk(log_probs, config.beam_size * 2)

            dec_h, dec_c = s_t
            dec_h = dec_h.squeeze()
            dec_c = dec_c.squeeze()

            all_beams = []
            num_orig_beams = 1 if steps == 0 else len(beams)
            for i in range(num_orig_beams):
                h = beams[i]
                state_i = (dec_h[i], dec_c[i])
                context_i = c_t[i]
                coverage_i = (coverage_t[i] if config.is_coverage else None)

                for j in range(config.beam_size * 2):  # for each of the top 2*beam_size hyps:
                    new_beam = h.extend(token=topk_ids[i, j].item(),
                                   log_prob=topk_log_probs[i, j].item(),
                                   state=state_i,
                                   context=context_i,
                                   coverage=coverage_i)
                    all_beams.append(new_beam)

            beams = []
            for h in self.sort_beams(all_beams):
                if h.latest_token == self.vocab.word2id(data.STOP_DECODING):
                    if steps >= config.min_dec_steps:
                        results.append(h)
                else:
                    beams.append(h)
                if len(beams) == config.beam_size or len(results) == config.beam_size:
                    break

            steps += 1

        if len(results) == 0:
            results = beams

        beams_sorted = self.sort_beams(results)

        return beams_sorted[0]

if __name__ == '__main__':
    model_path = sys.argv[1]
    beam_processor = BeamSearch(model_path)
    beam_processor.decode()


