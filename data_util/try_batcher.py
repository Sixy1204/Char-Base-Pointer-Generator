from batcher import Batcher
import config
from data import Vocab
vocab = Vocab('../data/vocab', 20000)
batcher = Batcher('../data/train.npy', vocab, 'train', config.batch_size, single_pass=False)
batch = batcher.next_batch()

print('enc input ', batch.enc_batch)
print('dec input ', batch.dec_batch)
