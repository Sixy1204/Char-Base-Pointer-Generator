import numpy as np
import os
import struct
import collections
from tensorflow.core.example import example_pb2
import argparse

# In[ ]:

parser = argparse.ArgumentParser(description='chunk2bin')
parser.add_argument("-load_data_path",
                    dest="data_path", 
                    required=True,
                    help="Read in files path")

parser.add_argument("-finished_file_path",
                    dest="finish_path", 
                    required=False,
                    default='../data/finish_token_bin',
                    help="Saving bin files path")

args = parser.parse_args()


VOCAB_SIZE = 50_000  # 词汇表大小
CHUNK_SIZE = 1000    # 每个分块example的数量，用于分块的数据

# tf模型数据文件存放目录
FINISHED_FILE_DIR = args.finish_path
CHUNKS_DIR = os.path.join(FINISHED_FILE_DIR, 'chunked')


data_pth = args.data_path
train = np.load(os.path.join(data_pth, 'train_sub.npy'), allow_pickle=True)
test = np.load(os.path.join(data_pth, 'test_sub.npy'), allow_pickle=True)

# In[ ]:


def chunk_file(finished_files_dir, chunks_dir, name, chunk_size):
    """构建二进制文件"""
    in_file = os.path.join(finished_files_dir, '%s.bin' % name)
    print(in_file)
    reader = open(in_file, "rb")
    chunk = 0
    finished = False
    while not finished:
        chunk_fname = os.path.join(chunks_dir, '%s_%03d.bin' % (name, chunk))  # 新的分块
        with open(chunk_fname, 'wb') as writer:
            for _ in range(chunk_size):
                len_bytes = reader.read(8)
                if not len_bytes:
                    finished = True
                    break
                str_len = struct.unpack('q', len_bytes)[0]
                example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
                writer.write(struct.pack('q', str_len))
                writer.write(struct.pack('%ds' % str_len, example_str))
            chunk += 1


def chunk_all():
    # 创建一个文件夹来保存分块
    if not os.path.isdir(CHUNKS_DIR):
        os.mkdir(CHUNKS_DIR)
    # 将数据分块
    for name in ['train', 'test']:
        print("Splitting %s data into chunks..." % name)
        chunk_file(FINISHED_FILE_DIR, CHUNKS_DIR, name, CHUNK_SIZE)
    print("Saved chunked data in %s" % CHUNKS_DIR)


def write_to_bin(input_file, out_file, makevocab=True):
    """生成模型需要的文件"""
    if makevocab:
        vocab_counter = collections.Counter()

    with open(out_file, 'wb') as writer:
        
        for data in input_file:
            #TODO Note encode target var name
            article = data['content']
            abstract = data['title']
            # 写入tf.Example
            tf_example = example_pb2.Example()
            tf_example.features.feature['article'].bytes_list.value.extend([bytes(article, encoding='utf-8')])
            tf_example.features.feature['abstract'].bytes_list.value.extend([bytes(abstract, encoding='utf-8')])
            tf_example_str = tf_example.SerializeToString()
            str_len = len(tf_example_str)
            writer.write(struct.pack('q', str_len))
            writer.write(struct.pack('%ds' % str_len, tf_example_str))

            # 如果可以，将词典写入文件
            if makevocab: 
                vocab_counter.update(article)
                vocab_counter.update(abstract)
    
    print("Finished writing file %s\n" % out_file)

    # 将词典写入文件
    if makevocab:
        print("Writing vocab file...")
        with open(os.path.join(FINISHED_FILE_DIR, "vocab"), 'w', encoding='utf-8') as writer:
            for word, count in vocab_counter.most_common(VOCAB_SIZE):
                writer.write(word + ' ' + str(count) + '\n')
        print("Finished writing vocab file")

if not os.path.exists(FINISHED_FILE_DIR):
    os.makedirs(FINISHED_FILE_DIR)

write_to_bin(test, os.path.join(FINISHED_FILE_DIR, "test.bin"))
write_to_bin(train, os.path.join(FINISHED_FILE_DIR, "train.bin"), makevocab=True)

chunk_all()
