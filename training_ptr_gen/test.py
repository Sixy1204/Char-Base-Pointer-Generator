from os.path import isfile, isdir, join
from os import listdir
from decode import BeamSearch
'''
log_pth = '../log'
lastest_log = join(log_pth, listdir(log_pth)[-1])
model_check_file = join(lastest_log, listdir(lastest_log)[0])
model_path = join(model_check_file, listdir(model_check_file)[9])
'''
model_path = "../log/train_model_1100000_coverage/model/model_1002000"
beam_search_processor = BeamSearch(model_path)
beam_search_processor.decode()
