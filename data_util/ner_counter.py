import numpy as np
import argparse

parser = argparse.ArgumentParser(description="Compute Ner coverage rate")

parser.add_argument("-ref_test_npy",
                    dest="ref_test", 
                    required=True,
                    help="Ref test dataset path")

parser.add_argument("-ref_dec_txt",
                    dest="ref_dec", 
                    required=True,
                    help="ref_dec.txt in log file")


args = parser.parse_args()
 
ref = np.load(args.ref_test, allow_pickle=True)

with open(args.ref_dec, 'r') as f:
    lines = f.readlines()
    idx = 0
    cover_rate = []
    for line in lines:
        counter = 0
        line = line.split()
        dec = line[2]
        ref_ner = [x[1] for x in ref[idx]['ner']]
        if ref_ner == []:
            continue
        for w in ref_ner:
            if w in dec:
                counter += 1

        rate = counter/len(ref_ner)
        cover_rate.append(rate)
        idx += 1
        print('Dec ', dec)
        print('Ner ', ref_ner)
        print('Ner counter ', counter)
        print('Rate ', rate)
avg_cover_rate = sum(cover_rate)/len(cover_rate)
print('Avg NER coverage rate is %3f'%avg_cover_rate)
