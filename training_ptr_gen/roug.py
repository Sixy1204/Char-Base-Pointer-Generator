import os
import numpy as np
import argparse


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dir', default='decode', type=str)

    args = parser.parse_args()
    return args

def rouge(n, reference, output):
    count = 0
    for i in range(len(reference)-n+1):
        if reference[i:i+n] in output:
            count += 1
    return count

def lcs(reference, output):
    dp = [[0 for _ in range(len(output)+1)] for _ in range(len(reference)+1)]
    for i in range(1, len(reference)+1):
        for j in range(1, len(output)+1):
            if reference[i-1] == output[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[-1][-1]

args = arg_parse()
dec_files = os.listdir(os.path.join(args.dir, 'rouge_dec_dir'))
ref_files = os.listdir(os.path.join(args.dir, 'rouge_ref'))
dec_files.sort()
ref_files.sort()

rouge1, rouge2, rougeL =  [], [], []
R1, R2, RL = [], [], []

for i in range(len(dec_files)):
    with open(os.path.join(args.dir, 'rouge_dec_dir', dec_files[i])) as f:
        output = f.readlines()[0]
    with open(os.path.join(args.dir, 'rouge_ref', ref_files[i])) as f:
        reference = f.readlines()[0]
    
    n1 = rouge(1, reference, output)
    n2 = rouge(2, reference, output)
    l = lcs(reference, output)
    
    p1, r1 = n1/len(output), n1/len(reference)
    p2, r2 = n2/(len(output)-1), n2/(len(reference)-1)
    pl, rl = l/len(output), l/len(reference)

    R1.append(r1)
    R2.append(r2)
    RL.append(rl)

    if n1 == 0:
        rouge1.append(0)
    else:
        rouge1.append(2*p1*r1/(p1+r1))


    if n2 == 0:
        rouge2.append(0)
    else:
        rouge2.append(2*p2*r2/(p2+r2))

    if l == 0:
        rougeL.append(0)
    else:
        rougeL.append(2*pl*rl/(pl+rl))


avg_rouge1_f1 = np.mean(rouge1)
avg_rouge2_f1 = np.mean(rouge2)
avg_rougeL_f1 = np.mean(rougeL)

print('rouge1: {}'.format(np.mean(R1)))
print('rouge2: {}'.format(np.mean(R2)))
print('rouge_L: {}'.format(np.mean(RL)))

