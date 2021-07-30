import numpy as np
#val = np.load('../data/val.npy', allow_pickle=True)
test = np.load('../data/original_data/test.npy', allow_pickle=True)
train = np.load('../data/original_data/train.npy', allow_pickle=True)

print('Original train length %d'%len(train))
print('Original test length %d'%len(test))

train_chunk = train
#val_chunk = val[:int(len(val)//2)]
test_chunk= test[:int(len(test)//2)]

print('Start truncate.....')

np.save('../data/original_data/original_subset/train_sub.npy', train_chunk)
#np.save('../data/val_chunk.npy', val_chunk)
np.save('../data/original_data/original_subset/test_sub.npy', test_chunk)

print('Done chunk length train=%d'%(len(train_chunk)))
print('Done chunk length test=%d'%(len(test_chunk)))
