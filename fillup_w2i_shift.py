from data_preprocess2 import fillup_table_w2i
from utils import savepkl, loadpkl
import os

path = './data/w_all_data/'

x_unpad = loadpkl(os.path.join(path, 'x_tokenised_preprocessed.pkl'))
vocab = loadpkl(os.path.join(path, 'vocab_5-15_unk.pkl'))

print(x_unpad.shape, len(vocab))
x_unpad = fillup_table_w2i(vocab, x_unpad)
print(x_unpad.shape, len(vocab))
savepkl(os.path.join(path, 'x_tokenised_preprocessed_.pkl'), x_unpad)
