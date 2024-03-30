import torch

## Data Preprocessing
data_file = 'data/full.zip'
train_file = 'data/train.gz'
val_file = 'data/val.gz'
vocab_file = 'data/vocab.pkl'
special_tokens = ['[PAD]', '[UNK]', '[START]', '[END]']
unk_token = '[UNK]'
pad_idx = 0
max_words = 10000
min_freq = 5

## Training
batch_size = 1024
lr = 1e-3
epochs = 10
device = 'cuda' if torch.cuda.is_available() else 'cpu'

## Model
embedding_dim = 100
hidden_dim = 256
output_dim = 2
model_file = 'models/model.pt'
ckpt_file = 'models/ckpt.pt'
