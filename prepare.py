import os
import pandas as pd
import spacy

import config
from vocab import Vocab

def split_data(file, train_file, val_file, ratio):
    if os.path.exists(train_file) or os.path.exists(val_file):
        print('Found train/val files, skipping split...')
        return

    df = pd.read_csv(file)

    ## shrink data for small model testing
    df = df.sample(frac=0.25, random_state=1024)

    train_df = df.sample(frac=ratio, random_state=1024)
    val_df = df.drop(train_df.index)

    train_df.to_csv(train_file, index=False)
    val_df.to_csv(val_file, index=False)

def create_vocab(file, vocab_file):
    vocab = Vocab(max_words=config.max_words, min_freq=config.min_freq,
                  special_tokens=config.special_tokens, default_token=config.unk_token)
    vocab.build_vocab_from_file(file, vocab_file)
    return vocab

def tokenize_data(file, vocab, out_file):
    if os.path.exists(out_file):
        print(f'Found {out_file}, skipping tokenization...')
        return

    print("Tokenizing data...")

    nlp = spacy.blank('fr')
    df = pd.read_csv(file)

    with open(out_file, 'w') as f:
        f.write('text,label\n')
        for text, label in zip(df['text'], df['label']):
            tokens = [token.text for token in nlp(text.lower())]
            tokens = [vocab[token] for token in tokens]
            tokens = ' '.join(map(str, tokens))
            f.write(f'{tokens},{label}\n')

    print("Tokenized data saved to", out_file)

if __name__ == '__main__':
    split_data(config.data_file, config.train_file, config.val_file, 0.8)
    vocab = create_vocab(config.train_file, config.vocab_file)
    tokenize_data(config.train_file, vocab, config.train_file.replace('.gz', '.tkn'))
    tokenize_data(config.val_file, vocab, config.val_file.replace('.gz', '.tkn'))
