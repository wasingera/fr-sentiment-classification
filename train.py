import torch, pandas, tqdm, os
from torch.utils.data import Dataset, DataLoader

import config
from model import FrenchSentimentAnalysis
from vocab import Vocab

class TweetDataset(Dataset):
    def __init__(self, file):
        self.data = pandas.read_csv(file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = [int(x) for x in self.data['text'][idx].split()]
        label = self.data['label'][idx]
        return torch.tensor(text), torch.tensor(label)

def collate_fn(batch):
    texts, labels = zip(*batch)
    texts = torch.nn.utils.rnn.pad_sequence(texts, batch_first=True)
    labels = torch.stack(labels)
    return texts, labels

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    device = config.device

    it = tqdm.tqdm(dataloader, desc="Training...", ascii=True)

    model.train()
    for batch, (X, y) in enumerate(it):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            it.set_description(f"Training...  Loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            # print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    device = config.device

    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        it = tqdm.tqdm(dataloader, desc="Testing...", ascii=True)
        for X, y in it:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    return test_loss

def train_model(train_loader, val_loader, model, loss_fn, optimizer, curr_epoch, epochs):
    best_loss = float('inf')
    for t in range(curr_epoch, epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_loader, model, loss_fn, optimizer)
        test_loss = test(val_loader, model, loss_fn)
        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(model.state_dict(), config.model_file)

        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': test_loss,
            'epoch': t,
        }, config.ckpt_file)
    print("Done!")

if __name__ == '__main__':
    train_dataset = TweetDataset(config.train_file.replace('.gz', '.tkn'))
    val_dataset = TweetDataset(config.val_file.replace('.gz', '.tkn'))

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    vocab = Vocab()
    vocab.load(config.vocab_file)

    model = FrenchSentimentAnalysis(len(vocab), config.embedding_dim, config.hidden_dim, config.output_dim).to(config.device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    epoch = 0

    if os.path.exists(config.ckpt_file):
        checkpoint = torch.load(config.ckpt_file)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        print(f"Loaded model from epoch {epoch}")

    train_model(train_loader, val_loader, model, loss_fn, optimizer, epoch, config.epochs)
