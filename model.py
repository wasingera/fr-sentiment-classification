import torch

class FrenchSentimentAnalysis(torch.nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
        super(FrenchSentimentAnalysis, self).__init__()
        self.embedding = torch.nn.Embedding(input_dim, embedding_dim)
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        output, (hidden, cell) = self.lstm(embedded)
        return self.fc(hidden.squeeze(0))
        # output, hidden = self.rnn(embedded)
        #
        # ## D * num_layers, N, H
        # ## we can squeeze the first dimension because it is 1
        # return self.fc(hidden.squeeze(0))
