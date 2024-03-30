import torch

import config
from model import FrenchSentimentAnalysis
from vocab import Vocab

vocab = Vocab(max_words=config.max_words, min_freq=config.min_freq,
              special_tokens=config.special_tokens, default_token=config.unk_token)
vocab.build_vocab_from_file(config.train_file, config.vocab_file)

model = FrenchSentimentAnalysis(len(vocab), config.embedding_dim, config.hidden_dim, config.output_dim).to(config.device)
model.load_state_dict(torch.load(config.model_file))

def predict_sentiment(text, model, vocab):
    model.eval()
    with torch.no_grad():
        tokens = vocab.numericalize_text(text)
        tokens = torch.tensor(tokens).unsqueeze(0).to(config.device)
        output = model(tokens)
        prediction = torch.argmax(output, dim=1).item()
        predicted_prob = torch.softmax(output, dim=1).squeeze().tolist()
        return prediction, predicted_prob

sentiment_map = {0: 'Negative', 1: 'Positive'}

text = "Ce film est super. Je l'ai adoré!"
sentiment, prob = predict_sentiment(text, model, vocab)
print(f"Predicted sentiment: {sentiment_map[int(sentiment)]}")
print(f"Confidence: {100*prob[int(sentiment)]:.2f}%")
