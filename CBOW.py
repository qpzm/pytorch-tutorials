import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pdb

CONTEXT_SIZE = 2  # 2 words to the left, 2 to the right
EMBEDDING_DIM = 10
raw_text = """We are about to study the idea of a computational process.
Computational processes are abstract beings that inhabit computers.
As they evolve, processes manipulate other abstract things called data.
The evolution of a process is directed by a pattern of rules
called a program. People create programs to direct processes. In effect,
we conjure the spirits of the computer with our spells.""".split()

vocab = set(raw_text)
vocab_size = len(vocab)

word_to_ix = {word: i for i, word in enumerate(vocab)}
data = []
for i in range(2, len(raw_text) - 2):
    context = [raw_text[j] for j in [i - 2, i - 1, i + 1, i + 2]]
    target = raw_text[i]
    data.append((context, target))
print(data[:5])


class CBOW(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).sum(dim=0).view(1,-1)
        out = self.linear(embeds)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs


def make_context_vector(context, word_to_ix):
    return torch.tensor([word_to_ix[word] for word in context], dtype=torch.long)


print(make_context_vector(data[0][0], word_to_ix))

model = CBOW(vocab_size, EMBEDDING_DIM, CONTEXT_SIZE)
losses = []
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

for epoch in range(100):
    total_loss = 0
    for context, target in data:
        model.zero_grad()
        log_probs = model(make_context_vector(context, word_to_ix))
        loss = loss_function(log_probs, torch.tensor([word_to_ix[target]], dtype=torch.long))
        loss.backward()
        optimizer.step()

        # Get the Python number from a 1-element Tensor by calling tensor.item()
        total_loss += loss.item()
    losses.append(total_loss)
print(losses)
