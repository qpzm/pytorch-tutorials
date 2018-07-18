import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# In this tutorial use 2 output probability even though one can be computed by 1 - the_other
class BoWClassifier(nn.Module):
    def __init__(self, num_labels, vocab_size):
        super(BoWClassifier, self).__init__()
        self.linear = nn.Linear(vocab_size, num_labels)

    def forward(self, bow_vec):
        return F.log_softmax(self.linear(bow_vec), dim=1)


def make_bow_vector(sentence, word_to_idx):
    vec = torch.zeros(len(word_to_idx))
    for word in sentence:
        vec[word_to_idx[word]] += 1
    return vec.view(1, -1)


def make_target(label, label_to_idx):
    # CPU version
    return torch.LongTensor([label_to_idx[label]])


def train(data):
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    for epoch in range(100):
        for sentence, label in data:
            model.zero_grad()
            bow_vec = make_bow_vector(sentence, word_to_idx)
            target = make_target(label, label_to_idx)

            log_probs = model(bow_vec)

            loss = loss_function(log_probs, target)
            loss.backward()
            optimizer.step()


def test(data, word_to_idx):
    # class_correct = list(0.0 for i in range(2))
    # class_total = list(0.0 for i in range(2))
    with torch.no_grad():
        for sentence, label in data:
            print(predict(sentence, word_to_idx))


def predict(sentence, word_to_idx):
    bow_vec = make_bow_vector(sentence, word_to_idx)
    log_probs = model(bow_vec)
    return torch.argmax(log_probs)


data = [("me gusta comer en la cafeteria".split(), "SPANISH"),
        ("Give it to me".split(), "ENGLISH"),
        ("No creo que sea una buena idea".split(), "SPANISH"),
        ("No it is not a good idea to get lost at sea".split(), "ENGLISH")]

test_data = [("Yo creo que si".split(), "SPANISH"),
             ("it is lost on me".split(), "ENGLISH")]

label_to_idx = {"SPANISH": 0, "ENGLISH": 1}
word_to_idx = {}
for sentence, _ in data + test_data:
    for word in sentence:
        if word not in word_to_idx:
            word_to_idx[word] = len(word_to_idx)

print(word_to_idx)

VOCAB_SIZE = len(word_to_idx)
NUM_LABELS = 2

model = BoWClassifier(NUM_LABELS, VOCAB_SIZE)

for param in model.parameters():
    print(param)

print("Test statistics before training")
test(test_data, word_to_idx)
# model.parameters() is a generator, so you can access the first parameter by next()
print(next(model.parameters()).size())
# how to go to 2nd?
print(next(model.parameters()).size())
# print(next(model.parameters())[:, word_to_idx['creo']])

train(data)

print("Test statistics after training")
test(test_data, word_to_idx)
print(next(model.parameters())[:, word_to_idx['creo']])
