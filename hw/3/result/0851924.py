import numpy as np
import os
import time
import math
import string
import random
import torch
import torch.nn as nn
from torch.autograd import Variable
import unidecode
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm

# Parse command line arguments
argparser = argparse.ArgumentParser()
argparser.add_argument('--filename', type=str, default="../dataset_Shakespeare/shakespeare_train.txt")
argparser.add_argument('--model', type=str, default="lstm")
argparser.add_argument('--n_epochs', type=int, default=2000)
argparser.add_argument('--print_every', type=int, default=400)
argparser.add_argument('--hidden_size', type=int, default=150)
argparser.add_argument('--n_layers', type=int, default=2)
argparser.add_argument('--learning_rate', type=float, default=0.001)
argparser.add_argument('--seq_len', type=int, default=50)
argparser.add_argument('--batch_size', type=int, default=100)
argparser.add_argument('--shuffle', action='store_true')
argparser.add_argument('--cuda', action='store_true')
args = argparser.parse_args()

def read_file(filename):
    file = unidecode.unidecode(open(filename).read())
    file = file.lower()
    '''vocab = set(file)
    vocab_to_int = {c:i for i,c in enumerate(vocab)}
    int_to_vocab = dict(enumerate(vocab))
    file = np.array([vocab_to_int[c] for c in file],dtype=np.int32)'''
    return file, len(file)

# Turning a string into a tensor
def char2tensor(string):
    tensor = torch.zeros(len(string)).long()
    for c in range(len(string)):
        try:
            tensor[c] = all_characters.index(string[c])
        except:
            continue
    return tensor

def generate(decoder, prime_str='I', predict_len=500, temperature=0.8, cuda=False):
    hidden = decoder.init_hidden(1)
    prime_input = Variable(char2tensor(prime_str).unsqueeze(0))

    if cuda:
        hidden = hidden.cuda()
        prime_input = prime_input.cuda()
    predicted = prime_str

    # Use priming string to "build up" hidden state
    for p in range(len(prime_str) - 1):
        _, hidden = decoder(prime_input[:,p], hidden)
        
    inp = prime_input[:,-1]
    
    for p in range(predict_len):
        output, hidden = decoder(inp, hidden)
        
        # Sample from the network as a multinomial distribution
        output_dist = output.data.view(-1).div(temperature).exp()
        top_i = torch.multinomial(output_dist, 1)[0]

        # Add predicted character to string and use as next input
        predicted_char = all_characters[top_i]
        predicted += predicted_char
        inp = Variable(char2tensor(predicted_char).unsqueeze(0))
        if cuda:
            inp = inp.cuda()

    return predicted

def random_training_set(seq_len, batch_size):
    inp = torch.LongTensor(batch_size, seq_len)
    target = torch.LongTensor(batch_size, seq_len)
    for bi in range(batch_size):
        start_index = random.randint(0, file_len - seq_len)
        end_index = start_index + seq_len + 1
        
        seq = file[start_index:end_index]
        inp[bi] = char2tensor(seq[:-1])
        target[bi] = char2tensor(seq[1:])
    inp = Variable(inp)
    target = Variable(target)
    if args.cuda:
        inp = inp.cuda()
        target = target.cuda()
    return inp, target

def random_validation_set(seq_len, batch_size):
    inp = torch.LongTensor(batch_size, seq_len)
    target = torch.LongTensor(batch_size, seq_len)
    for bi in range(batch_size):
        start_index = random.randint(0, val_file_len - seq_len-1)
        end_index = start_index + seq_len + 1
        
        seq = val_file[start_index:end_index]
        inp[bi] = char2tensor(seq[:-1])
        target[bi] = char2tensor(seq[1:])
    inp = Variable(inp)
    target = Variable(target)
    if args.cuda:
        inp = inp.cuda()
        target = target.cuda()
    return inp, target

def train(inp, target):
    hidden = decoder.init_hidden(args.batch_size)
    if args.cuda:
        hidden = hidden.cuda()
    decoder.zero_grad()
    loss = 0
    acc=0
    for c in range(args.seq_len):
        output, hidden = decoder(inp[:,c], hidden)
        loss += criterion(output.view(args.batch_size, -1), target[:,c])
            
    loss.backward()
    decoder_optimizer.step()

    return loss.item() / args.seq_len,hidden

def validate(inp, target,hidden_inp):
    loss = 0
    hidden=hidden_inp
    acc=0
    for c in range(args.seq_len):
        output, hidden = decoder(inp[:,c], hidden)
        loss += criterion(output.view(args.batch_size, -1), target[:,c])
        
    return loss.item() / args.seq_len

def save():
    if args.model == "rnn":
        save_filename = os.path.splitext(os.path.basename(args.filename))[0] + '.pt'
    elif args.model == "lstm":
        save_filename = os.path.splitext(os.path.basename(args.filename))[0] + '_lstm.pt'
    
    torch.save(decoder, save_filename)
    print('Saved as %s' % save_filename)

class CharRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, model="rnn", n_layers=1):
        super(CharRNN, self).__init__()
        self.model = model.lower()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.encoder = nn.Embedding(input_size, hidden_size)
        if self.model == "rnn":
            self.rnn = nn.RNN(hidden_size, hidden_size, n_layers)
        elif self.model == "lstm":
            self.rnn = nn.LSTM(hidden_size, hidden_size, n_layers)
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        batch_size = input.size(0)
        encoded = self.encoder(input)
        output, hidden = self.rnn(encoded.view(1, batch_size, -1), hidden)
        output = self.decoder(output.view(batch_size, -1))
        return output, hidden

    def forward2(self, input, hidden):
        encoded = self.encoder(input.view(1, -1))
        output, hidden = self.rnn(encoded.view(1, 1, -1), hidden)
        output = self.decoder(output.view(1, -1))
        return output, hidden

    def init_hidden(self, batch_size):
        if self.model == "lstm":
            return (Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)),
                    Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)))
        return Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size))

all_characters = string.printable
n_characters = len(all_characters)

file, file_len = read_file(args.filename)
val_file, val_file_len = read_file('../dataset_Shakespeare/shakespeare_valid.txt')

# Initialize models and start training
decoder = CharRNN(
    n_characters,
    args.hidden_size,
    n_characters,
    model=args.model,
    n_layers=args.n_layers,
)
decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=args.learning_rate)
criterion = nn.CrossEntropyLoss()

if args.cuda:
    decoder.cuda()

start = time.time()
train_losses = []
loss_avg = 0
val_losses=[]

try:
    print("Training for %d epochs..." % args.n_epochs)
    for epoch in range(1, args.n_epochs + 1):
        train_loss,hidden_inp = train(*random_training_set(args.seq_len, args.batch_size))
        
        loss_avg += train_loss
        train_losses.append(train_loss)
        
        val_loss = validate(*random_validation_set(args.seq_len, args.batch_size),hidden_inp=hidden_inp)
        val_losses.append(val_loss)
        
        if epoch % args.print_every == 0:            
            out_res = generate(decoder, 'he', 100, cuda=args.cuda)
            print(str(epoch) + ":\n" + out_res)
            print('\ntrain_loss: ', train_loss, 'val_loss: ', val_loss, '\n')

    print("Saving...")
    save()
    
    plt.title('training curve')
    plt.plot(val_losses, color='Red', label='Test Loss')
    plt.plot(train_losses, color='Blue', label='Train Loss')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
    print('Hidden Size:' + str(args.hidden_size) + ' Sequence length:' + str(args.seq_len))
    # print(train_losses)
    # print(val_losses)

except KeyboardInterrupt:
    print("Saving before quit...")
    save()