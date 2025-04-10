import torch
from torch import nn
from torch.nn import CrossEntropyLoss

class RNNRegressor(nn.Module):
    def __init__(self, n_feature=128, n_rnn_layer=1, n_char=46, layer_type='GRU'): # why feature is 128? Is there any reason?
        super().__init__()
        self.embedding = nn.Embedding(n_char, n_feature) # n_char: 46, n_feature: 128
        if layer_type == 'GRU':
            self.rnn = nn.GRU(input_size=n_feature, hidden_size=n_feature, num_layers=n_rnn_layer, batch_first=True)
        elif layer_type =='LSTM':
            self.rnn = nn.LSTM(input_size=n_feature, hidden_size=n_feature, num_layers=n_rnn_layer, batch_first=True)
        elif layer_type == 'RNN':
            self.rnn = nn.RNN(input_size=n_feature, hidden_size=n_feature, num_layers=n_rnn_layer, batch_first=True)
        self.fc = nn.Linear(n_feature, 1) 

    def forward(self, x, l):
        embedding_output = self.embedding(x) 
        rnn_output, _ = self.rnn(embedding_output) 

        last_hidden_state_list = []
        for i in range(len(l)): # for each sample
            # i: i-th sample
            # l[i]: length of i-th sample
            last_hidden_state_list.append(rnn_output[i, l[i]-1, :]) # extract hidden state from the last valid position of each sample

        last_hidden_state = torch.stack(last_hidden_state_list) # stack all hidden states
        output = self.fc(last_hidden_state) # output have to learn log p

        return output

