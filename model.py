import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()
        # Embedding layer that turns words into a vector of a specified size
        self.word_embed = nn.Embedding(vocab_size,embed_size)
        # The LSTM takes embedded word vectors as input
        # and outputs hidden states of size hidden_dim
        self.lstm = nn.LSTM(embed_size,hidden_size,num_layers,batch_first = True)
        # The linear layer that maps the hidden state output dimension
        # to the number of words we want as output, vocab_size
        self.linear = nn.Linear(hidden_size, vocab_size)
        

    
    def forward(self, features, captions):
        """ Define the feedforward behavior of the model """
        
        # Create embedded word vectors for each word in the captions
        # captions: Discard the <end> word to avoid predicting when <end> is the input of the RNN
        embeddings = self.word_embed(captions[:, :-1])
        # Stack the features and captions
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        # Get the output by passing the lstm over our word embeddings
        LSTM_out, _ = self.lstm(embeddings)
        outputs = self.linear(LSTM_out)
        return outputs

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
       
        result_ids = []
        prediction = None
        max_len = 20
        states = None 
        features = inputs

        for i in range(max_len):
            if(prediction != 1): 
                
                LSTM_out, states = self.lstm(features, states)
                output = self.linear(LSTM_out)
                _, predicted = output.max(2)
                prediction = predicted.item()
                result_ids.append(prediction)
                features = self.word_embed(predicted)

        return result_ids