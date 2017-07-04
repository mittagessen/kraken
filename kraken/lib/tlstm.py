import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np

import clstm_pb2


from warpctc_pytorch import CTCLoss


class TBIDILSTM(nn.Module):
    """
    A torch module implementing a bidirectional LSTM with a linear layer as decoder.
    Very much the same as ocropy BIDILSTM but with logsoftmax instead of softmax at 
    the output (needed for warp_ctc). The serialization format is the same as clstm's.
    """
    def __init__(self, ninp, nhid, nop):
        super(TBIDILSTM, self).__init__()
        
        self.rnn = nn.LSTM(ninp+1, nhid, 1, bias=False, bidirectional=True)
        self.decoder = nn.Linear(2*nhid+1, nop, bias=False)
        self.softmax = nn.LogSoftmax()
        
        self.init_weights()
        
        self.ninput = ninp
        self.noutput = nop
        self.nhidden = nhid
        self.learning_rate = 0
        self.momentum = 0
        
        
    def init_weights(self, initrange=0.1):
        self.decoder.weight.data.uniform_(-initrange, initrange)
        for p in self.rnn.parameters():
            p.data.uniform_(-initrange, initrange)
        
        
    def forward(self, inp, hidden):
        leninp = len(inp)
        cuda = True if inp.is_cuda else False
        
        ones1 = Variable(torch.ones(inp.size()[0], inp.size()[1], 1))
        if cuda:
            ones1 = ones1.cuda()
        inp_oneup = torch.cat([ones1, inp], 2)

        lstm_out, hidden = self.rnn(inp_oneup, hidden)
        
        ones2 = Variable(torch.ones(lstm_out.size(0)*lstm_out.size(1),1))
        if cuda:
            ones2 = ones2.cuda()
        lstm_out_oneup = torch.cat([ ones2, 
                            lstm_out.view(lstm_out.size(0)*lstm_out.size(1), lstm_out.size(2)) ], 1)
        decoded = self.softmax(self.decoder(lstm_out_oneup))
        
        return decoded.view(lstm_out.size(0), lstm_out.size(1), decoded.size(1)), hidden

    
    def init_hidden(self, bsz=1):
        if self.rnn.weight_hh_l0.is_cuda:
            return (Variable(torch.zeros(2, bsz, self.nhidden)).cuda(),
                    Variable(torch.zeros(2, bsz, self.nhidden)).cuda())           
        return (Variable(torch.zeros(2, bsz, self.nhidden)),
                Variable(torch.zeros(2, bsz, self.nhidden)))
    
    

class TlstmSeqRecognizer(kraken.lib.lstm.SeqRecognizer):
    """
    Something like ClstmSeqRecognizer, using pytorch instead of clstm.
    """
    def __init__(self, fname='', normalize=kraken.lib.lstm.normalize_nfkc, cuda=torch.cuda.is_available()):
        self.fname = fname
        self.rnn = None
        self.normalize = normalize
        self.cuda_available = cuda
        if fname:
            self._load_model()
    
    @classmethod
    def init_model(cls, ninput, nhidden, noutput, codec, normalize=kraken.lib.lstm.normalize_nfkc, cuda=torch.cuda.is_available()):
        self = cls()
        self.codec = codec
        self.normalize = normalize
        self.rnn = TBIDILSTM(ninput, nhidden, noutput)
        self.setLearningRate()
        self.trial = 0
        self.criterion = CTCLoss()
        self.cuda_available = cuda
        if self.cuda_available:
            self.cuda()
        return self
    
    def cuda(self):
        if not self.cuda_available:
            return 'CUDA not available!'
        
        self.rnn = self.rnn.cuda()
        self.criterion = self.criterion.cuda()
    
    def save_model(self, path):
        network = clstm_pb2.NetworkProto(kind='Stacked', ninput=self.rnn.ninput, noutput = self.rnn.noutput)
        
        network.codec.extend([0]+[ord(c) for c in newmodel.codec.code2char.values()][1:])
        
        network.attribute.extend([
            clstm_pb2.KeyValue(key='kind', value='bidi'),
            clstm_pb2.KeyValue(key='learning_rate', value='{:4f}'.format(self.rnn.learning_rate)),
            clstm_pb2.KeyValue(key='momentum', value='{:4f}'.format(self.rnn.momentum)),
            clstm_pb2.KeyValue(key='trial', value=repr(self.trial))
        ])
        
        hiddenattr = clstm_pb2.KeyValue(key='nhidden', value=repr(self.rnn.nhidden))
        networks = {}
        networks['paral'] = clstm_pb2.NetworkProto(kind='Parallel', ninput=self.rnn.ninput, noutput=self.rnn.nhidden*2)

        networks['lstm1'] = clstm_pb2.NetworkProto(kind='NPLSTM', ninput=self.rnn.ninput, noutput=self.rnn.nhidden)
        networks['lstm1'].attribute.extend([hiddenattr])
        
        networks['rev'] = clstm_pb2.NetworkProto(kind='Reversed', ninput=self.rnn.ninput, noutput=self.rnn.nhidden)
        networks['lstm2'] = clstm_pb2.NetworkProto(kind='NPLSTM', ninput=self.rnn.ninput, noutput=self.rnn.nhidden)
        networks['lstm2'].attribute.extend([hiddenattr])
        
        networks['softm'] = clstm_pb2.NetworkProto(kind='SoftmaxLayer', ninput=self.rnn.nhidden*2, noutput=self.rnn.noutput)
        networks['softm'].attribute.extend([hiddenattr])
        
        
        # weights
        weights = {}
        weights['lstm1'] = {}
        weights['lstm2'] = {}
        weights['softm'] = {}
        weights['lstm1']['WGI'], weights['lstm1']['WGF'], weights['lstm1']['WCI'], weights['lstm1']['WGO'] = \
            torch.cat([self.rnn.rnn.weight_ih_l0, self.rnn.rnn.weight_hh_l0], 1).split(100, 0)
        weights['lstm2']['WGI'], weights['lstm2']['WGF'], weights['lstm2']['WCI'], weights['lstm2']['WGO'] = \
            torch.cat([self.rnn.rnn.weight_ih_l0_reverse, self.rnn.rnn.weight_hh_l0_reverse], 1).split(100, 0)
        weights['softm']['W1'] = self.rnn.decoder.weight
        
        for n in weights.keys():
            for w in sorted(weights[n].keys()):
                warray = clstm_pb2.Array(name=w, dim=list(weights[n][w].size()))
                for v in weights[n][w].data.cpu().numpy().tolist():
                    warray.value.extend(v)
                networks[n].weights.extend([warray])
        
        networks['rev'].sub.extend([networks['lstm2']])
        networks['paral'].sub.extend([networks['lstm1'], networks['rev']])
        network.sub.extend([networks['paral'], networks['softm']])
        
        with open(path, 'wb') as fp:
            fp.write(network.SerializeToString())
    
    def _load_model(self):
        network = clstm_pb2.NetworkProto()
        with open(self.fname, 'rb') as f:
            network.ParseFromString(f.read())
        
        ninput = network.ninput
        noutput = network.noutput
        attributes = {a.key: a.value for a in network.attribute[:]}
        self.kind = attributes['kind']
        lrate = float(attributes['learning_rate'])
        momentum = float(attributes['momentum'])
        self.trial = int(attributes['trial'])
        
        # Codec
        self.codec = kraken.lib.lstm.Codec()
        code2char, char2code = {}, {}
        for code, char in enumerate([126] + network.codec[1:]):
            code2char[code] = chr(char)
            char2code[chr(char)] = code
        self.codec.code2char = code2char
        self.codec.char2code = char2code
        
        # Networks
        networks = {}
        networks['softm'] = [n for n in network.sub[:] if n.kind == 'SoftmaxLayer'][0]
        parallel = [n for n in network.sub[:] if n.kind == 'Parallel'][0]
        networks['lstm1'] = [n for n in parallel.sub[:] if n.kind == 'NPLSTM'][0]
        rev = [n for n in parallel.sub[:] if n.kind == 'Reversed'][0]
        networks['lstm2'] = rev.sub[0]
        
        nhidden = int(networks['lstm1'].attribute[0].value)
        
        weights = {}
        for n in networks:
            weights[n] = {}
            for w in networks[n].weights[:]:
                weights[n][w.name] = np.array(w.value).reshape(w.dim[:])
        self.weights = weights
        
        weightnames = ('WGI', 'WGF', 'WCI', 'WGO')
        # lstm
        ih_hh_splits = torch.cat([torch.from_numpy(w.astype('float32')) \
                                  for w in [weights['lstm1'][wn] \
                                        for wn in weightnames]],0).split(ninput+1,1)
        weight_ih_l0 = ih_hh_splits[0]
        weight_hh_l0 = torch.cat(ih_hh_splits[1:], 1)
        
        # lstm_reversed
        ih_hh_splits = torch.cat([torch.from_numpy(w.astype('float32')) \
                                  for w in [weights['lstm2'][wn] \
                                        for wn in weightnames]],0).split(ninput+1,1)
        weight_ih_l0_rev = ih_hh_splits[0]
        weight_hh_l0_rev = torch.cat(ih_hh_splits[1:], 1)
        
        # softmax
        weight_softm = torch.from_numpy(weights['softm']['W1'].astype('float32'))

        # attach weights
        self.rnn = TBIDILSTM(ninput, nhidden, noutput)
        self.rnn.rnn.weight_ih_l0 = nn.Parameter(weight_ih_l0)
        self.rnn.rnn.weight_hh_l0 = nn.Parameter(weight_hh_l0)
        self.rnn.rnn.weight_ih_l0_reverse = nn.Parameter(weight_ih_l0_rev)
        self.rnn.rnn.weight_hh_l0_reverse = nn.Parameter(weight_hh_l0_rev)
        self.rnn.decoder.weight = nn.Parameter(weight_softm)
        
        self.setLearningRate(lrate, momentum)        
        self.rnn.zero_grad()
        
        self.criterion = CTCLoss()
        
        if self.cuda_available:
            self.cuda()
        
    def translate_back(self, output):
        _, preds = output.cpu().max(2) # max() outputs values +1 when on gpu. why?
        dec = preds.squeeze(2).transpose(1,0).contiguous().view(-1).data
        char_list = []
        for i in range(len(dec)):
            if dec[i] != 0 and (not (i > 0 and dec[i-1] == dec[i])):
                char_list.append(dec[i])
        return char_list

    def predictString(self, line):
        line = Variable(torch.from_numpy(line.reshape(-1, 1, self.rnn.ninput).astype('float32')))
        
        if self.cuda_available:
            line = line.cuda()
        
        out, _ = self.rnn.forward(line, self.rnn.init_hidden())

        codes = self.translate_back(out)
        #codes = lstm.translate_back(out.exp().cpu().squeeze().data.numpy())
        res = ''.join(self.codec.decode(codes))
        return res.strip()
    
    def trainSequence(self, line, labels, update=1):        
        line = Variable(torch.from_numpy(line.reshape(-1, 1, self.rnn.ninput).astype('float32')), requires_grad=True)
        
        if self.cuda_available:
            line = line.cuda()
            
        if not hasattr(self, 'hidden'):
            self.hidden = self.rnn.init_hidden()
        
        # repackage hidden
        self.hidden = tuple(Variable(h.data) for h in self.hidden)
        
        out, self.hidden = self.rnn.forward(line, self.hidden)
        
        tlabels = Variable(torch.IntTensor(labels))
        probs_sizes = Variable(torch.IntTensor([len(out)])) # why Variable?
        label_sizes = Variable(torch.IntTensor([len(labels)]))
        loss = self.criterion(out, tlabels, probs_sizes, label_sizes)
        
        self.rnn.zero_grad()

        loss.backward()
        
        if update:
            self.optim.step()
            self.trial += 1
            
        cls = self.translate_back(out)
        return cls
    
    def trainString(self, line, s, update=1):
        labels = self.codec.encode(s)
        cls = self.trainSequence(line, labels)
        return ''.join(self.codec.decode(cls)).strip()
    
    def setLearningRate(self, rate=1e-4, momentum=0.9):
        self.rnn.learning_rate = rate
        self.rnn.momentum = momentum
        self.optim = torch.optim.RMSprop(self.rnn.parameters(), lr=self.rnn.learning_rate, momentum=self.rnn.momentum)
        
        
