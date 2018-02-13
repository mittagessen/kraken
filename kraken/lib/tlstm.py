"""
clstm in pytorch.

Inspiration drawn from:
https://github.com/tmbdev/ocropy
https://github.com/meijieru/crnn.pytorch
https://github.com/pytorch/examples/blob/master/word_language_model/main.py

"""


import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np

import kraken.lib.lstm
from kraken.lib.ctc import CTCCriterion
from kraken.lib import clstm_pb2

class TlstmSeqRecognizer(kraken.lib.lstm.SeqRecognizer):
    """
    Something like ClstmSeqRecognizer, using pytorch instead of clstm.
    The serialization format is the same as the clstm/master branch.
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
        self.mode = 'clstm'
        self.criterion = CTCCriterion()
        self.cuda_available = cuda
        if self.cuda_available:
            self.cuda()
        return self
    
    def cuda(self):
        if not self.cuda_available:
            return 'CUDA not available!'
        
        self.rnn = self.rnn.cuda()
        self.criterion = self.criterion.cuda()
    
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
            if self.mode == 'clstm_compatibility':
                self.mode = 'clstm'

            
        cls = self.translate_back(out)
        return cls
    
    def trainString(self, line, s, update=1):
        labels = self.codec.encode(s)
        cls = self.trainSequence(line, labels)
        return ''.join(self.codec.decode(cls))
    
    def setLearningRate(self, rate=1e-4, momentum=0.9):
        self.rnn.learning_rate = rate
        self.rnn.momentum = momentum
        self.optim = torch.optim.RMSprop(self.rnn.parameters(), lr=self.rnn.learning_rate, momentum=self.rnn.momentum)
