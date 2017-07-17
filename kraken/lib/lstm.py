from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals
from builtins import zip
from builtins import range
from builtins import object

import unicodedata
import numpy as np

from scipy.ndimage import measurements
from scipy.special import expit

initial_range = 0.1


class Codec(object):
    """Translate between integer codes and characters."""
    def init(self, charset):
        charset = sorted(list(set(charset)))
        self.code2char = {}
        self.char2code = {}
        for code,char in enumerate(charset):
            self.code2char[code] = char
            self.char2code[char] = code
        return self
    def size(self):
        """The total number of codes (use this for the number of output
        classes when training a classifier."""
        return len(list(self.code2char.keys()))
    def encode(self, s):
        "Encode the string `s` into a code sequence."
        tab = self.char2code
        dflt = self.char2code["~"]
        return [self.char2code.get(c,dflt) for c in s]
    def decode(self, l):
        "Decode a code sequence into a string."
        s = [self.code2char.get(c,"~") for c in l]
        return s


def normalize_nfkc(s):
    return unicodedata.normalize('NFKC',s)


def prepare_line(line, pad=16):
    """Prepare a line for recognition; this inverts it, transposes
    it, and pads it."""
    line = line * 1.0/np.amax(line)
    line = np.amax(line)-line
    line = line.T
    if pad>0:
        w = line.shape[1]
        line = np.vstack([np.zeros((pad,w)),line,np.zeros((pad,w))])
    return line


def randu(*shape):
    # ATTENTION: whether you use randu or randn can make a difference.
    """Generate uniformly random values in the range (-1,1).
    This can usually be used as a drop-in replacement for `randn`
    resulting in a different distribution."""
    return 2*np.random.rand(*shape)-1


def sigmoid(x):
    """
    Compute the sigmoid function. We don't bother with clipping the input
    value because IEEE floating point behaves reasonably with this function
    even for infinities. 
    
    Further we use scipy's expit function which is ~50% faster for decently
    sized arrays.
    """
    return expit(x)

# These are the nonlinearities used by the LSTM network.
# We don't bother parameterizing them here

def ffunc(x):
    "Nonlinearity used for gates."
    return 1.0/(1.0+np.exp(-x))
def gfunc(x):
    "Nonlinearity used for input to state."
    return np.tanh(x)
# ATTENTION: try linear for hfunc
def hfunc(x):
    "Nonlinearity used for output."
    return np.tanh(x)

################################################################
# LSTM classification with forward/backward alignment ("CTC")
################################################################

def translate_back(outputs, threshold=0.5, pos=0):
    """Translate back. Thresholds on class 0, then assigns
    the maximum class to each region."""
    labels, n = measurements.label(outputs[:,0] < threshold)
    mask = np.tile(labels.reshape(-1,1), (1,outputs.shape[1]))
    maxima = measurements.maximum_position(outputs, mask, np.arange(1, np.amax(mask)+1))
    if pos: return maxima
    return [c for (r,c) in maxima if c != 0]

def translate_back_locations(outputs, threshold=0.5):
    """
    Translates back the network output to a class sequence.

    Thresholds on class 0, then assigns the maximum (non-zero) class to each
    region. Difference to translate_back is the output region not just the
    maximum's position is returned.

    Args:

    Returns:
        A list with tuples (class, start, end, max). max is the maximum value
        of the softmax layer in the region.
    """
    labels, n = measurements.label(outputs[:,0] < threshold)
    mask = np.tile(labels.reshape(-1,1), (1,outputs.shape[1]))
    maxima = measurements.maximum_position(outputs, mask, np.arange(1, np.amax(mask)+1))
    p = 0
    start = None
    x = []
    for idx, val in enumerate(labels):
        if val != 0 and start is None:
            start = idx
            p += 1
        if val == 0 and start is not None:
            if maxima[p-1][1] == 0:
                start = None
            else:
                x.append((maxima[p-1][1], start, idx, outputs[maxima[p-1]]))
                start = None
    # append last non-zero region to list of no zero region occurs after it
    if start:
        x.append((maxima[p-1][1], start, len(outputs), outputs[maxima[p-1]]))
    return x


class Network:
    def predict(self,xs):
        """Prediction is the same as forward propagation."""
        return self.forward(xs)

class Softmax(Network):
    """A logistic regression network."""
    def __init__(self,Nh,No,initial_range=initial_range,rand=np.random.rand):
        self.Nh = Nh
        self.No = No
        self.W2 = randu(No,Nh+1)*initial_range
        self.DW2 = np.zeros((No,Nh+1))
    def ninputs(self):
        return self.Nh
    def noutputs(self):
        return self.No
    def forward(self,ys):
        n = len(ys)
        inputs,zs = [None]*n,[None]*n
        for i in range(n):
            inputs[i] = np.concatenate([np.ones(1),ys[i]])
            temp = np.dot(self.W2,inputs[i])
            temp = np.exp(np.clip(temp,-100,100))
            temp /= np.sum(temp)
            zs[i] = temp
        self.state = (inputs,zs)
        return zs
    def backward(self,deltas):
        inputs,zs = self.state
        n = len(zs)
        assert len(deltas)==len(inputs)
        dzspre,dys = [None]*n,[None]*n
        for i in reversed(list(range(len(zs)))):
            dzspre[i] = deltas[i]
            dys[i] = np.dot(dzspre[i],self.W2)[1:]
        self.DW2 = sumouter(dzspre,inputs)
        return dys
    def info(self):
        vars = sorted("W2".split())
        for v in vars:
            a = np.array(getattr(self,v))
            print(v, a.shape, np.amin(a), np.amax(a))
    def weights(self):
        yield self.W2,self.DW2,"Softmax"


class LSTM(Network):
    """A standard LSTM network. This is a direct implementation of all the forward
    and backward propagation formulas, mainly for speed. (There is another, more
    abstract implementation as well, but that's significantly slower in Python
    due to function call overhead.)"""
    def __init__(self,ni,ns,initial=initial_range,maxlen=5000):
        na = 1+ni+ns
        self.dims = ni,ns,na
        self.init_weights(initial)
        self.allocate(maxlen)
    def init_weights(self,initial):
        "Initialize the weight matrices and derivatives"
        ni,ns,na = self.dims
        # gate weights
        for w in "WGI WGF WGO WCI".split():
            setattr(self,w,randu(ns,na)*initial)
            setattr(self,"D"+w,np.zeros((ns,na)))
        # peep weights
        for w in "WIP WFP WOP".split():
            setattr(self,w,randu(ns)*initial)
            setattr(self,"D"+w,np.zeros(ns))
    def allocate(self,n):
        """Allocate space for the internal state variables.
        `n` is the maximum sequence length that can be processed."""
        ni,ns,na = self.dims
        vars = "cix ci gix gi gox go gfx gf"
        vars += " state output gierr gferr goerr cierr stateerr outerr"
        for v in vars.split():
            setattr(self,v,np.nan*np.ones((n,ns)))
        self.source = np.nan*np.ones((n,na))
        self.sourceerr = np.nan*np.ones((n,na))
    def reset(self,n):
        """Reset the contents of the internal state variables to `nan`"""
        vars = "cix ci gix gi gox go gfx gf"
        vars += " state output gierr gferr goerr cierr stateerr outerr"
        vars += " source sourceerr"
        for v in vars.split():
            getattr(self,v)[:,:] = np.nan
    def forward(self,xs):
        """Perform forward propagation of activations."""
        ni,ns,na = self.dims
        assert len(xs[0])==ni
        n = len(xs)
        # grow internal state arrays if len(xs) > maxlen
        if n > self.gi.shape[0]:
            self.allocate(n)
        self.last_n = n
        self.reset(n)
        for t in range(n):
            prev = np.zeros(ns) if t==0 else self.output[t-1]
            self.source[t,0] = 1
            self.source[t,1:1+ni] = xs[t]
            self.source[t,1+ni:] = prev
            np.dot(self.WGI,self.source[t],out=self.gix[t])
            np.dot(self.WGF,self.source[t],out=self.gfx[t])
            np.dot(self.WGO,self.source[t],out=self.gox[t])
            np.dot(self.WCI,self.source[t],out=self.cix[t])
            if t>0:
                # ATTENTION: peep weights are diagonal matrices
                self.gix[t] += self.WIP*self.state[t-1]
                self.gfx[t] += self.WFP*self.state[t-1]
            self.gi[t] = ffunc(self.gix[t])
            self.gf[t] = ffunc(self.gfx[t])
            self.ci[t] = gfunc(self.cix[t])
            self.state[t] = self.ci[t]*self.gi[t]
            if t>0:
                self.state[t] += self.gf[t]*self.state[t-1]
                self.gox[t] += self.WOP*self.state[t]
            self.go[t] = ffunc(self.gox[t])
            self.output[t] = hfunc(self.state[t]) * self.go[t]
        assert not np.isnan(self.output[:n]).any()
        return self.output[:n]

################################################################
# combination classifiers
################################################################

class Stacked(Network):
    """Stack two networks on top of each other."""
    def __init__(self,nets):
        self.nets = nets
    def forward(self,xs):
        for i,net in enumerate(self.nets):
            xs = net.forward(xs)
        return xs

class Reversed(Network):
    """Run a network on the time-reversed input."""
    def __init__(self,net):
        self.net = net
    def forward(self,xs):
        return self.net.forward(xs[::-1])[::-1]

class Parallel(Network):
    """Run multiple networks in parallel on the same input."""
    def __init__(self,*nets):
        self.nets = nets
    def forward(self,xs):
        outputs = [net.forward(xs) for net in self.nets]
        outputs = list(zip(*outputs))
        outputs = [np.concatenate(l) for l in outputs]
        return outputs

def BIDILSTM(Ni,Ns,No):
    """A bidirectional LSTM, constructed from regular and reversed LSTMs."""
    lstm1 = LSTM(Ni,Ns)
    lstm2 = Reversed(LSTM(Ni,Ns))
    bidi = Parallel(lstm1,lstm2)
    assert No>1
    logreg = Softmax(2*Ns,No)
    stacked = Stacked([bidi,logreg])
    return stacked


class SeqRecognizer(Network):
    """Perform sequence recognition using BIDILSTM and alignment."""
    def __init__(self,ninput,nstates,noutput=-1,codec=None,normalize=normalize_nfkc):
        self.Ni = ninput
        if codec: noutput = codec.size()
        assert noutput>0
        self.No = noutput
        self.lstm = BIDILSTM(ninput,nstates,noutput)
        self.normalize = normalize
        self.codec = codec
    def predictSequence(self,xs):
        "Predict an integer sequence of codes."
        assert xs.shape[1]==self.Ni,"wrong image height (image: %d, expected: %d)"%(xs.shape[1],self.Ni)
        self.outputs = np.array(self.lstm.forward(xs))
        return translate_back(self.outputs)
    def l2s(self,l):
        "Convert a code sequence into a unicode string after recognition."
        l = self.codec.decode(l)
        return u"".join(l)
    def predictString(self,xs):
        "Predict output as a string. This uses codec and normalizer."
        cs = self.predictSequence(xs)
        return self.l2s(cs)
