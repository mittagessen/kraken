"""
kraken.lib.models
~~~~~~~~~~~~~~~~~

Wraps around legacy pyrnn and HDF5 models to provide a single interface. In the
future it will also include support for clstm models.
"""

from __future__ import absolute_import
from __future__ import unicode_literals
from future import standard_library
standard_library.install_aliases()
from future.utils import PY2
from builtins import next
from builtins import chr

import h5py
import numpy
import gzip
import bz2
import sys

import kraken.lib.lstm
import kraken.lib.lineest

from kraken.lib.exceptions import KrakenInvalidModelException


def load_any(fname):
    """
    Loads anything that was, is, and will be a valid ocropus model and
    instantiates a shiny new kraken.lib.lstm.SeqRecognizer from the RNN
    configuration in the file.

    Currently it recognizes the following kinds of models:
        * pyrnn models containing BIDILSTMs
        * HDF5 models containing converted python BIDILSTMs

    Additionally an attribute 'kind' will be added to the SeqRecognizer
    containing a string representation of the source kind. Current known values
    are:
        * pyrnn for pickled BIDILSTMs
        * hdf-pyrnn for HDF models converted from pickled objects

    Args:
        fname (unicode): Path to the model

    Returns:
        A kraken.lib.lstm.SeqRecognizer object.

    Raises:
        KrakenInvalidModelException if the model file could not be recognized.
    """
    seq = None
    try:
        seq = load_hdf5(fname)
        seq.kind = 'hdf-pyrnn'
        return seq
    except:
        try:
            seq = load_pyrnn(fname)
            seq.kind = 'pyrnn'
            return seq
        except Exception as e:
            raise KrakenInvalidModelException(e.message)


def load_hdf5(fname, line_height=0):
    """
    Loads a model in HDF5 format and instantiates a
    kraken.lib.lstm.SeqRecognizer object.

    Args:
        fname (unicode): Path to the HDF5 file
        line_height (int): Target height of input. Will be extracted from model
        if set to 0.

    Returns:
        A kraken.lib.lstm.SeqRecognizer object
    """
    known_hdf5 = [b'pyrnn-bidi']
    with h5py.File(fname, 'r') as rnn:
        try:
            if rnn.attrs['kind'] not in known_hdf5:
                raise KrakenInvalidModelException(b'Unknown model kind ' +
                                                  rnn.attrs['kind'])
            # first extract the codec character set
            charset = [chr(x) for x in rnn.get('codec')]
            # code 0 is handled separately by the model
            charset[0] = ''
            codec = kraken.lib.lstm.Codec().init(charset)

            # get number of states from shape of first array
            hiddensize = rnn['.bidilstm.0.parallel.0.lstm.WGI'].shape[0]
            if not line_height:
                line_height = rnn['.bidilstm.0.parallel.0.lstm.WGI'].shape[1] - hiddensize - 1

            # next build a line estimator
            lnorm = kraken.lib.lineest.CenterNormalizer(line_height)
            network = kraken.lib.lstm.SeqRecognizer(lnorm.target_height,
                                                    hiddensize,
                                                    codec=codec,
                                                    normalize=kraken.lib.lstm.normalize_nfkc)
            parallel, softmax = network.lstm.nets
            nornet, rev = parallel.nets
            revnet = rev.net
            for w in ('WGI', 'WGF', 'WGO', 'WCI', 'WIP', 'WFP', 'WOP'):
                setattr(nornet, w, rnn['.bidilstm.0.parallel.0.lstm.' +
                        w][:].reshape(getattr(nornet, w).shape))
                setattr(revnet, w, rnn['.bidilstm.0.parallel.1.reversed.0.lstm.' +
                        w][:].reshape(getattr(nornet, w).shape))
            softmax.W2 = numpy.hstack((rnn['.bidilstm.1.softmax.w'][:],
                                       rnn['.bidilstm.1.softmax.W'][:]))
        except (KeyError, TypeError):
            raise KrakenInvalidModelException('Model incomplete')
        network.lnorm = lnorm
        return network


def load_pyrnn(fname):
    """
    Loads a legacy RNN from a pickle file.

    Args:
        fname (unicode): Path to the pickle object

    Returns:
        Unpickled object

    """

    if not PY2:
        raise KrakenInvalidModelException('Loading pickle models is not '
                                          'supported on python 3')
    import cPickle

    def find_global(mname, cname):
        aliases = {
            'lstm.lstm': kraken.lib.lstm,
            'ocrolib.lstm': kraken.lib.lstm,
            'ocrolib.lineest': kraken.lib.lineest,
        }
        if mname in aliases:
            return getattr(aliases[mname], cname)
        return getattr(sys.modules[mname], cname)

    of = open
    if fname.endswith(u'.gz'):
        of = gzip.open
    elif fname.endswith(u'.bz2'):
        of = bz2.BZ2File
    with of(fname, 'rb') as fp:
        unpickler = cPickle.Unpickler(fp)
        unpickler.find_global = find_global
        try:
            rnn = unpickler.load()
        except cPickle.UnpicklingError as e:
            raise KrakenInvalidModelException(str(e))
        if not isinstance(rnn, kraken.lib.lstm.SeqRecognizer):
            raise KrakenInvalidModelException('Pickle is %s instead of '
                                              'SeqRecognizer' %
                                              type(rnn).__name__)
        return rnn


def pyrnn_to_hdf5(pyrnn=None, output='en-default.hdf5'):
    """
    Converts a legacy python RNN to the new HDF5 format. Benefits of the new
    format include independence from particular python versions and no
    arbitrary code execution issues inherent in pickle.

    Args:
        pyrnn (kraken.lib.lstm.SegRecognizer): pyrnn model
        output (unicode): path of the converted HDF5 model
    """

    parallel, softmax = pyrnn.lstm.nets
    fwdnet, revnet = parallel.nets

    with h5py.File(output, 'w') as nf:
        # write metadata first
        nf.attrs['kind'] = 'pyrnn-bidi'

        for w in ('WGI', 'WGF', 'WGO', 'WCI'):
            dset = nf.create_dataset(".bidilstm.0.parallel.0.lstm." + w,
                                     getattr(fwdnet, w).shape, dtype='f')
            dset[...] = getattr(fwdnet, w)
            dset = nf.create_dataset(".bidilstm.0.parallel.1.reversed.0.lstm." + w,
                                     getattr(revnet.net, w).shape, dtype='f')
            dset[...] = getattr(revnet.net, w)

        for w in ('WIP', 'WFP', 'WOP'):
            data = getattr(fwdnet, w).reshape((-1, 1))
            dset = nf.create_dataset(".bidilstm.0.parallel.0.lstm." + w,
                                     data.shape, dtype='f')
            dset[...] = data

            data = getattr(revnet.net, w).reshape((-1, 1))
            dset = nf.create_dataset(".bidilstm.0.parallel.1.reversed.0.lstm." + w,
                                     data.shape, dtype='f')
            dset[...] = data

        dset = nf.create_dataset(".bidilstm.1.softmax.w",
                                 (softmax.W2[:, 0].shape[0], 1), dtype='f')
        dset[:] = softmax.W2[:, 0].reshape((-1, 1))

        dset = nf.create_dataset(".bidilstm.1.softmax.W", softmax.W2[:, 1:].shape,
                                 dtype='f')
        dset[:] = softmax.W2[:, 1:]
        cvals = iter(pyrnn.codec.code2char.values())
        next(cvals)
        codec = numpy.array([0]+[ord(x) for x in cvals], dtype='f').reshape((-1, 1))
        dset = nf.create_dataset("codec", codec.shape, dtype='f')
        dset[:] = codec
