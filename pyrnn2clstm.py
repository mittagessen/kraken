#!/usr/bin/python

import h5py
import numpy
import json
import argparse

parser = argparse.ArgumentParser("convert pyrnn to clstm")
parser.add_argument("file")

parser.add_argument("-o","--out",default="en-default.hdf5", 
	help="Filename to export the parameters")

args = parser.parse_args()


recognizer = ocrolib.load_object(file)
# TODO: load gzipped things
# recognizer = pickle.load(open('en-default.pyrnn'))

parallel, softmax = recognizer.lstm.nets
fwdnet, revnet = parallel.nets

nf = h5py.File(args.out, "w")

for w in "WGI WGF WGO WCI".split():
    print getattr(fwdnet, w).shape
    dset = nf.create_dataset(".bidilstm.0.parallel.0.lstm." + w, getattr(fwdnet, w).shape, dtype='f')
    dset[...] = getattr(fwdnet, w)
    dset = nf.create_dataset(".bidilstm.0.parallel.1.reversed.0.lstm." + w, getattr(revnet.net, w).shape, dtype='f')
    dset[...] = getattr(revnet.net, w)

for w in "WIP WFP WOP".split():

    data = getattr(fwdnet, w).reshape((-1, 1))
    print data.shape
    dset = nf.create_dataset(".bidilstm.0.parallel.0.lstm." + w, data.shape, dtype='f')
    dset[...] = data
    
    data = getattr(revnet.net, w).reshape((-1, 1))
    dset = nf.create_dataset(".bidilstm.0.parallel.1.reversed.0.lstm." + w, data.shape, dtype='f')
    dset[...] = data

print softmax.W2[:,0].shape
dset = nf.create_dataset(".bidilstm.1.softmax.w", (softmax.W2[:,0].shape[0], 1), dtype='f')
dset[:] = softmax.W2[:,0].reshape((-1, 1))

dset = nf.create_dataset(".bidilstm.1.softmax.W", softmax.W2[:,1:].shape, dtype='f')
dset[:] = softmax.W2[:,1:]

codec = numpy.array(range(recognizer.codec.size()), dtype='f').reshape((-1, 1))
dset = nf.create_dataset("codec", codec.shape, dtype='f')
dset[:] = codec

nf.close()