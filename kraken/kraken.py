#!/usr/bin/env python

import numpy
import unicodedata
import argparse
import pickle

from PIL import Image

from kraken import segment
from kraken import rpred
from kraken import hocr

def main():
    parser = argparse.ArgumentParser(description=


    parser = argparse.ArgumentParser("apply an RNN recognizer")
    parser.add_argument('-m','--model',default="en-default.pyrnn.gz",
                        help="line recognition model")
    parser.add_argument("-q","--quiet",action="store_true",
                        help="turn off most output")
    parser.add_argument("file", help="Binarized input file")
    args = parser.parse_args()

# page segmentation
page = Image.open(fname)
page.convert('L')


# recognition
network = ocrolib.load_object(args.model,verbose=1)

with gzip.GzipFile(args.model, 'rb') as fp:
    network = pickle.load(fp)

for box in segmentation:
    line = numpy.fromstring(page.crop(box).tobytes(), dtype='uint8')
    line = line.T * 1/numpy.amax(line.T)
    line = line/255.0
    w = line.shape[1]
    line = numpy.vstack([numpy.zeros((16,w)),line,numpy.zeros((16,w))])
    
    pred = network.predictString(line)
    pred = unicodedata.normalize('NFD', pred)
    print(pred)
