#!/usr/bin/python

import h5py
import numpy
import json
import argparse

parser = argparse.ArgumentParser("apply an RNN recognizer")
parser.add_argument("file")

parser.add_argument("-o","--out",default="params.js", 
	help="Filename to export the parameters")

parser.add_argument("-d","--dense",action="store_true",
	help="Dense file format")

args = parser.parse_args()

f = h5py.File(args.file, "r")

js = open(args.out, 'w')

dictionary = []
gamut = 4.0

for i in range(32, 126):
	if i in [92, 34, 39, 60]:
		continue
	dictionary += chr(i)

print len(dictionary)

def encode(array):
	if isinstance(array[0], list):
		return [encode(el) for el in array]

	return ''.join([dictionary[int(round((float(len(dictionary) - 1)) * (max(-gamut, min(gamut, el)) + gamut) / (2.0 * gamut)))] for el in array])
		
def format(obj):	
	if args.dense != True:
		return json.dumps(obj)

	return json.dumps(encode(obj), separators=(',\n',':'))

js.write('encGamut = %f;\n\n' % gamut)

for w in "WGI WGF WGO WCI".split():
    js.write("fwd%s = %s;\n" % (w, format(f['.bidilstm.0.parallel.0.lstm.' + w][:].tolist())))
    js.write("rev%s = %s;\n\n" % (w, format(f['.bidilstm.0.parallel.1.reversed.0.lstm.' + w][:].tolist())))

for w in "WIP WFP WOP".split():
    js.write("fwd%s = %s;\n" % (w, format(f['.bidilstm.0.parallel.0.lstm.' + w][:][:,0].tolist())))
    js.write("rev%s = %s;\n\n" % (w, format(f['.bidilstm.0.parallel.1.reversed.0.lstm.' + w][:,0].tolist())))


js.write("softW  = %s;\n" % format(f['.bidilstm.1.softmax.W'][:].tolist()))
js.write("softB  = %s;\n\n" % format(f['.bidilstm.1.softmax.w'][:,0].tolist()))