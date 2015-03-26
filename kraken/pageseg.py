
from __future__ import absolute_import

import numpy as np

from scipy.ndimage.filters import gaussian_filter,uniform_filter,maximum_filter,minimum_filter
from kraken.lib import morph, sl

class record:
    def __init__(self,**kw): self.__dict__.update(kw)

def pil2array(im,alpha=0):
    if im.mode=="L":
        a = np.fromstring(im.tostring(),'B')
        a.shape = im.size[1],im.size[0]
        return a
    if im.mode=="RGB":
        a = np.fromstring(im.tostring(),'B')
        a.shape = im.size[1],im.size[0],3
        return a
    if im.mode=="RGBA":
        a = np.fromstring(im.tostring(),'B')
        a.shape = im.size[1],im.size[0],4
        if not alpha: a = a[:,:,:3]
        return a
    return pil2array(im.convert("L"))

def find(condition):
    "Return the indices where ravel(condition) is true"
    res, = np.nonzero(np.ravel(condition))
    return res

def binary_objects(binary):
    labels,n = morph.label(binary)
    objects = morph.find_objects(labels)
    return objects

def estimate_scale(binary):
    objects = binary_objects(binary)
    bysize = sorted(objects,key=sl.area)
    scalemap = np.zeros(binary.shape)
    for o in bysize:
        if np.amax(scalemap[o])>0: continue
        scalemap[o] = sl.area(o)**0.5
    scale = np.median(scalemap[(scalemap>3)&(scalemap<100)])
    return scale

def compute_boxmap(binary,scale,threshold=(.5,4),dtype='i'):
    objects = binary_objects(binary)
    bysize = sorted(objects,key=sl.area)
    boxmap = np.zeros(binary.shape,dtype)
    for o in bysize:
        if sl.area(o)**.5<threshold[0]*scale: continue
        if sl.area(o)**.5>threshold[1]*scale: continue
        boxmap[o] = 1
    return boxmap

def compute_lines(segmentation,scale):
    """Given a line segmentation map, computes a list
    of tuples consisting of 2D slices and masked images."""
    lobjects = morph.find_objects(segmentation)
    lines = []
    for i,o in enumerate(lobjects):
        if o is None: continue
        if sl.dim1(o)<2*scale or sl.dim0(o)<scale: continue
        mask = (segmentation[o]==i+1)
        if np.amax(mask)==0: continue
        result = record()
        result.label = i+1
        result.bounds = o
        result.mask = mask
        lines.append(result)
    return lines

def reading_order(lines):
    """Given the list of lines (a list of 2D slices), computes
    the partial reading order.  The output is a binary 2D array
    such that order[i,j] is true if line i comes before line j
    in reading order."""
    order = np.zeros((len(lines),len(lines)),'B')
    def x_overlaps(u,v):
        return u[1].start<v[1].stop and u[1].stop>v[1].start
    def above(u,v):
        return u[0].start<v[0].start
    def left_of(u,v):
        return u[1].stop<v[1].start
    def separates(w,u,v):
        if w[0].stop<min(u[0].start,v[0].start): return 0
        if w[0].start>max(u[0].stop,v[0].stop): return 0
        if w[1].start<u[1].stop and w[1].stop>v[1].start: return 1
    for i,u in enumerate(lines):
        for j,v in enumerate(lines):
            if x_overlaps(u,v):
                if above(u,v):
                    order[i,j] = 1
            else:
                if [w for w in lines if separates(w,u,v)]==[]:
                    if left_of(u,v): order[i,j] = 1
    return order

def topsort(order):
    """Given a binary array defining a partial order (o[i,j]==True means i<j),
    compute a topological sort.  This is a quick and dirty implementation
    that works for up to a few thousand elements."""
    n = len(order)
    visited = np.zeros(n)
    L = []
    def visit(k):
        if visited[k]: return
        visited[k] = 1
        a, = np.nonzero(np.ravel(order[:,k]))
        for l in a:
            visit(l)
        L.append(k)
    for k in range(n):
        visit(k)
    return L #[::-1]

def compute_separators_morph(binary,scale):
    """Finds vertical black lines corresponding to column separators."""
    d0 = int(max(5,scale/4))
    d1 = int(max(5,scale))+args.sepwiden
    thick = morph.r_dilation(binary,(d0,d1))
    vert = morph.rb_opening(thick,(10*scale,1))
    vert = morph.r_erosion(vert,(d0//2,args.sepwiden))
    vert = morph.select_regions(vert,sl.dim1,min=3,nbest=2*args.maxseps)
    vert = morph.select_regions(vert,sl.dim0,min=20*scale,nbest=args.maxseps)
    return vert

def compute_colseps_conv(binary, scale=1.0, minheight=10, maxcolseps=2):
    """Find column separators by convolution and thresholding.
    
    Args:
        binary (numpy.array):
        scale (float):
        minheight (int):
        maxcolseps (int):

    Returns:
        Separators
    """

    h, w = binary.shape
    # find vertical whitespace by thresholding
    smoothed = gaussian_filter(1.0*binary,(scale,scale*0.5))
    smoothed = uniform_filter(smoothed,(5.0*scale,1))
    thresh = (smoothed < np.amax(smoothed)*0.1)
    # find column edges by filtering
    grad = gaussian_filter(1.0*binary,(scale,scale*0.5),order=(0,1))
    grad = uniform_filter(grad,(10.0*scale,1))
    grad = (grad>0.5*np.amax(grad))
    # combine edges and whitespace
    seps = np.minimum(thresh,maximum_filter(grad,(int(scale),int(5*scale))))
    seps = maximum_filter(seps,(int(2*scale),1))
    # select only the biggest column separators
    seps = morph.select_regions(seps,sl.dim0,min=minheight*scale,nbest=maxcolseps+1)
    return seps

def compute_black_colseps(binary, scale):
    """
    Computes column separators from vertical black lines.

    Args:
        binary (numpy.array): Numpy array of the binary image
        scale (float):

    Returns:
        (colseps, binary): 
    """
    seps = compute_separators_morph(binary, scale)
    colseps = np.maximum(compute_colseps_conv(binary, scale), seps)
    binary = np.minimum(binary, 1-seps)
    return colseps, binary

def compute_white_colseps(binary, scale):
    """
    Computes column separators either from vertical black lines or whitespace.
    
    Args:
        binary (numpy.array): Numpy array of the binary image
        scale (float):

    Returns:
        colseps: 
    """
    return compute_colseps_conv(binary, scale)

def norm_max(v):
    return v/np.amax(v)

def compute_gradmaps(binary, scale, gauss=False):
    """
    Use gradient filtering to find baselines

    Args:
        binary (numpy.array):
        scale (float):
        gauss (bool): Use gaussian instead of uniform filtering

    Returns:
        (bottom, top, boxmap)
    """
    # use gradient filtering to find baselines
    boxmap = compute_boxmap(binary,scale)
    cleaned = boxmap*binary
    if gauss:
        grad = gaussian_filter(1.0*cleaned,(0.3*scale, 6*scale),order=(1,0))
    else:
        grad = gaussian_filter(1.0*cleaned,(max(4,0.3*scale),
                                            scale),order=(1,0))
        grad = uniform_filter(grad,(1,6*scale))
    bottom = norm_max((grad<0)*(-grad))
    top = norm_max((grad>0)*grad)
    return bottom,top,boxmap

def compute_line_seeds(binary, bottom, top, colseps, scale, threshold=0.2):
    """
    Base on gradient maps, computes candidates for baselines and xheights.
    Then, it marks the regions between the two as a line seed.
    """
    vrange = int(scale)
    bmarked = maximum_filter(bottom==maximum_filter(bottom,(vrange,0)),(2,2))
    bmarked *= (bottom>threshold*np.amax(bottom)*threshold)*(1-colseps)
    tmarked = maximum_filter(top==maximum_filter(top,(vrange,0)),(2,2))
    tmarked *= (top>threshold*np.amax(top)*threshold/2)*(1-colseps)
    tmarked = maximum_filter(tmarked,(1,20))
    seeds = np.zeros(binary.shape,'i')
    delta = max(3,int(scale/2))
    for x in range(bmarked.shape[1]):
        transitions = sorted([(y,1) for y in find(bmarked[:,x])]+[(y,0) for y in find(tmarked[:,x])])[::-1]
        transitions += [(0,0)]
        for l in range(len(transitions)-1):
            y0,s0 = transitions[l]
            if s0==0: continue
            seeds[y0-delta:y0,x] = 1
            y1,s1 = transitions[l+1]
            if s1==0 and (y0-y1)<5*scale: seeds[y1:y0,x] = 1
    seeds = maximum_filter(seeds,(1,int(1+scale)))
    seeds *= (1-colseps)
    seeds,_ = morph.label(seeds)
    return seeds

def remove_hlines(binary, scale, maxsize=10):
    """
    Removes horizontal black lines that only interfere with page segmentation.

        Args:
            binary (numpy.array):
            scale (float):
            maxsize (int): maximum size of removed lines

        Returns:
            numpy.array containing the filtered image.

    """
    labels, _ = morph.label(binary)
    objects = morph.find_objects(labels)
    for i,b in enumerate(objects):
        if sl.width(b)>maxsize*scale:
            labels[b][labels[b]==i+1] = 0
    return np.array(labels!=0,'B')

def segment(im, scale=None, black_colseps=False):
    """
    Segments a page into text lines.

    Segments a page into text lines and returns the absolute coordinates of
    each line in reading order.

    Args:
        im (PIL.Image): A page of mode '1'
        scale (float): Scale of the image
        black_colseps (bool): Whether column separators are assumed to be
                              vertical black lines or not

    Returns:
        
    """

    # honestly I've got no idea what's going on here. In theory a simple
    # np.array(im, 'i') should suffice here but for some reason the
    # tostring/fromstring magic in pil2array alters the array in a way that is
    # needed for the algorithm to work correctly.
    a = pil2array(im)
    binary = np.array(a>0.5*(np.amin(a)+np.amax(a)), 'i')
    binary = 1 - binary

    if not scale:
        scale = estimate_scale(binary)

    binary = remove_hlines(binary, scale)
    if black_colseps:
        colseps, binary = compute_black_colseps(binary, scale)
    else:
        colseps = compute_white_colseps(binary, scale)
    bottom, top, boxmap = compute_gradmaps(binary, scale)
    seeds = compute_line_seeds(binary, bottom, top, colseps, scale)
    llabels = morph.propagate_labels(boxmap,seeds,conflict=0)
    spread = morph.spread_labels(seeds,maxdist=scale)
    llabels = np.where(llabels>0,llabels,spread*binary)
    segmentation = llabels*binary

    lines = compute_lines(segmentation, scale)
    order = reading_order([l.bounds for l in lines])
    lsort = topsort(order)
    lines = [lines[i].bounds for i in lsort]
    return [(s2.start, s1.start, s2.stop, s1.stop) for s1, s2 in lines]
