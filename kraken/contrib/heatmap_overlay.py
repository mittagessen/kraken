#! /usr/bin/env python
"""
Produces semi-transparent neural segmenter output overlays
"""

import sys
import torch
from PIL import Image
from kraken.lib import vgsl, dataset
import torch.nn.functional as F
from os.path import splitext

model = vgsl.TorchVGSLModel.load_model(sys.argv[1])
model.eval()
batch, channels, height, width = model.input

transforms = dataset.generate_input_transforms(batch, height, width, channels, 0, valid_norm=False)

imgs = sys.argv[2:]
torch.set_num_threads(1)

for img in imgs:
    print(img)
    im = Image.open(img)
    with torch.no_grad():
        o = model.nn(transforms(im).unsqueeze(0))
        o = F.interpolate(o, size=im.size[::-1])
        o = o.squeeze().numpy()
    heat = Image.fromarray((o[1]*255).astype('uint8'))
    heat.save(splitext(img)[0] + '.heat.png')
    overlay = Image.new('RGBA', im.size, (0, 130, 200, 255))
    Image.composite(overlay, im.convert('RGBA'), heat).save(splitext(img)[0] + '.overlay.png')
    del o
    del im
