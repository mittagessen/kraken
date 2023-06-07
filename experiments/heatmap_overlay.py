#! /usr/bin/env python
"""
Produces semi-transparent neural segmenter output overlays
"""
import click
import cv2
import numpy as np
from PIL import ImageOps


# @click.command()
# # @click.option('-i', '--model', default=None, show_default=True, type=click.Path(exists=True),
# #               help='Baseline detection model to use.')
# @click.argument('files', nargs=-1)
def cli(img):
    """
    Applies a BLLA baseline segmentation model and outputs the raw heatmaps of
    the first baseline class.
    """
    import torch
    from PIL import Image
    from kraken.lib import vgsl, dataset
    import torch.nn.functional as F
    from os.path import splitext
    import torchvision.transforms as tf

    model = vgsl.TorchVGSLModel.load_model('blla.mlmodel')
    model.eval()
    batch, channels, height, width = model.input

    transforms = dataset.ImageInputTransforms(batch, height, width, channels, 0, valid_norm=False)

    torch.set_num_threads(1)

    # for img in files:
    print(img)
    im = Image.open(img)
    xs = transforms(im)

    with torch.no_grad():
        o, _ = model.nn(xs.unsqueeze(0))
        o = F.interpolate(o, size=xs.shape[1:])
        o = o.squeeze().numpy()

    scal_im = tf.ToPILImage()(1-xs)
    heat = ImageOps.exif_transpose(Image.fromarray((o[2]*255).astype('uint8')))
    # image_array = np.array(heat)

    print(heat)
    # cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    # cv2.imshow('img', image_array)
    # cv2.waitKey(0)
    # Image.open(heat)

    heat.show()

    # heat.save(splitext(img)[0] + '.heat.png')
    # overlay = Image.new('RGBA', scal_im.size, (0, 130, 200, 255))
    # bl = Image.composite(overlay, scal_im.convert('RGBA'), heat)
    # heat = Image.fromarray((o[1]*255).astype('uint8'))
    # overlay = Image.new('RGBA', scal_im.size, (230, 25, 75, 255))
    # bl = Image.composite(overlay, bl, heat)
    # heat = Image.fromarray((o[0]*255).astype('uint8'))
    # overlay = Image.new('RGBA', scal_im.size, (60, 180, 75, 255))
    # Image.composite(overlay, bl, heat).save(splitext(img)[0] + '.overlay.png')
    # del o
    # del im


if __name__ == '__main__':
    cli("/home/dell/Documents/handwritten_images/testingimages/d5.jpg")
