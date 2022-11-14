#! /usr/bin/env python
"""
Produces semi-transparent neural segmenter output overlays
"""
import click


@click.command()
@click.argument('files', nargs=-1)
def cli(files):

    import torch
    from PIL import Image
    from os.path import splitext
    import torchvision.transforms as tf
    from kraken.lib import dataset

    batch, channels, height, width = 1, 3, 1200, 0
    transforms = dataset.ImageInputTransforms(batch, height, width, channels, 0, valid_norm=False)

    torch.set_num_threads(1)

    ds = dataset.BaselineSet(files, im_transforms=transforms, mode='xml')

    for idx, batch in enumerate(ds):
        img = ds.imgs[idx]
        print(img)
        im = Image.open(img)
        res_tf = tf.Compose(transforms.transforms[:2])
        scal_im = res_tf(im)
        o = batch['target'].numpy()
        heat = Image.fromarray((o[ds.class_mapping['baselines']['default']]*255).astype('uint8'))
        heat.save(splitext(img)[0] + '.heat.png')
        overlay = Image.new('RGBA', scal_im.size, (0, 130, 200, 255))
        bl = Image.composite(overlay, scal_im.convert('RGBA'), heat)
        heat = Image.fromarray((o[ds.class_mapping['aux']['_start_separator']]*255).astype('uint8'))
        overlay = Image.new('RGBA', scal_im.size, (230, 25, 75, 255))
        bl = Image.composite(overlay, bl, heat)
        heat = Image.fromarray((o[ds.class_mapping['aux']['_end_separator']]*255).astype('uint8'))
        overlay = Image.new('RGBA', scal_im.size, (60, 180, 75, 255))
        Image.composite(overlay, bl, heat).save(splitext(img)[0] + '.overlay.png')
        del o
        del im


if __name__ == '__main__':
    cli()
