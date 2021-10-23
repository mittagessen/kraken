#!/usr/bin/env python
"""
A script setting the metadata of segmentation models.
"""
import click
import shutil

@click.command()
@click.option('-b', '--bounding-region', multiple=True, help='Sets region identifiers which bound line bounding polygons')
@click.option('--topline/--baseline', default=False, help='Sets model line type to baseline or topline')
@click.argument('model', nargs=1, type=click.Path(exists=True))
def cli(bounding_region, topline, model):
    """
    A script setting the metadata of segmentation models.
    """
    from PIL import Image, ImageDraw

    from kraken.lib import vgsl

    net = vgsl.TorchVGSLModel.load_model(model)
    if net.model_type != 'segmentation':
        print('Model is not a segmentation model.')
        return

    print('detectable line and region types:')
    for k, v in net.user_metadata['class_mapping']['baselines'].items():
        print(f'  {k}\t{v}')
    print('Training region types:')
    for k, v in net.user_metadata['class_mapping']['regions'].items():
        print(f'  {k}\t{v}')

    print(f'existing bounding regions: {net.user_metadata["bounding_regions"]}')

    if bounding_region:
        br = set(net.user_metadata["bounding_regions"])
        br_new = set(bounding_region)

        print(f'removing: {br.difference(br_new)}')
        print(f'adding: {br_new.difference(br)}')
        net.user_metadata["bounding_regions"] = bounding_region

    print(f'Model is {"topline" if "topline" in net.user_metadata and net.user_metadata["topline"] else "baseline"}')
    print(f'Setting to  {"topline" if topline else "baseline"}')
    net.user_metadata['topline'] = topline
    shutil.copy(model, f'{model}.bak')
    net.save_model(model)

if __name__ == '__main__':
    cli()
