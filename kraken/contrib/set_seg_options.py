#!/usr/bin/env python
"""
A script setting the metadata of segmentation models.
"""
import shutil

import click


@click.command()
@click.option('-b', '--bounding-region', multiple=True, help='Sets region identifiers which bound line bounding polygons')
@click.option('--topline', 'topline',
              help='Sets model metadata baseline location to either `--topline`, `--centerline`, or `--baseline`',
              flag_value='topline',
              show_default=True)
@click.option('--centerline', 'topline', flag_value='centerline')
@click.option('--baseline', 'topline', flag_value='topline')
@click.option('--pad', show_default=True, type=(int, int), default=(0, 0),
              help='Padding (left/right, top/bottom) around the page image')
@click.option('--output-identifiers', type=click.Path(exists=True), help='Path '
              'to a json file containing a dict updating the string identifiers '
              'of line/region classes.')
@click.argument('model', nargs=1, type=click.Path(exists=True))
def cli(bounding_region, topline, pad, output_identifiers, model):
    """
    A script setting the metadata of segmentation models.
    """
    import json
    from kraken.lib import vgsl

    net = vgsl.TorchVGSLModel.load_model(model)
    if net.model_type != 'segmentation':
        print('Model is not a segmentation model.')
        return

    print('detectable line types:')
    for k, v in net.user_metadata['class_mapping']['baselines'].items():
        print(f'  {k}\t{v}')
    print('detectable region types:')
    for k, v in net.user_metadata['class_mapping']['regions'].items():
        print(f'  {k}\t{v}')

    if output_identifiers:
        with open(output_identifiers, 'r') as fp:
            new_cls_map = json.load(fp)
        print('-> Updating class maps')
        if 'baselines' in new_cls_map:
            print('new baseline identifiers:')
            old_cls = {v: k for k, v in net.user_metadata['class_mapping']['baselines'].items()}
            new_cls = {v: k for k, v in new_cls_map['baselines'].items()}
            old_cls.update(new_cls)
            net.user_metadata['class_mapping']['baselines'] = {v: k for k, v in old_cls.items()}
            for k, v in net.user_metadata['class_mapping']['baselines'].items():
                print(f'  {k}\t{v}')
        if 'regions' in new_cls_map:
            print('new region identifiers:')
            old_cls = {v: k for k, v in net.user_metadata['class_mapping']['regions'].items()}
            new_cls = {v: k for k, v in new_cls_map['regions'].items()}
            old_cls.update(new_cls)
            net.user_metadata['class_mapping']['regions'] = {v: k for k, v in old_cls.items()}
            for k, v in net.user_metadata['class_mapping']['regions'].items():
                print(f'  {k}\t{v}')

    print(f'existing bounding regions: {net.user_metadata["bounding_regions"]}')

    if bounding_region:
        br = set(net.user_metadata["bounding_regions"])
        br_new = set(bounding_region)

        print(f'-> removing: {br.difference(br_new)}')
        print(f'-> adding: {br_new.difference(br)}')
        net.user_metadata["bounding_regions"] = bounding_region

    loc = {'topline': True,
           'baseline': False,
           'centerline': None}

    rloc = {True: 'topline',
            False: 'baseline',
            None: 'centerline'}
    line_loc = rloc[net.user_metadata.get('topline', False)]

    print(f'Model is {line_loc}')
    print(f'-> Setting to {topline}')
    net.user_metadata['topline'] = loc[topline]

    print(f"Model has padding {net.user_metadata['hyper_params']['padding'] if 'padding' in net.user_metadata['hyper_params'] else (0, 0)}")
    print(f'-> Setting to {pad}')
    net.user_metadata['hyper_params']['padding'] = pad

    shutil.copy(model, f'{model}.bak')
    net.save_model(model)


if __name__ == '__main__':
    cli()
