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
@click.option('--baseline', 'topline', flag_value='baseline')
@click.option('--pad', show_default=True, type=(int, int), default=None,
              help='Padding (left/right, top/bottom) around the page image')
@click.option('--output-identifiers', type=click.Path(exists=True), help='Path '
              'to a json file containing a dict updating the string identifiers '
              'of line/region classes.')
@click.option('-f', '--format', 'output_format', type=click.Choice(['safetensors', 'coreml']),
              default='safetensors', show_default=True, help='Output model format')
@click.argument('model', nargs=1, type=click.Path(exists=True))
def cli(bounding_region, topline, pad, output_identifiers, output_format, model):
    """
    A script setting the metadata of segmentation models.
    """
    import json
    from pathlib import Path
    from kraken.models import load_models
    from kraken.models.writers import write_safetensors, write_coreml

    all_models = load_models(model)
    seg_models = [m for m in all_models if 'segmentation' in m.model_type]

    if not seg_models:
        print('No segmentation model found in file.')
        return

    for net in seg_models:
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
            net.user_metadata["bounding_regions"] = list(bounding_region)

        loc = {'topline': True,
               'baseline': False,
               'centerline': None}

        rloc = {True: 'topline',
                False: 'baseline',
                None: 'centerline'}
        line_loc = rloc[net.user_metadata.get('topline', False)]

        print(f'Model is {line_loc}')
        if topline is not None:
            print(f'-> Setting to {topline}')
            net.user_metadata['topline'] = loc[topline]

        hyper_params = net.user_metadata.get('hyper_params', {})
        print(f"Model has padding {hyper_params.get('padding', (0, 0))}")
        if pad is not None:
            print(f'-> Setting to {pad}')
            net.user_metadata.setdefault('hyper_params', {})['padding'] = pad

    path = Path(model)
    shutil.copy(path, path.with_suffix(path.suffix + '.bak'))
    path.unlink()
    writer = write_safetensors if output_format == 'safetensors' else write_coreml
    writer(all_models, path)


if __name__ == '__main__':
    cli()
