#! /usr/bin/env python
import click


@click.command()
@click.option('-f', '--format-type', type=click.Choice(['xml', 'alto', 'page', 'binary']), default='xml',
              help='Sets the input document format. In ALTO and PageXML mode all '
              'data is extracted from xml files containing both baselines, polygons, and a '
              'link to source images.')
@click.option('-i', '--model', default=None, show_default=True, type=click.Path(exists=True),
              help='Baseline detection model to use. Overrides format type and expects image files as input.')
@click.option('--legacy-polygons', is_flag=True, help='Use the legacy polygon extractor.')
@click.argument('files', nargs=-1)
def cli(format_type, model, legacy_polygons, files):
    """
    A small script extracting rectified line polygons as defined in either ALTO or
    PageXML files or run a model to do the same.
    """
    if len(files) == 0:
        ctx = click.get_current_context()
        click.echo(ctx.get_help())
        ctx.exit()

    import io
    import json
    from os.path import splitext

    import pyarrow as pa
    from PIL import Image

    from kraken import blla
    from kraken.lib import segmentation, vgsl, xml

    if model is None:
        for doc in files:
            click.echo(f'Processing {doc} ', nl=False)
            if format_type != 'binary':
                data = xml.XMLPage(doc, format_type)
                if len(data.lines) > 0:
                    bounds = data.to_container()
                    for idx, (im, box) in enumerate(segmentation.extract_polygons(Image.open(bounds.imagename), bounds, legacy=legacy_polygons)):
                        click.echo('.', nl=False)
                        im.save('{}.{}.jpg'.format(splitext(bounds.imagename)[0], idx))
                        with open('{}.{}.gt.txt'.format(splitext(bounds.imagename)[0], idx), 'w') as fp:
                            fp.write(box.text)
            else:
                with pa.memory_map(doc, 'rb') as source:
                    ds_table = pa.ipc.open_file(source).read_all()
                    raw_metadata = ds_table.schema.metadata
                    if not raw_metadata or b'lines' not in raw_metadata:
                        raise ValueError(f'{doc} does not contain a valid metadata record.')
                    metadata = json.loads(raw_metadata[b'lines'])
                    for idx in range(metadata['counts']['all']):
                        sample = ds_table.column('lines')[idx].as_py()
                        im = Image.open(io.BytesIO(sample['im']))
                        im.save('{}.{}.jpg'.format(splitext(doc)[0], idx))
                        with open('{}.{}.gt.txt'.format(splitext(doc)[0], idx), 'w') as fp:
                            fp.write(sample['text'])
            click.echo()
    else:
        net = vgsl.TorchVGSLModel.load_model(model)
        for doc in files:
            click.echo(f'Processing {doc} ', nl=False)
            full_im = Image.open(doc)
            bounds = blla.segment(full_im, model=net)
            for idx, (im, box) in enumerate(segmentation.extract_polygons(full_im, bounds, legacy=legacy_polygons)):
                click.echo('.', nl=False)
                im.save('{}.{}.jpg'.format(splitext(doc)[0], idx))
            click.echo()


if __name__ == '__main__':
    cli()
