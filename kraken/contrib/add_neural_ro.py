#!/usr/bin/env python
"""
Computes an additional reading order from a neural RO model and adds it to an
ALTO document.
"""
import click

@click.command()
@click.option('-f', '--format-type', type=click.Choice(['alto']), default='alto',
              help='Sets the input document format. In ALTO and PageXML mode all '
              'data is extracted from xml files containing both baselines, polygons, and a '
              'link to source images.')
@click.option('-i', '--model', default=None, show_default=True, type=click.Path(exists=True),
              help='Baseline detection model to use. Overrides format type and expects image files as input.')
@click.argument('files', nargs=-1)
def cli(format_type, model, files):
    """
    A script adding new neural reading orders to the input documents.
    """
    if len(files) == 0:
        ctx = click.get_current_context()
        click.echo(ctx.get_help())
        ctx.exit()

    import uuid

    from kraken import blla
    from kraken.lib import segmentation, vgsl, xml

    from lxml import etree
    from dataclasses import asdict

    try:
        net = vgsl.TorchVGSLModel.load_model(model)
        ro_class_mapping = net.user_metadata['ro_class_mapping']
        ro_net = net.aux_layers['ro_model']
    except:
        from kraken.lib.ro import ROModel
        net = ROModel.load_from_checkpoint(model)
        ro_class_mapping = net.class_mapping
        ro_model = net.ro_net

    for doc in files:
        click.echo(f'Processing {doc} ', nl=False)
        doc = xml.XMLPage(doc)
        if doc.filetype != 'alto':
            click.echo(f'Not an ALTO file. Skipping.')
            continue
        seg = doc.to_container()
        lines = list(map(asdict, seg.lines))
        _order = segmentation.neural_reading_order(lines=lines,
                                                   regions=seg.regions,
                                                   model=ro_model,
                                                   im_size=doc.image_size[::-1],
                                                   class_mapping=ro_class_mapping)
        # reorder 
        lines = [lines[idx] for idx in _order]
        # add ReadingOrder block to ALTO
        tree = etree.parse(doc.filename)
        alto = tree.getroot()
        if alto.find('./{*}ReadingOrder'):
            click.echo(f'Addition to files with explicit reading order not yet supported. Skipping.')
            continue
        ro = etree.Element('ReadingOrder')
        og = etree.SubElement(ro, 'OrderedGroup')
        og.set('ID', f'_{uuid.uuid4()}')
        for line in lines:
            el = etree.SubElement(og, 'ElementRef')
            el.set('ID', f'_{uuid.uuid4()}')
            el.set('REF', f'{line["id"]}')
        tree.find('.//{*}Layout').addprevious(ro)
        with open(doc.filename.with_suffix('.ro.xml'), 'wb') as fo:
            fo.write(etree.tostring(tree, encoding='UTF-8', xml_declaration=True, pretty_print=True))
        click.secho('\u2713', fg='green')


if __name__ == '__main__':
    cli()
