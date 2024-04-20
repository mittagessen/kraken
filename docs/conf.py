# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'kraken'
copyright = '2015-2024, Benjamin Kiessling'
author = 'Benjamin Kiessling'

from subprocess import Popen, PIPE
pipe = Popen('git describe --tags --always main', stdout=PIPE, shell=True)
release = pipe.stdout.read().decode('utf-8')

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autodoc.typehints',
    'autoapi.extension',
    'sphinx.ext.napoleon',
    'sphinx.ext.githubpages',
    'sphinx_multiversion',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

autodoc_typehints = 'description'

autoapi_type = 'python'
autoapi_dirs = ['../kraken']

autoapi_options = ['members',
                   'undoc-members',
                   #'private-members',
                   #'special-members',
                   'show-inheritance',
                   'show-module-summary',
                   #'imported-members',
                   ]
autoapi_generate_api_docs = False

source_suffix = '.rst'

master_doc = 'index'

language = 'en'

pygments_style = 'sphinx'
todo_include_todos = False


html_theme = 'alabaster'
html_theme_options = {
    'github_user': 'mittagessen',
    'github_repo': 'kraken',
}

html_static_path = ['_static']
html_css_files = [
    'custom.css',
]

html_sidebars = {
    'index':    ['sidebarintro.html', 'navigation.html', 'searchbox.html', 'versions.html'],
    '**':       ['localtoc.html', 'relations.html', 'searchbox.html', 'versions.html']
}

html_baseurl = 'kraken.re'
htmlhelp_basename = 'krakendoc'

smv_branch_whitelist = r'main'
smv_tag_whitelist = r'^[2-9]\.\d+(\.0)?$'
