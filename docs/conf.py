import datetime
import pycave

project = 'PyCave'
# version = pycave.__version__ # pylint: disable=no-member
# release = pycave.__version__ # pylint: disable=no-member

author = 'Oliver Borchert'
copyright = f'{datetime.datetime.now().year}, {author}' # pylint: disable=redefined-builtin

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx_rtd_theme'
]
templates_path = []
exclude_patterns = []

html_theme = 'sphinx_rtd_theme'
html_static_path = []
