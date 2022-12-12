# pylint: disable=all
from __future__ import annotations
import datetime
import os
import sys
from typing import Any

filepath = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(filepath, ".."))

# -------------------------------------------------------------------------------------------------
# BASICS

project = "PyCave"
copyright = f"{datetime.datetime.now().year}, Oliver Borchert"

# -------------------------------------------------------------------------------------------------
# PLUGINS

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "sphinx_automodapi.smart_resolver",
    "sphinx_copybutton",
]
if os.uname().machine != "arm64":
    extensions.append("sphinxcontrib.spelling")
templates_path = ["_templates"]

# -------------------------------------------------------------------------------------------------
# CONFIGURATION

html_theme = "pydata_sphinx_theme"
pygments_style = "lovelace"
html_theme_options = {
    "show_prev_next": False,
    "github_url": "https://github.com/borchero/pycave",
}
html_logo = "_static/logo.svg"
html_favicon = "_static/favicon.ico"
html_permalinks = True

autosummary_generate = True
autosummary_imported_members = True
autodoc_member_order = "groupwise"
autodoc_type_aliases = {
    "CovarianceType": ":class:`~pycave.bayes.core.CovarianceType`",
    "SequenceData": ":class:`~pycave.data.SequenceData`",
    "TabularData": ":class:`~pycave.data.TabularData`",
    "GaussianMixtureInitStrategy": ":class:`~pycave.bayes.gmm.types.GaussianMixtureInitStrategy`",
    "KMeansInitStrategy": ":class:`~pycave.clustering.kmeans.types.KMeansInitStrategy`",
}
autoclass_content = "both"

simplify_optional_unions = False

spelling_lang = "en_US"
spelling_word_list_filename = "spelling_wordlist.txt"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pytorch_lightning": ("https://pytorch-lightning.readthedocs.io/en/stable/", None),
}
