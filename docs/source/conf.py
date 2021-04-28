import sys
import os
from pathlib import Path
from zipline import __version__ as version

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
sys.path.insert(0, Path(".").resolve(strict=True).as_posix())
sys.path.insert(0, Path("..").resolve(strict=True).as_posix())

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.doctest",
    "sphinx.ext.extlinks",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.todo",
    "sphinx.ext.napoleon",
    "m2r2",
    "sphinx_markdown_tables",
]

extlinks = {
    "issue": ("https://github.com/stefan-jansen/zipline/issues/%s", "#"),
    "commit": ("https://github.com/stefan-jansen/zipline/commit/%s", ""),
}

numpydoc_show_class_members = False

# Add any paths that contain templates here, relative to this directory.
templates_path = [".templates"]

# The suffix of source filenames.
# source_parsers = {'.md': CommonMarkParser}
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}

# The master toctree document.
master_doc = "index"

# General information about the project.
project = "Zipline"
copyright = "2020, Quantopian Inc."

# The full version, including alpha/beta/rc tags, but excluding the commit hash
version = release = version.split("+", 1)[0]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = []

on_rtd = os.environ.get("READTHEDOCS", None) == "True"
if not on_rtd:  # only import and set the theme if we're building docs locally
    try:
        import pydata_sphinx_theme
    except ImportError:
        html_theme = "default"
        html_theme_path = []
    else:
        html_theme = "pydata_sphinx_theme"
        html_theme_path = pydata_sphinx_theme.get_html_theme_path()

# The name of the Pygments (syntax highlighting) style to use.
highlight_language = "python"

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
html_favicon = os.path.join("..", "icons", "zipline.ico")

html_theme_options = {
    "github_url": "https://github.com/stefan-jansen/zipline-reloaded",
    "twitter_url": "https://twitter.com/ml4trading",
    "external_links": [
        {"name": "ML for Trading", "url": "https://ml4trading.io"},
        {"name": "Community", "url": "https://exchange.ml4trading.io"},
    ],
    "google_analytics_id": "UA-74956955-3",
}


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named 'default.css' will overwrite the builtin 'default.css'.
html_static_path = []

# If false, no index is generated.
html_use_index = True

# If true, 'Created using Sphinx' is shown in the HTML footer. Default is True.
html_show_sphinx = True

# If true, '(C) Copyright ...' is shown in the HTML footer. Default is True.
html_show_copyright = True

# Output file base name for HTML help builder.
htmlhelp_basename = "ziplinedoc"

intersphinx_mapping = {
    "https://docs.python.org/dev/": None,
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
}

doctest_global_setup = "import zipline"

todo_include_todos = True
