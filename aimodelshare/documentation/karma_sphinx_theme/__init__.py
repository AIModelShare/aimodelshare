import os

from karma_sphinx_theme import _version as version


def get_path():
    """
    Shortcut for users whose theme is next to their conf.py.
    """
    # Theme directory is defined as our parent directory
    return os.path.abspath(os.path.dirname(os.path.dirname(__file__)))


def update_context(app, pagename, templatename, context, doctree):
    context['karma_sphinx_theme_version'] = version.__version__

def setup(app):
    """ add_html_theme is new in Sphinx 1.6+ """

    if hasattr(app, 'add_html_theme'):
        theme_path = os.path.abspath(os.path.dirname(__file__))
        app.add_html_theme('karma_sphinx_theme', theme_path)
    app.connect('html-page-context', update_context)

    return {
        'version': version.__version__,
        'parallel_read_safe': True
    }
