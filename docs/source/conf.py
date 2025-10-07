# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
sys.path.insert(0, os.path.abspath('../../'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Causal AI Scientist'
copyright = '2024, CAIS Team'
author = 'CAIS Team'
version = '0.1.2'
release = '0.1.2'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.githubpages',
    'sphinx_sitemap',
    'sphinxcontrib.mermaid',
    'myst_parser',
    'sphinx_rtd_theme',
    'nbsphinx',
    'sphinxext.opengraph',
]

templates_path = ['_templates']
exclude_patterns = [
    'tutorials/notebooks/*.ipynb',  # Temporarily exclude notebooks
]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
html_theme_options = {
    'analytics_id': 'G-XXXXXXXXXX',  # Replace with actual Google Analytics ID
    'analytics_anonymize_ip': True,
    'logo_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'vcs_pageview_mode': '',
    'style_nav_header_background': 'white',
    # Toc options
    'collapse_navigation': False,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False
}

# Google Analytics configuration
html_context = {
    'google_analytics_id': 'G-XXXXXXXXXX',  # Replace with actual GA4 ID
    'google_tag_manager_id': 'GTM-XXXXXXX',  # Optional: Google Tag Manager
}

# Search configuration
html_search_language = 'en'
html_search_options = {
    'type': 'default',
    'scorer': 'default',
    'word_stem_language': 'english'
}

# Enable search highlighting (use default)

# SEO and sitemap configuration
html_baseurl = 'https://causal-ai-scientist.readthedocs.io/'
sitemap_url_scheme = "{link}"
sitemap_locales = [None]
sitemap_filename = "sitemap.xml"

# Additional HTML options for SEO
html_title = f"{project} v{version} Documentation"
html_short_title = project
html_use_opensearch = 'https://causal-ai-scientist.readthedocs.io/'

# Meta tags for SEO
html_meta = {
    'description': 'Comprehensive documentation for Causal AI Scientist (CAIS), an autonomous agent that combines Large Language Models with causal inference methods for automated causal analysis.',
    'keywords': 'causal inference, machine learning, AI agent, LLM, statistics, econometrics, data science',
    'author': 'CAIS Team',
    'robots': 'index, follow',
    'viewport': 'width=device-width, initial-scale=1.0',
}

# -- Extension configuration -------------------------------------------------

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__',
    'show-inheritance': True,
    'inherited-members': True,
}

# Autodoc configuration
autodoc_typehints = 'description'
autodoc_typehints_description_target = 'documented'
autodoc_typehints_format = 'short'
autodoc_preserve_defaults = True
autodoc_mock_imports = [
    'langchain_core',
    'langchain_openai', 
    'langchain_anthropic',
    'langchain_google_genai',
    'langchain_deepseek',
    'langchain_together',
    'openai',
    'vertexai',
    'google.cloud.aiplatform',
    'docker',
    'together',
]

# Autosummary settings
autosummary_generate = True
autosummary_generate_overwrite = True
autosummary_imported_members = True

# MyST parser settings
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_admonition",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "substitution",
    "tasklist",
]

# nbsphinx settings
nbsphinx_execute = 'never'  # Don't execute notebooks during build
nbsphinx_allow_errors = True
nbsphinx_timeout = 60  # Timeout for notebook execution in seconds
nbsphinx_codecell_lexer = 'ipython3'
nbsphinx_requirejs_path = ''  # Disable requirejs to avoid pandoc dependency

# Custom notebook prolog and epilog
nbsphinx_prolog = """
.. note::
   This notebook is part of the CAIS tutorial series. You can download and run it locally,
   or open it in cloud environments like Google Colab or Binder.

.. raw:: html

   <div class="notebook-buttons">
     <a href="https://colab.research.google.com/github/causal-ai-scientist/causal-ai-scientist/blob/main/docs/source/tutorials/notebooks/{{ env.docname }}.ipynb" target="_blank">
       <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
     </a>
     <a href="https://mybinder.org/v2/gh/causal-ai-scientist/causal-ai-scientist/main?filepath=docs/source/tutorials/notebooks/{{ env.docname }}.ipynb" target="_blank">
       <img src="https://mybinder.org/badge_logo.svg" alt="Launch Binder"/>
     </a>
   </div>

"""

nbsphinx_epilog = """
.. raw:: html

   <div class="notebook-footer">
     <p>
       <strong>Next Steps:</strong>
       <ul>
         <li>Try modifying the code and re-running the analysis</li>
         <li>Experiment with different datasets from the CAIS data collection</li>
         <li>Explore other causal inference methods in our tutorials</li>
       </ul>
     </p>
   </div>
"""

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'sklearn': ('https://scikit-learn.org/stable/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
    'seaborn': ('https://seaborn.pydata.org/', None),
}

# Add custom cross-reference types and search enhancements
def setup(app):
    """Custom Sphinx setup function."""
    app.add_crossref_type(
        directivename="method",
        rolename="method",
        indextemplate="pair: %s; causal inference method",
    )
    app.add_crossref_type(
        directivename="diagnostic",
        rolename="diagnostic", 
        indextemplate="pair: %s; diagnostic test",
    )
    
    # Add custom CSS and JavaScript
    app.add_css_file('custom.css')
    app.add_js_file('search_enhancements.js')
    app.add_js_file('feedback.js')
    
    # Connect to build events for sitemap generation
    app.connect('build-finished', generate_sitemap)

def generate_sitemap(app, exception):
    """Generate sitemap.xml after build completion."""
    if exception is not None:
        return
    
    import os
    from datetime import datetime
    
    # Get all HTML files
    html_files = []
    build_dir = app.outdir
    
    for root, dirs, files in os.walk(build_dir):
        for file in files:
            if file.endswith('.html') and file != 'sitemap.xml':
                rel_path = os.path.relpath(os.path.join(root, file), build_dir)
                if not rel_path.startswith('_'):  # Skip internal files
                    html_files.append(rel_path)
    
    # Generate sitemap content
    sitemap_content = '''<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
'''
    
    base_url = app.config.html_baseurl or 'https://causal-ai-scientist.readthedocs.io/'
    current_date = datetime.now().strftime('%Y-%m-%d')
    
    # Define priority mapping
    priority_map = {
        'index.html': '1.0',
        'getting_started/index.html': '0.9',
        'methods/index.html': '0.9',
        'tutorials/index.html': '0.9',
        'api/index.html': '0.9',
    }
    
    for html_file in sorted(html_files):
        url = base_url + html_file.replace('\\', '/')
        priority = priority_map.get(html_file, '0.8')
        
        # Determine change frequency based on file type
        if 'api/' in html_file:
            changefreq = 'weekly'
        elif 'tutorials/' in html_file or 'methods/' in html_file:
            changefreq = 'monthly'
        else:
            changefreq = 'weekly'
        
        sitemap_content += f'''  <url>
    <loc>{url}</loc>
    <lastmod>{current_date}</lastmod>
    <changefreq>{changefreq}</changefreq>
    <priority>{priority}</priority>
  </url>
'''
    
    sitemap_content += '</urlset>'
    
    # Write sitemap file
    sitemap_path = os.path.join(build_dir, 'sitemap.xml')
    with open(sitemap_path, 'w', encoding='utf-8') as f:
        f.write(sitemap_content)
    
    print(f"Generated sitemap with {len(html_files)} URLs: {sitemap_path}")

# OpenGraph settings
ogp_site_url = "https://causal-ai-scientist.readthedocs.io/"
ogp_description_length = 300
ogp_image = "https://causal-ai-scientist.readthedocs.io/en/latest/_static/logo.png"
ogp_site_name = "Causal AI Scientist Documentation"
ogp_type = "website"

# Todo extension settings
todo_include_todos = True

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Custom CSS files
html_css_files = [
    'custom.css',
    'performance_optimizations.css',
]

# Custom JavaScript files
html_js_files = [
    'search_enhancements.js',
    'accessibility_enhancements.js',
    'feedback.js',
    'analytics_dashboard.js',
]

# Additional files to copy
html_extra_path = ['_static/robots.txt']

# The master toctree document.
master_doc = 'index'

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True