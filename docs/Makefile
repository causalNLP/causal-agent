# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD  ?= sphinx-build
SOURCEDIR    = source
BUILDDIR     = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Testing and validation targets
.PHONY: test-all test-build test-links test-spell test-code install-test-deps

# Install testing dependencies
install-test-deps:
	pip install -r test_requirements.txt

# Run all validation tests
test-all: install-test-deps
	python test_all_validation.py .

# Test documentation build
test-build: install-test-deps
	python test_documentation_build.py .

# Test links (requires built documentation)
test-links: html install-test-deps
	python test_link_checker.py build/html

# Test spelling and grammar
test-spell: install-test-deps
	python test_spell_grammar.py .

# Test code examples
test-code: install-test-deps
	python test_code_examples.py .

# Build and test everything
build-and-test: clean html test-all

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Custom targets for development
clean:
	@$(SPHINXBUILD) -M clean "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

apidoc:
	python generate_api_docs.py

html: apidoc
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

html-noapi:
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

livehtml:
	sphinx-autobuild "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS) $(O)

linkcheck:
	@$(SPHINXBUILD) -M linkcheck "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

doctest:
	@$(SPHINXBUILD) -M doctest "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

coverage:
	@$(SPHINXBUILD) -M coverage "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

validate-docstrings:
	python validate_docstrings.py

test-api:
	python test_api_docs.py

check: validate-docstrings test-api linkcheck
	@echo "✅ Documentation checks completed"

full-build: clean apidoc html
	@echo "✅ Full documentation build completed"