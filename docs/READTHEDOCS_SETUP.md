# ReadTheDocs Integration Setup

This document provides instructions for setting up and configuring ReadTheDocs integration for the Causal AI Scientist documentation.

## Configuration Files

### .readthedocs.yaml
The main configuration file for ReadTheDocs builds is located in the project root. This file:
- Specifies Python 3.11 and Ubuntu 22.04 as the build environment
- Points to `docs/source/conf.py` as the Sphinx configuration
- Enables PDF and ePub format generation
- Installs dependencies from both `requirements.txt` and `docs/requirements.txt`

### docs/requirements.txt
Contains all documentation-specific dependencies including:
- Sphinx and ReadTheDocs theme
- Extensions for notebooks, API docs, and diagrams
- Jupyter support for interactive tutorials

## ReadTheDocs Project Setup

### 1. Account Setup
1. Go to [ReadTheDocs.org](https://readthedocs.org/)
2. Sign in with your GitHub account
3. Import the repository

### 2. Project Configuration
1. **Project Name**: `causal-ai-scientist`
2. **Repository URL**: Your GitHub repository URL
3. **Default Branch**: `main`
4. **Language**: `English`
5. **Programming Language**: `Python`

### 3. Advanced Settings
- **Install Project**: Enable (installs the package for API docs)
- **Use system packages**: Disable
- **Requirements file**: `docs/requirements.txt`
- **Python configuration file**: `docs/source/conf.py`
- **Build image**: `Ubuntu 22.04`

### 4. Webhook Configuration
ReadTheDocs will automatically configure webhooks for:
- Push events to trigger documentation builds
- Pull request events for preview builds

## Custom Domain Setup (Optional)

### 1. DNS Configuration
If you want to use a custom domain (e.g., `docs.causal-ai-scientist.org`):

1. Create a CNAME record pointing to `readthedocs.io`
2. In ReadTheDocs project settings, add the custom domain
3. Enable HTTPS (ReadTheDocs provides free SSL certificates)

### 2. GitHub Pages Fallback
The GitHub Actions workflow also deploys to GitHub Pages as a fallback:
- Accessible at `https://yourusername.github.io/causal-ai-scientist/`
- Automatically updates on pushes to main branch

## Build Process

### Automatic Builds
ReadTheDocs will automatically build documentation when:
- Code is pushed to the main branch
- Pull requests are created or updated
- Documentation files are modified

### Manual Builds
You can trigger manual builds from the ReadTheDocs dashboard or via API.

### Local Testing
To test documentation builds locally:

```bash
# Install dependencies
pip install -r requirements.txt
pip install -r docs/requirements.txt
pip install -e .

# Build documentation
cd docs
make html

# Serve locally
python -m http.server 8000 -d build/html
```

## Troubleshooting

### Common Build Issues

1. **Import Errors**: Ensure all dependencies are listed in `docs/requirements.txt`
2. **Path Issues**: Check that `sys.path` is correctly configured in `conf.py`
3. **Missing Files**: Verify all referenced files exist and paths are correct

### Build Logs
- Check ReadTheDocs build logs for detailed error information
- GitHub Actions logs provide additional debugging information

### Version Management
- ReadTheDocs automatically builds documentation for all branches and tags
- Configure which versions to display in the ReadTheDocs admin panel

## Integration Checklist

- [x] `.readthedocs.yaml` configuration file created
- [x] `docs/requirements.txt` with all dependencies
- [x] Sphinx configuration (`docs/source/conf.py`) optimized for ReadTheDocs
- [x] GitHub Actions workflow for testing builds
- [ ] ReadTheDocs project imported and configured
- [ ] Webhook integration tested
- [ ] Custom domain configured (optional)
- [ ] SSL certificate enabled
- [ ] Build process validated with sample content

## Next Steps

1. Import the project on ReadTheDocs.org
2. Configure project settings as described above
3. Test the build process with initial content
4. Set up custom domain if desired
5. Configure team access and permissions