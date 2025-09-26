#!/usr/bin/env python3
"""
Create downloadable packages for tutorial notebooks.

This script creates ZIP files containing notebooks and supporting materials
for easy download and local use.
"""

import os
import zipfile
import shutil
from pathlib import Path
import json

def create_notebook_package():
    """Create a ZIP package with all notebooks and supporting files."""
    
    notebooks_dir = Path("docs/source/tutorials/notebooks")
    output_dir = Path("docs/source/_static")
    package_name = "cais_tutorial_notebooks.zip"
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create temporary directory for package contents
    temp_dir = Path("temp_notebook_package")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir()
    
    try:
        # Copy notebooks
        notebooks_dest = temp_dir / "notebooks"
        notebooks_dest.mkdir()
        
        for notebook in notebooks_dir.glob("*.ipynb"):
            shutil.copy2(notebook, notebooks_dest)
        
        # Copy sample data (if exists)
        data_dir = Path("data/all_data")
        if data_dir.exists():
            data_dest = temp_dir / "data"
            data_dest.mkdir()
            
            # Copy a subset of data files used in tutorials
            sample_files = [
                "learning_mindset.csv",
                "hospital_treatment.csv", 
                "min_wage_data.csv"
            ]
            
            for file in sample_files:
                src_file = data_dir / file
                if src_file.exists():
                    shutil.copy2(src_file, data_dest)
        
        # Create requirements.txt
        requirements_content = """# Requirements for CAIS Tutorial Notebooks
causal-agent
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.5.0
seaborn>=0.11.0
scipy>=1.7.0
scikit-learn>=1.0.0
jupyter>=1.0.0
notebook>=6.4.0
"""
        
        with open(temp_dir / "requirements.txt", "w") as f:
            f.write(requirements_content)
        
        # Create README
        readme_content = """# CAIS Tutorial Notebooks

This package contains interactive Jupyter notebooks for learning causal inference with CAIS.

## Installation

1. Install Python 3.8 or higher
2. Install required packages:
   ```
   pip install -r requirements.txt
   ```

## Running the Notebooks

1. Start Jupyter:
   ```
   jupyter notebook
   ```
2. Navigate to the `notebooks/` folder
3. Open any tutorial notebook
4. Follow the step-by-step instructions

## Notebooks Included

- **education_analysis_tutorial.ipynb** - Educational intervention analysis using RCTs
- **healthcare_analysis_tutorial.ipynb** - Medical treatment analysis using propensity scores  
- **economics_analysis_tutorial.ipynb** - Policy analysis using difference-in-differences

## API Keys

For full functionality, you'll need API keys from an LLM provider:
- OpenAI (recommended)
- Anthropic
- Google Gemini
- Together AI

Set your API key as an environment variable:
```
export OPENAI_API_KEY="your-key-here"
```

## Support

- Documentation: https://causal-ai-scientist.readthedocs.io/
- GitHub: https://github.com/causal-ai-scientist/causal-ai-scientist
- Issues: https://github.com/causal-ai-scientist/causal-ai-scientist/issues

## License

This project is licensed under the MIT License.
"""
        
        with open(temp_dir / "README.md", "w") as f:
            f.write(readme_content)
        
        # Create the ZIP file
        zip_path = output_dir / package_name
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    file_path = Path(root) / file
                    arc_path = file_path.relative_to(temp_dir)
                    zipf.write(file_path, arc_path)
        
        print(f"Created notebook package: {zip_path}")
        print(f"Package size: {zip_path.stat().st_size / 1024 / 1024:.1f} MB")
        
        # Create individual notebook downloads
        for notebook in notebooks_dir.glob("*.ipynb"):
            individual_zip = output_dir / f"{notebook.stem}_tutorial.zip"
            with zipfile.ZipFile(individual_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
                zipf.write(notebook, notebook.name)
                zipf.writestr("requirements.txt", requirements_content)
                zipf.writestr("README.md", f"# {notebook.stem.replace('_', ' ').title()}\n\n" + readme_content)
            
            print(f"Created individual package: {individual_zip}")
    
    finally:
        # Clean up temporary directory
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

def create_colab_links():
    """Create a file with Google Colab links for each notebook."""
    
    notebooks_dir = Path("docs/source/tutorials/notebooks")
    output_file = Path("docs/source/_static/colab_links.json")
    
    base_url = "https://colab.research.google.com/github/causal-ai-scientist/causal-ai-scientist/blob/main/docs/source/tutorials/notebooks/"
    
    links = {}
    for notebook in notebooks_dir.glob("*.ipynb"):
        notebook_name = notebook.stem
        colab_url = base_url + notebook.name
        links[notebook_name] = {
            "title": notebook_name.replace('_', ' ').title(),
            "colab_url": colab_url,
            "github_url": f"https://github.com/causal-ai-scientist/causal-ai-scientist/blob/main/docs/source/tutorials/notebooks/{notebook.name}"
        }
    
    with open(output_file, 'w') as f:
        json.dump(links, f, indent=2)
    
    print(f"Created Colab links file: {output_file}")

def main():
    """Main function to create all download packages."""
    print("Creating notebook download packages...")
    
    create_notebook_package()
    create_colab_links()
    
    print("\nDownload packages created successfully!")
    print("\nTo update the packages, run this script again after making changes to notebooks.")

if __name__ == "__main__":
    main()