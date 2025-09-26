#!/usr/bin/env python3
"""
Generate a comprehensive report about the tutorial notebooks.

This script analyzes the tutorial notebooks and generates a report including:
- Notebook statistics
- Content analysis
- Quality metrics
- Recommendations
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Any
import nbformat
import re
from collections import Counter

class NotebookAnalyzer:
    """Analyze tutorial notebooks for comprehensive reporting."""
    
    def __init__(self, notebooks_dir: str):
        self.notebooks_dir = Path(notebooks_dir)
        self.notebooks = []
        self.analysis = {}
        
    def load_notebooks(self):
        """Load all notebooks from the directory."""
        notebook_paths = list(self.notebooks_dir.glob("*.ipynb"))
        
        for path in notebook_paths:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    nb = nbformat.read(f, as_version=4)
                    self.notebooks.append({
                        'path': path,
                        'name': path.name,
                        'notebook': nb
                    })
            except Exception as e:
                print(f"Warning: Could not load {path}: {e}")
    
    def analyze_notebook_structure(self, nb_data: Dict) -> Dict[str, Any]:
        """Analyze the structure of a single notebook."""
        nb = nb_data['notebook']
        
        analysis = {
            'total_cells': len(nb.cells),
            'code_cells': 0,
            'markdown_cells': 0,
            'raw_cells': 0,
            'empty_cells': 0,
            'lines_of_code': 0,
            'lines_of_markdown': 0,
            'imports': [],
            'functions_defined': [],
            'topics_covered': []
        }
        
        for cell in nb.cells:
            if cell.cell_type == 'code':
                analysis['code_cells'] += 1
                if cell.source.strip():
                    analysis['lines_of_code'] += len(cell.source.split('\n'))
                    
                    # Extract imports
                    imports = re.findall(r'^(?:from\s+(\S+)\s+)?import\s+([^\n]+)', cell.source, re.MULTILINE)
                    for imp in imports:
                        if imp[0]:  # from X import Y
                            analysis['imports'].append(f"from {imp[0]} import {imp[1]}")
                        else:  # import X
                            analysis['imports'].append(f"import {imp[1]}")
                    
                    # Extract function definitions
                    functions = re.findall(r'^def\s+(\w+)', cell.source, re.MULTILINE)
                    analysis['functions_defined'].extend(functions)
                else:
                    analysis['empty_cells'] += 1
                    
            elif cell.cell_type == 'markdown':
                analysis['markdown_cells'] += 1
                if cell.source.strip():
                    analysis['lines_of_markdown'] += len(cell.source.split('\n'))
                    
                    # Extract topics from headers
                    headers = re.findall(r'^#+\s+(.+)$', cell.source, re.MULTILINE)
                    analysis['topics_covered'].extend(headers)
                else:
                    analysis['empty_cells'] += 1
                    
            elif cell.cell_type == 'raw':
                analysis['raw_cells'] += 1
        
        # Remove duplicates
        analysis['imports'] = list(set(analysis['imports']))
        analysis['functions_defined'] = list(set(analysis['functions_defined']))
        
        return analysis
    
    def analyze_content_quality(self, nb_data: Dict) -> Dict[str, Any]:
        """Analyze content quality metrics."""
        nb = nb_data['notebook']
        
        quality = {
            'has_title': False,
            'has_introduction': False,
            'has_learning_objectives': False,
            'has_conclusion': False,
            'has_exercises': False,
            'code_to_markdown_ratio': 0,
            'avg_code_cell_length': 0,
            'documentation_coverage': 0,
            'educational_elements': []
        }
        
        markdown_cells = [cell for cell in nb.cells if cell.cell_type == 'markdown']
        code_cells = [cell for cell in nb.cells if cell.cell_type == 'code' and cell.source.strip()]
        
        if markdown_cells:
            first_cell = markdown_cells[0].source.lower()
            quality['has_title'] = any(line.startswith('#') for line in first_cell.split('\n'))
            
            all_markdown = ' '.join(cell.source.lower() for cell in markdown_cells)
            quality['has_introduction'] = 'introduction' in all_markdown or 'overview' in all_markdown
            quality['has_learning_objectives'] = 'learning objective' in all_markdown or 'you will learn' in all_markdown
            quality['has_conclusion'] = 'conclusion' in all_markdown or 'summary' in all_markdown
            quality['has_exercises'] = 'exercise' in all_markdown or 'try it yourself' in all_markdown
            
            # Educational elements
            if 'example' in all_markdown:
                quality['educational_elements'].append('examples')
            if 'note:' in all_markdown or 'important:' in all_markdown:
                quality['educational_elements'].append('callouts')
            if 'step' in all_markdown:
                quality['educational_elements'].append('step_by_step')
        
        if code_cells:
            quality['avg_code_cell_length'] = sum(len(cell.source.split('\n')) for cell in code_cells) / len(code_cells)
            quality['code_to_markdown_ratio'] = len(code_cells) / max(len(markdown_cells), 1)
        
        # Documentation coverage (ratio of markdown to code)
        if code_cells:
            quality['documentation_coverage'] = len(markdown_cells) / len(code_cells)
        
        return quality
    
    def analyze_technical_content(self, nb_data: Dict) -> Dict[str, Any]:
        """Analyze technical aspects of the notebook."""
        nb = nb_data['notebook']
        
        technical = {
            'causal_methods_mentioned': [],
            'statistical_concepts': [],
            'visualization_types': [],
            'data_operations': [],
            'error_handling': False,
            'best_practices': []
        }
        
        all_code = '\n'.join(cell.source for cell in nb.cells if cell.cell_type == 'code')
        all_text = '\n'.join(cell.source for cell in nb.cells if cell.cell_type == 'markdown')
        
        # Causal methods
        causal_methods = [
            'randomized controlled trial', 'rct', 'difference-in-differences', 'did',
            'instrumental variables', 'propensity score', 'regression discontinuity',
            'matching', 'backdoor adjustment', 'front-door'
        ]
        
        for method in causal_methods:
            if method in all_text.lower() or method in all_code.lower():
                technical['causal_methods_mentioned'].append(method)
        
        # Statistical concepts
        stats_concepts = [
            'p-value', 'confidence interval', 'standard error', 'effect size',
            'statistical significance', 'correlation', 'causation', 'bias',
            'confounding', 'assumption'
        ]
        
        for concept in stats_concepts:
            if concept in all_text.lower():
                technical['statistical_concepts'].append(concept)
        
        # Visualization types
        viz_patterns = {
            'scatter plot': r'scatter|scatterplot',
            'bar chart': r'bar\s*\(',
            'histogram': r'hist\s*\(',
            'box plot': r'boxplot',
            'line plot': r'plot\s*\(',
            'heatmap': r'heatmap'
        }
        
        for viz_type, pattern in viz_patterns.items():
            if re.search(pattern, all_code, re.IGNORECASE):
                technical['visualization_types'].append(viz_type)
        
        # Data operations
        data_ops = {
            'data loading': r'read_csv|load_data',
            'data cleaning': r'dropna|fillna|clean',
            'data transformation': r'transform|apply|map',
            'groupby operations': r'groupby',
            'merging/joining': r'merge|join',
            'filtering': r'\[.*==.*\]|query\('
        }
        
        for op_type, pattern in data_ops.items():
            if re.search(pattern, all_code, re.IGNORECASE):
                technical['data_operations'].append(op_type)
        
        # Error handling
        technical['error_handling'] = 'try:' in all_code or 'except' in all_code
        
        # Best practices
        if 'random.seed' in all_code or 'np.random.seed' in all_code:
            technical['best_practices'].append('reproducible_random_seed')
        if 'plt.style.use' in all_code or 'sns.set' in all_code:
            technical['best_practices'].append('consistent_plotting_style')
        if '# ' in all_code:  # Comments in code
            technical['best_practices'].append('code_comments')
        
        return technical
    
    def generate_overall_statistics(self) -> Dict[str, Any]:
        """Generate overall statistics across all notebooks."""
        if not self.notebooks:
            return {}
        
        stats = {
            'total_notebooks': len(self.notebooks),
            'total_cells': 0,
            'total_code_cells': 0,
            'total_markdown_cells': 0,
            'total_lines_of_code': 0,
            'total_lines_of_markdown': 0,
            'common_imports': Counter(),
            'common_topics': Counter(),
            'domains_covered': [],
            'quality_scores': []
        }
        
        for nb_data in self.notebooks:
            analysis = self.analysis[nb_data['name']]
            
            stats['total_cells'] += analysis['structure']['total_cells']
            stats['total_code_cells'] += analysis['structure']['code_cells']
            stats['total_markdown_cells'] += analysis['structure']['markdown_cells']
            stats['total_lines_of_code'] += analysis['structure']['lines_of_code']
            stats['total_lines_of_markdown'] += analysis['structure']['lines_of_markdown']
            
            # Count imports and topics
            for imp in analysis['structure']['imports']:
                stats['common_imports'][imp] += 1
            for topic in analysis['structure']['topics_covered']:
                stats['common_topics'][topic] += 1
            
            # Determine domain from filename
            name_lower = nb_data['name'].lower()
            if 'education' in name_lower:
                stats['domains_covered'].append('Education')
            elif 'healthcare' in name_lower or 'medical' in name_lower:
                stats['domains_covered'].append('Healthcare')
            elif 'economics' in name_lower or 'policy' in name_lower:
                stats['domains_covered'].append('Economics')
            
            # Calculate quality score
            quality = analysis['quality']
            quality_score = sum([
                quality['has_title'],
                quality['has_introduction'],
                quality['has_learning_objectives'],
                quality['has_conclusion'],
                quality['has_exercises'],
                quality['documentation_coverage'] > 0.5,
                len(quality['educational_elements']) > 2
            ]) / 7 * 100
            stats['quality_scores'].append(quality_score)
        
        stats['domains_covered'] = list(set(stats['domains_covered']))
        stats['avg_quality_score'] = sum(stats['quality_scores']) / len(stats['quality_scores']) if stats['quality_scores'] else 0
        
        return stats
    
    def run_analysis(self):
        """Run complete analysis on all notebooks."""
        self.load_notebooks()
        
        for nb_data in self.notebooks:
            self.analysis[nb_data['name']] = {
                'structure': self.analyze_notebook_structure(nb_data),
                'quality': self.analyze_content_quality(nb_data),
                'technical': self.analyze_technical_content(nb_data)
            }
    
    def generate_report(self) -> str:
        """Generate comprehensive report."""
        if not self.notebooks:
            return "No notebooks found for analysis."
        
        stats = self.generate_overall_statistics()
        
        report = f"""
Tutorial Notebooks Analysis Report
=================================

Overview
--------
Total Notebooks: {stats['total_notebooks']}
Domains Covered: {', '.join(stats['domains_covered'])}
Average Quality Score: {stats['avg_quality_score']:.1f}%

Content Statistics
-----------------
Total Cells: {stats['total_cells']}
Code Cells: {stats['total_code_cells']}
Markdown Cells: {stats['total_markdown_cells']}
Lines of Code: {stats['total_lines_of_code']}
Lines of Documentation: {stats['total_lines_of_markdown']}

Most Common Imports:
"""
        
        for imp, count in stats['common_imports'].most_common(10):
            report += f"  - {imp} ({count} notebooks)\n"
        
        report += "\nNotebook Details\n"
        report += "================\n"
        
        for nb_data in self.notebooks:
            name = nb_data['name']
            analysis = self.analysis[name]
            
            report += f"\n{name}:\n"
            report += f"  Structure:\n"
            report += f"    - Total cells: {analysis['structure']['total_cells']}\n"
            report += f"    - Code cells: {analysis['structure']['code_cells']}\n"
            report += f"    - Markdown cells: {analysis['structure']['markdown_cells']}\n"
            report += f"    - Lines of code: {analysis['structure']['lines_of_code']}\n"
            
            report += f"  Quality:\n"
            quality = analysis['quality']
            report += f"    - Has title: {'✓' if quality['has_title'] else '✗'}\n"
            report += f"    - Has introduction: {'✓' if quality['has_introduction'] else '✗'}\n"
            report += f"    - Has learning objectives: {'✓' if quality['has_learning_objectives'] else '✗'}\n"
            report += f"    - Has conclusion: {'✓' if quality['has_conclusion'] else '✗'}\n"
            report += f"    - Has exercises: {'✓' if quality['has_exercises'] else '✗'}\n"
            report += f"    - Documentation coverage: {quality['documentation_coverage']:.2f}\n"
            
            report += f"  Technical Content:\n"
            technical = analysis['technical']
            if technical['causal_methods_mentioned']:
                report += f"    - Causal methods: {', '.join(technical['causal_methods_mentioned'][:3])}\n"
            if technical['visualization_types']:
                report += f"    - Visualizations: {', '.join(technical['visualization_types'][:3])}\n"
            report += f"    - Error handling: {'✓' if technical['error_handling'] else '✗'}\n"
            if technical['best_practices']:
                report += f"    - Best practices: {', '.join(technical['best_practices'])}\n"
        
        report += "\nRecommendations\n"
        report += "===============\n"
        
        # Generate recommendations based on analysis
        recommendations = []
        
        if stats['avg_quality_score'] < 80:
            recommendations.append("Consider improving overall notebook quality by adding missing elements (titles, objectives, conclusions)")
        
        low_quality_notebooks = [nb for nb, score in zip(self.notebooks, stats['quality_scores']) if score < 70]
        if low_quality_notebooks:
            recommendations.append(f"Focus on improving: {', '.join(nb['name'] for nb in low_quality_notebooks[:3])}")
        
        if len(stats['domains_covered']) < 3:
            recommendations.append("Consider adding notebooks for additional domains (e.g., marketing, technology)")
        
        for rec in recommendations:
            report += f"- {rec}\n"
        
        if not recommendations:
            report += "- All notebooks meet quality standards! Consider adding advanced topics or additional domains.\n"
        
        return report


def main():
    """Main function to generate notebook report."""
    notebooks_dir = "docs/source/tutorials/notebooks"
    
    if not os.path.exists(notebooks_dir):
        print(f"Notebooks directory not found: {notebooks_dir}")
        return
    
    analyzer = NotebookAnalyzer(notebooks_dir)
    analyzer.run_analysis()
    
    report = analyzer.generate_report()
    print(report)


if __name__ == "__main__":
    main()