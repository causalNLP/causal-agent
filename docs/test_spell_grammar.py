#!/usr/bin/env python3
"""
Spell checking and grammar validation for documentation.
Uses pyspellchecker for spell checking and basic grammar rules.
"""

import os
import sys
import re
import json
from pathlib import Path
from typing import List, Dict, Set, Tuple
from collections import defaultdict
import html

try:
    from spellchecker import SpellChecker
    SPELLCHECKER_AVAILABLE = True
except ImportError:
    SPELLCHECKER_AVAILABLE = False
    print("WARNING: pyspellchecker not available. Install with: pip install pyspellchecker")

class SpellGrammarChecker:
    """Check spelling and basic grammar in documentation."""
    
    def __init__(self, docs_dir: str):
        self.docs_dir = Path(docs_dir)
        self.source_dir = self.docs_dir / "source"
        self.html_dir = self.docs_dir / "build" / "html"
        
        # Initialize spell checker
        if SPELLCHECKER_AVAILABLE:
            self.spell = SpellChecker()
            self._load_custom_dictionary()
        else:
            self.spell = None
            
        self.spelling_errors = []
        self.grammar_issues = []
        
        # Common technical terms to ignore
        self.technical_terms = {
            'api', 'apis', 'cli', 'gui', 'url', 'urls', 'http', 'https', 'json', 'xml',
            'html', 'css', 'javascript', 'python', 'jupyter', 'notebook', 'notebooks',
            'github', 'readthedocs', 'sphinx', 'rst', 'markdown', 'yaml', 'yml',
            'causal', 'inference', 'cais', 'llm', 'llms', 'ai', 'ml', 'rct', 'did',
            'iv', 'rdd', 'propensity', 'backdoor', 'frontdoor', 'confounding',
            'confounders', 'estimand', 'estimator', 'heterogeneity', 'endogeneity',
            'exogeneity', 'instrumentality', 'randomization', 'stratification',
            'sklearn', 'pandas', 'numpy', 'scipy', 'matplotlib', 'seaborn',
            'dataframe', 'dataset', 'datasets', 'preprocessing', 'postprocessing',
            'hyperparameter', 'hyperparameters', 'workflow', 'workflows',
            'config', 'configs', 'param', 'params', 'arg', 'args', 'kwargs',
            'docstring', 'docstrings', 'boolean', 'tuple', 'dict', 'str', 'int', 'float',
            'mermaid', 'flowchart', 'flowcharts', 'tooltip', 'tooltips',
            'dropdown', 'checkbox', 'textarea', 'onclick', 'onload',
            'css', 'js', 'html', 'dom', 'ajax', 'json', 'api', 'rest', 'restful'
        }
        
        if self.spell:
            self.spell.word_frequency.load_words(self.technical_terms)
    
    def _load_custom_dictionary(self):
        """Load custom dictionary for domain-specific terms."""
        custom_dict_file = self.docs_dir / "custom_dictionary.txt"
        
        if custom_dict_file.exists():
            try:
                with open(custom_dict_file, 'r') as f:
                    custom_words = [line.strip().lower() for line in f if line.strip()]
                    self.spell.word_frequency.load_words(custom_words)
                    print(f"Loaded {len(custom_words)} custom dictionary words")
            except Exception as e:
                print(f"WARNING: Could not load custom dictionary: {e}")
    
    def run_all_checks(self) -> bool:
        """Run all spelling and grammar checks."""
        print("Starting spelling and grammar validation...")
        
        if not self.spell:
            print("ERROR: Spell checker not available")
            return False
            
        # Check RST source files
        rst_ok = self._check_rst_files()
        
        # Check HTML files (for final output validation)
        html_ok = self._check_html_files()
        
        # Generate report
        self._generate_report()
        
        # Don't fail build for spelling/grammar issues, just report them
        return True
    
    def _check_rst_files(self) -> bool:
        """Check spelling in RST source files."""
        print("Checking RST source files...")
        
        if not self.source_dir.exists():
            print(f"WARNING: Source directory not found: {self.source_dir}")
            return True
            
        rst_files = list(self.source_dir.rglob("*.rst"))
        print(f"Found {len(rst_files)} RST files")
        
        for rst_file in rst_files:
            self._check_file_spelling(rst_file, 'rst')
            self._check_file_grammar(rst_file, 'rst')
            
        return True
    
    def _check_html_files(self) -> bool:
        """Check spelling in HTML output files."""
        print("Checking HTML output files...")
        
        if not self.html_dir.exists():
            print(f"WARNING: HTML directory not found: {self.html_dir}")
            return True
            
        # Check a sample of HTML files to avoid processing too many
        html_files = list(self.html_dir.rglob("*.html"))[:20]  # Limit to 20 files
        print(f"Checking {len(html_files)} HTML files")
        
        for html_file in html_files:
            self._check_file_spelling(html_file, 'html')
            
        return True
    
    def _check_file_spelling(self, file_path: Path, file_type: str):
        """Check spelling in a single file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Extract text based on file type
            if file_type == 'html':
                text = self._extract_text_from_html(content)
            else:
                text = self._extract_text_from_rst(content)
                
            # Check spelling
            words = self._extract_words(text)
            misspelled = self.spell.unknown(words)
            
            for word in misspelled:
                # Skip very short words and numbers
                if len(word) < 3 or word.isdigit():
                    continue
                    
                # Skip words that are likely code or technical terms
                if self._is_likely_code(word):
                    continue
                    
                suggestions = list(self.spell.candidates(word))[:3]
                
                self.spelling_errors.append({
                    'file': str(file_path.relative_to(self.docs_dir)),
                    'word': word,
                    'suggestions': suggestions,
                    'type': 'spelling'
                })
                
        except Exception as e:
            print(f"WARNING: Could not check spelling in {file_path}: {e}")
    
    def _check_file_grammar(self, file_path: Path, file_type: str):
        """Check basic grammar rules in a file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Extract text
            if file_type == 'html':
                text = self._extract_text_from_html(content)
            else:
                text = self._extract_text_from_rst(content)
                
            # Check basic grammar rules
            self._check_basic_grammar(file_path, text)
            
        except Exception as e:
            print(f"WARNING: Could not check grammar in {file_path}: {e}")
    
    def _extract_text_from_html(self, html_content: str) -> str:
        """Extract readable text from HTML content."""
        # Remove script and style elements
        html_content = re.sub(r'<script[^>]*>.*?</script>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
        html_content = re.sub(r'<style[^>]*>.*?</style>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', html_content)
        
        # Decode HTML entities
        text = html.unescape(text)
        
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _extract_text_from_rst(self, rst_content: str) -> str:
        """Extract readable text from RST content."""
        # Remove RST directives
        rst_content = re.sub(r'^\.\. [^:]+::', '', rst_content, flags=re.MULTILINE)
        
        # Remove code blocks
        rst_content = re.sub(r'^\.\. code-block::.*?(?=^\S|\Z)', '', rst_content, flags=re.MULTILINE | re.DOTALL)
        
        # Remove inline code
        rst_content = re.sub(r'``[^`]+``', '', rst_content)
        
        # Remove references
        rst_content = re.sub(r':ref:`[^`]+`', '', rst_content)
        rst_content = re.sub(r':doc:`[^`]+`', '', rst_content)
        rst_content = re.sub(r':class:`[^`]+`', '', rst_content)
        rst_content = re.sub(r':func:`[^`]+`', '', rst_content)
        
        # Remove section markers
        rst_content = re.sub(r'^[=\-~^"\'`#*+<>_]{3,}$', '', rst_content, flags=re.MULTILINE)
        
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', rst_content).strip()
        
        return text
    
    def _extract_words(self, text: str) -> List[str]:
        """Extract words from text for spell checking."""
        # Extract words (letters only, minimum 2 characters)
        words = re.findall(r'\b[a-zA-Z]{2,}\b', text.lower())
        return words
    
    def _is_likely_code(self, word: str) -> bool:
        """Check if a word is likely code or technical term."""
        # Check for common code patterns
        code_patterns = [
            r'^[a-z]+_[a-z]+',  # snake_case
            r'^[a-z]+[A-Z]',    # camelCase
            r'^[A-Z][a-z]*[A-Z]', # PascalCase
            r'^\w*\d+\w*$',     # contains numbers
            r'^[a-z]{1,3}$',    # very short (likely abbreviation)
        ]
        
        for pattern in code_patterns:
            if re.match(pattern, word):
                return True
                
        return False
    
    def _check_basic_grammar(self, file_path: Path, text: str):
        """Check basic grammar rules."""
        sentences = re.split(r'[.!?]+', text)
        
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Check for common grammar issues
            issues = []
            
            # Double spaces
            if '  ' in sentence:
                issues.append("Double spaces found")
                
            # Sentence should start with capital letter
            if sentence and not sentence[0].isupper():
                issues.append("Sentence should start with capital letter")
                
            # Check for common word pairs
            common_errors = [
                (r'\bits\s+its\b', "Possible confusion: 'its its' should be 'it's its' or 'its'"),
                (r'\btheir\s+there\b', "Possible confusion: 'their there'"),
                (r'\bthen\s+than\b', "Possible confusion: 'then than'"),
            ]
            
            for pattern, message in common_errors:
                if re.search(pattern, sentence, re.IGNORECASE):
                    issues.append(message)
            
            # Record issues
            for issue in issues:
                self.grammar_issues.append({
                    'file': str(file_path.relative_to(self.docs_dir)),
                    'sentence_num': i + 1,
                    'sentence': sentence[:100] + "..." if len(sentence) > 100 else sentence,
                    'issue': issue,
                    'type': 'grammar'
                })
    
    def _generate_report(self):
        """Generate spelling and grammar report."""
        report_file = self.docs_dir / "spell_grammar_report.json"
        
        report = {
            'timestamp': __import__('time').strftime('%Y-%m-%d %H:%M:%S'),
            'summary': {
                'spelling_errors': len(self.spelling_errors),
                'grammar_issues': len(self.grammar_issues),
                'total_issues': len(self.spelling_errors) + len(self.grammar_issues)
            },
            'spelling_errors': self.spelling_errors,
            'grammar_issues': self.grammar_issues
        }
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
            
        print(f"\n📊 Spell/grammar report saved to: {report_file}")
        
        # Print summary
        print("\n" + "="*50)
        print("SPELL & GRAMMAR CHECK SUMMARY")
        print("="*50)
        print(f"Spelling errors: {report['summary']['spelling_errors']}")
        print(f"Grammar issues: {report['summary']['grammar_issues']}")
        print(f"Total issues: {report['summary']['total_issues']}")
        
        # Show some examples
        if self.spelling_errors:
            print(f"\nTop spelling errors:")
            for error in self.spelling_errors[:5]:
                suggestions = ", ".join(error['suggestions']) if error['suggestions'] else "No suggestions"
                print(f"  {error['file']}: '{error['word']}' -> {suggestions}")
                
        if self.grammar_issues:
            print(f"\nGrammar issues found:")
            for issue in self.grammar_issues[:3]:
                print(f"  {issue['file']}: {issue['issue']}")
        
        if report['summary']['total_issues'] == 0:
            print("🎉 No spelling or grammar issues found!")
        else:
            print(f"⚠️  {report['summary']['total_issues']} issues found (not blocking build)")

def main():
    """Main function to run spell and grammar checker."""
    if len(sys.argv) < 2:
        print("Usage: python test_spell_grammar.py <docs_directory>")
        sys.exit(1)
        
    docs_dir = sys.argv[1]
    
    checker = SpellGrammarChecker(docs_dir)
    success = checker.run_all_checks()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()