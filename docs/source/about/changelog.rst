Changelog
=========

This document tracks all notable changes to the Causal Agent  project. 


Version 0.1.2 (Current)
------------------------

*Released: October 2025*

**Improvements**

* **Build System Enhancement**: Fixed scipy version compatibility issues with statsmodels
* **Documentation Build**: Resolved langchain import dependencies for ReadTheDocs builds
* **Package Distribution**: Improved wheel and source distribution generation for PyPI
* **Testing Infrastructure**: Enhanced local build testing and validation processes
* **Dependency Management**: Added proper autodoc mock imports for documentation generation

**Bug Fixes**

* Fixed scipy version constraint to prevent conflicts with statsmodels (scipy>=1.10,<1.15)
* Resolved langchain import errors during documentation builds
* Fixed missing dependencies in docs/requirements.txt for proper Sphinx builds
* Corrected autodoc configuration to handle unavailable imports gracefully

**Technical Improvements**

* Enhanced pyproject.toml configuration for better package metadata
* Improved build process reliability for both wheel and source distributions
* Added comprehensive local testing workflow before PyPI releases
* Better error handling for missing optional dependencies
* **Critical Fix**: Added missing `langchain>=0.3.26` dependency to prevent import errors
* **Dependency Update**: Updated dowhy to version 0.12 for better compatibility

Version 0.1.1
--------------


**New Features**

* **Enhanced Decision Tree Logic**: Improved method selection algorithm with better handling of edge cases
* **Synthetic Data Generation**: Added comprehensive synthetic data generation system for testing and validation
* **Interactive Documentation**: Added interactive decision tree visualization and method selection tools
* **Batch Processing**: Enhanced support for processing multiple datasets in batch mode
* **LLM Provider Support**: Added support for multiple LLM providers (OpenAI, Anthropic, Google)

**Improvements**

* **Performance Optimization**: Reduced memory usage and improved processing speed for large datasets
* **Error Handling**: Better error messages and recovery mechanisms throughout the system
* **Documentation**: Comprehensive documentation overhaul with ReadTheDocs integration
* **Testing Framework**: Expanded test coverage with integration and performance tests
* **Code Quality**: Improved code organization and added type hints throughout

**Bug Fixes**

* Fixed issue with propensity score matching when treatment groups are highly imbalanced
* Resolved memory leak in batch processing mode
* Fixed incorrect confidence interval calculations for difference-in-differences estimator
* Corrected handling of missing values in instrumental variable analysis
* Fixed visualization rendering issues in Jupyter notebooks

**Breaking Changes**

* Changed API for custom method registration (see migration guide)
* Updated configuration file format for LLM providers
* Renamed several internal classes for consistency (backward compatibility maintained through deprecation warnings)

**Dependencies**

* Updated pandas to 2.1.0+ for better performance
* Added support for Python 3.11 and 3.12
* Updated scikit-learn to 1.3.0+ for improved estimators
* Added optional dependencies for enhanced visualization

Version 0.1.0
-------------

**Initial Release**

This is the first public release of the Causal AI Scientist, featuring:

**Core Features**

* **Autonomous Agent Architecture**: LLM-powered agent for automated causal inference
* **Decision Tree Algorithm**: Sophisticated method selection based on dataset properties
* **Multiple Causal Methods**: Support for RCT, DiD, IV, RDD, PSM, and observational methods
* **Automated Analysis Pipeline**: End-to-end analysis from data input to result interpretation
* **Result Interpretation**: Natural language explanations of causal analysis results

**Supported Methods**

* **Experimental Methods**
  
  * Randomized Controlled Trials (RCT)
  * Difference in Means
  
  * Difference-in-Differences (DiD)
  * Instrumental Variables (IV)
  * Regression Discontinuity Design (RDD)

* **Observational Methods**
  
  * Propensity Score Matching
  * Propensity Score Weighting
  * Backdoor Adjustment
  * Linear Regression

**Technical Infrastructure**

* **Python Package**: Installable via pip with comprehensive API
* **CLI Interface**: Command-line tool for batch processing
* **Jupyter Integration**: Seamless integration with Jupyter notebooks
* **Extensible Architecture**: Plugin system for adding new methods
* **Comprehensive Testing**: Unit, integration, and end-to-end tests

**Documentation**

* **Getting Started Guide**: Step-by-step installation and first analysis
* **API Documentation**: Complete reference for all functions and classes
* **Method Documentation**: Detailed explanation of each causal inference method
* **Tutorials**: Jupyter notebook tutorials for different domains
* **Case Studies**: Real-world examples across education, healthcare, and economics




Contributing to Changelog
--------------------------

When contributing to CAIS, please help maintain this changelog by:

* Adding entries for new features, improvements, and bug fixes
* Following the format established in this document
* Including breaking changes and migration information
* Updating the "Known Issues" section as appropriate

For more information on contributing, see our :doc:`../development/contributing` guide.

Release Process
---------------

Our release process follows these steps:

1. **Feature Development**: New features developed in feature branches
2. **Testing**: Comprehensive testing including unit, integration, and performance tests
3. **Documentation**: Update documentation and changelog
4. **Review**: Code review and approval process
5. **Release Candidate**: Create release candidate for final testing
6. **Release**: Tag release and publish to PyPI
7. **Announcement**: Announce release to community

**Release Schedule**

* **Major Releases** (x.0.0): Every 6-12 months with significant new features
* **Minor Releases** (x.y.0): Every 2-3 months with new features and improvements
* **Patch Releases** (x.y.z): As needed for bug fixes and security updates

**Support Policy**

* **Current Version**: Full support with new features and bug fixes
* **Previous Minor Version**: Bug fixes and security updates for 6 months
* **Older Versions**: Security updates only for critical vulnerabilities

Contact for Release Information
-------------------------------

* **Release Notifications**: Watch our GitHub repository for release notifications
* **Beta Testing**: Join our beta testing program by contacting cais-team@your-org.com
* **Release Questions**: Open an issue on GitHub or email cais-team@your-org.com