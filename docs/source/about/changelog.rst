Changelog
=========

This document tracks all notable changes to the Causal AI Scientist (CAIS) project. 
We follow `Semantic Versioning <https://semver.org/>`_ principles.

Version 0.1.1 (Current)
------------------------

*Released: December 2024*

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

*Released: September 2024*

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

* **Quasi-Experimental Methods**
  
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

Development Milestones
----------------------

**Pre-Release Development** (January 2024 - August 2024)

* **Research Phase**: Literature review and method selection
* **Architecture Design**: Agent-based system design and LLM integration patterns
* **Core Implementation**: Basic method implementations and decision tree logic
* **Testing Framework**: Initial testing infrastructure and synthetic data generation
* **Alpha Testing**: Internal testing and validation with synthetic datasets
* **Beta Testing**: External testing with real datasets and user feedback

**Future Roadmap**
------------------

**Version 0.2.0** (Planned: Q1 2025)

* **Advanced Methods**: Support for mediation analysis and sensitivity analysis
* **Enhanced LLM Integration**: Improved prompt engineering and response processing
* **Web Interface**: Browser-based interface for non-technical users
* **Cloud Integration**: Native support for cloud data sources and processing
* **Performance Improvements**: Optimizations for very large datasets

**Version 0.3.0** (Planned: Q2 2025)

* **Causal Discovery**: Automated causal graph discovery from data
* **Time Series Methods**: Support for time series causal inference
* **Multi-Treatment Analysis**: Enhanced support for multiple treatments
* **Advanced Diagnostics**: Expanded diagnostic tests and assumption checking
* **Integration APIs**: REST API for integration with other tools

**Long-term Vision**

* **Domain-Specific Modules**: Specialized modules for healthcare, economics, education
* **Real-time Analysis**: Support for streaming data and real-time causal inference
* **Collaborative Features**: Multi-user collaboration and analysis sharing
* **Educational Platform**: Interactive learning platform for causal inference
* **Enterprise Features**: Advanced security, audit trails, and compliance features

Migration Guides
-----------------

**Migrating from 0.1.0 to 0.1.1**

**API Changes**

.. code-block:: python

   # Old way (deprecated but still works)
   from causal_agent import CausalAgent
   agent = CausalAgent(config_file="config.yaml")
   
   # New way (recommended)
   from causal_agent import Agent
   agent = Agent.from_config("config.yaml")

**Configuration Changes**

The LLM provider configuration format has changed:

.. code-block:: yaml

   # Old format (still supported with warnings)
   llm:
     provider: "openai"
     api_key: "your-key"
   
   # New format (recommended)
   llm_providers:
     openai:
       api_key: "your-key"
       model: "gpt-4"
     anthropic:
       api_key: "your-key"
       model: "claude-3"

**Method Registration**

Custom method registration has been simplified:

.. code-block:: python

   # Old way (no longer supported)
   agent.register_method("custom_method", CustomMethodClass)
   
   # New way
   from causal_agent.methods import register_method
   register_method("custom_method", CustomMethodClass)

Known Issues
------------

**Current Known Issues** (Version 0.1.1)

* **Large Dataset Performance**: Processing datasets with >1M rows may be slow
* **Memory Usage**: High memory usage with certain propensity score methods
* **Windows Compatibility**: Some visualization features may not work on Windows
* **Jupyter Lab**: Interactive features may not work in JupyterLab (use Jupyter Notebook)

**Workarounds**

* For large datasets, consider using the batch processing mode
* Monitor memory usage and consider data sampling for very large datasets
* Use WSL on Windows for full feature compatibility
* Use Jupyter Notebook instead of JupyterLab for interactive features

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