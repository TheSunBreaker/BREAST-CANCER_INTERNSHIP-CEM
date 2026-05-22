# Contributing to Breast Cancer Radiomics Project

Thank you for your interest in contributing to the Breast Cancer Radiomics Project! This document provides guidelines and information for contributors.

## 🤝 How to Contribute

### Reporting Bugs

1. Check if the bug has already been reported in the [Issues](https://github.com/your-username/breast-cancer-radiomics/issues) section
2. Create a new issue with a clear and descriptive title
3. Include detailed information about the bug:
   - Steps to reproduce the issue
   - Expected vs actual behavior
   - Environment details (OS, Python version, SimpleITK version)
   - Error messages and stack traces

### Suggesting Enhancements

1. Check if the enhancement has already been suggested
2. Create a new issue with the "enhancement" label
3. Describe the proposed feature and its benefits
4. Include use cases and implementation ideas if possible

### Code Contributions

#### Development Setup

1. Fork the repository
2. Clone your fork locally:
   ```bash
   git clone https://github.com/your-username/breast-cancer-radiomics.git
   cd breast-cancer-radiomics
   ```

3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

5. Create a new branch for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```

#### Coding Standards

- Follow PEP 8 style guidelines
- Use meaningful variable and function names
- Add docstrings to all functions and classes
- Include type hints where appropriate
- Write clear, descriptive commit messages

#### Testing

- Add tests for new functionality
- Ensure all existing tests pass
- Test with different Python versions (3.7+)
- Test on different operating systems if possible

#### Submitting Changes

1. Commit your changes with clear messages:
   ```bash
   git commit -m "Add feature: brief description"
   ```

2. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

3. Create a Pull Request with:
   - Clear description of changes
   - Reference to related issues
   - Screenshots or examples if applicable

## 📋 Code Review Process

1. All contributions require review
2. Maintainers will review your code and provide feedback
3. Address any review comments
4. Once approved, your changes will be merged

## 🏷️ Issue Labels

- `bug`: Something isn't working
- `enhancement`: New feature or request
- `documentation`: Improvements or additions to documentation
- `good first issue`: Good for newcomers
- `help wanted`: Extra attention is needed
- `question`: Further information is requested

## 📚 Documentation

When contributing code, please also:

- Update relevant documentation
- Add comments for complex logic
- Include usage examples
- Update the README if necessary

## 🔒 Medical Data Guidelines

Since this project deals with medical imaging:

- Never include real patient data in contributions
- Use synthetic or anonymized test data
- Follow HIPAA and data protection guidelines
- Ensure compliance with institutional review board requirements

## 🎯 Areas for Contribution

We welcome contributions in these areas:

- **Performance improvements**: Optimize DICOM processing
- **New features**: Additional image processing capabilities
- **Documentation**: Improve README, add tutorials
- **Testing**: Add unit tests and integration tests
- **Bug fixes**: Resolve reported issues
- **Code quality**: Refactor and improve existing code

## 📞 Getting Help

If you need help with contributing:

1. Check existing issues and discussions
2. Create a new issue with the "question" label
3. Contact the maintainers directly

## 🙏 Recognition

Contributors will be:

- Listed in the README.md file
- Acknowledged in release notes
- Given credit in academic publications if applicable

Thank you for contributing to advancing breast cancer research through open-source collaboration! 
