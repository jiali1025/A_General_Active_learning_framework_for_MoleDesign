# Contributing to Unified Active Learning Framework

Thank you for your interest in contributing to the Unified Active Learning Framework for Molecular Design!

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/your-username/A_General_Active_learning_framework_for_MoleDesign.git
   ```
3. Create a new branch for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Setup

1. Install the development environment:
   ```bash
   conda create -n unified_al_dev python=3.8
   conda activate unified_al_dev
   pip install -r requirements.txt
   ```

2. Download the models (for testing):
   ```bash
   bash scripts/download_models.sh
   ```

## Code Style

- Follow PEP 8 Python style guidelines
- Use meaningful variable and function names
- Add docstrings to all functions and classes
- Comment complex algorithms and logic

## Adding New Acquisition Strategies

1. Create a new directory under `src/` for your strategy
2. Implement the core acquisition function
3. Add configuration to `configs/experiment_config.yaml`
4. Include example usage in documentation
5. Add tests if possible

## Testing

Before submitting a pull request:

1. Test your code with a small dataset
2. Ensure existing functionality still works
3. Update tests if you modify existing code

## Submitting Changes

1. Commit your changes with clear commit messages:
   ```bash
   git commit -m "Add new acquisition strategy: your_strategy_name"
   ```

2. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

3. Create a Pull Request on GitHub with:
   - Clear description of changes
   - Links to any relevant issues
   - Results/screenshots if applicable

## Reporting Issues

When reporting issues, please include:
- Python version and OS
- Complete error messages
- Steps to reproduce the issue
- Sample data (if possible)

## Questions?

Feel free to open an issue for any questions about contributing!