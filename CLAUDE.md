
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains Udacity's Intro to Machine Learning (ud120) course projects. It's a collection of independent Python scripts organized by machine learning topics, each demonstrating specific algorithms and techniques.

The main content is in the `ud120-projects/` directory, which includes subdirectories for each topic:
- `pca/` - Principal Component Analysis (eigenfaces, recent work)
- `svm/` - Support Vector Machines
- `decision_tree/` - Decision Trees
- `naive_bayes/` - Naive Bayes
- `feature_selection/` - Feature selection algorithms
- `k_means/` - K-means clustering
- `outliers/` - Outlier detection
- `regression/` - Regression analysis
- `text_learning/` - Text classification
- `final_project/` - Final project (POI identification)
- `tools/` - Shared utility functions
- `validation/`, `evaluation/`, `datasets_questions/`, `choose_your_own/`

Each subdirectory contains standalone Python scripts that can be run independently.

## Repository Configuration

- **Git Status**: Clean (no uncommitted changes)
- **Virtual Environment**: `.venv/` exists and is gitignored (see `.venv/.gitignore`)
- **Documentation**: `ud120-projects/README.md` and `ud120-projects/CHANGELOG.md` provide setup notes and change history
- **No Cursor/Copilot rules**: No `.cursorrules`, `.cursor/rules/`, or `.github/copilot-instructions.md` files
- **No build system**: No `setup.py`, `pyproject.toml`, `Makefile`
- **No CI/CD**: No `.github/workflows/` or other CI configuration

## Development Environment

### Virtual Environment
A Python virtual environment is already set up at `.venv/`. To create a fresh environment:
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Unix
```

### Dependencies
Install required packages:
```bash
pip install -r ud120-projects/requirements.txt
```
Key dependencies: scikit-learn, numpy, scipy, matplotlib, nltk, joblib, requests.

### Path Configuration
Many scripts import utility functions from `tools/` using `sys.path.append`. Depending on where you run the script, you may need to adjust the path from `"../tools/"` to `"./tools/"` or vice versa (see README.md Path Note).

## Running Code

Each script is executable independently. Examples:
```bash
# Run eigenfaces PCA analysis
python ud120-projects/pca/eigenfaces.py

# Run SVM author identification
python ud120-projects/svm/svm_author_id.py

# Run decision tree author identification
python ud120-projects/decision_tree/dt_author_id.py
```

## Testing

There is no formal test suite (no pytest, unittest). However, there are custom test scripts:

- `test_nan_detection.py` (root) - Runs eigenfaces.py 20 times to detect NaN errors. Must be executed from repository root directory (uses relative path `./ud120-projects/pca/eigenfaces.py`).
- `ud120-projects/pca/test_eigenfaces_components.py` - Experiments with varying n_components and saves results to CSV
- `ud120-projects/pca/run_test.sh` - Bash script to run the component test

Note: `run_test.sh` is a bash script; on Windows you may need to use Git Bash or run `python test_eigenfaces_components.py` directly.

To run the PCA component test:
```bash
cd ud120-projects/pca
./run_test.sh
# or directly:
python test_eigenfaces_components.py
```

## Recent Work (Based on Git History)

Recent commits focus on:
1. **PCA eigenfaces**: Resolving NaN issues in PCA whitening (debug_report.md details the problem and solution using `svd_solver='full'` and variance thresholding)
2. **Feature selection**: Completed feature selection module
3. **Eigenfaces component testing**: Added scripts to evaluate F1-score vs n_components for Ariel Sharon classification

Key files for recent work:
- `ud120-projects/pca/eigenfaces.py` - Main eigenfaces implementation with NaN checks
- `ud120-projects/pca/debug_report.md` - Technical report on NaN issues
- `ud120-projects/pca/test_eigenfaces_components.py` - Component testing script
- `ud120-projects/feature_selection/find_signature.py` - Feature selection script

## Architecture Notes

### Shared Utilities (`tools/` directory)
- `feature_format.py` - Converts dictionary data to numpy arrays for sklearn
- `email_preprocess.py` - Preprocesses email text data (TF-IDF vectorization, feature selection)
- `parse_out_email_text.py` - Email text parsing
- `startup.py` - Initialization script
- Pickle files (`word_data.pkl`, `email_authors.pkl`) contain preprocessed datasets

### Data Files
Large datasets (Enron email corpus, face images) are either included as pickle files or downloaded on-demand by scikit-learn's `fetch_lfw_people`. **Note**: `fetch_lfw_people` requires internet connection and may download ~200MB of data on first run.

### Project Structure
Each topic directory follows a similar pattern: one or more Python scripts that load data, apply ML algorithms, and output results. Some scripts generate visualizations using matplotlib.

## Common Issues

1. **Path errors**: If import fails for `tools/`, modify `sys.path.append` in the script to use relative path.
2. **NaN errors in PCA**: Solved by using `svd_solver='full'` and variance thresholding (already implemented in eigenfaces.py).
3. **Missing dependencies**: Ensure all packages from requirements.txt are installed.
4. **Matplotlib plots**: Some scripts generate plots; if running headless, set `matplotlib.use('Agg')` before importing pyplot or configure appropriate backend.

## Contributing

This is primarily a learning repository. If you're extending the projects:
- Follow existing patterns for imports and data loading
- Add NaN checks where appropriate for numerical stability
- Document any experiments in markdown files
- Use the shared utilities in `tools/` for consistency
