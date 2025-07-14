# Version Management for TestTeller RAG Agent

This document describes the centralized version management system implemented for the TestTeller RAG Agent.

## Overview

The version is now managed from a **single source of truth** located in `testteller/_version.py`. All other files that need version information import it from this location.

## Files Involved

### 1. Single Source of Truth
- **`testteller/_version.py`** - Contains the actual version string

### 2. Files that Import Version
- **`testteller/__init__.py`** - Imports and exposes `__version__`
- **`testteller/config.py`** - Imports for configuration settings
- **`setup.py`** - Extracts version for package building
- **`pyproject.toml`** - Uses dynamic versioning from `testteller._version.__version__`

## How It Works

1. **Version Definition**: The version is defined once in `testteller/_version.py`:
   ```python
   __version__ = "0.1.2"
   ```

2. **Import Chain**:
   ```
   _version.py → __init__.py → config.py
                ↓
              setup.py (via regex extraction)
                ↓
           pyproject.toml (via setuptools dynamic)
   ```

3. **Build Process**: Both `setuptools` and `wheel` correctly extract the version from the centralized location.

## Usage

### Manual Version Updates

1. **Edit the version directly** in `testteller/_version.py`:
   ```python
   __version__ = "0.2.0"  # New version
   ```

2. **All other locations automatically pick up the new version**

### Using the Version Bump Script

We provide a utility script `bump_version.py` for easy version management:

```bash
# Show current version
python bump_version.py show

# Set specific version
python bump_version.py set 1.0.0

# Bump version parts
python bump_version.py bump patch   # 0.1.2 → 0.1.3
python bump_version.py bump minor   # 0.1.2 → 0.2.0
python bump_version.py bump major   # 0.1.2 → 1.0.0
```

## Verification

You can verify that all components use the same version:

```bash
# Test Python imports
python -c "
from testteller import __version__, APP_VERSION
from testteller._version import __version__ as direct_version
print(f'Direct: {direct_version}')
print(f'Package: {__version__}')
print(f'Config: {APP_VERSION}')
"

# Test setup.py extraction
python -c "
import sys
sys.path.insert(0, '.')
from setup import get_version
print(f'Setup.py: {get_version()}')
"

# Build and check wheel version
python -m build --wheel
ls dist/  # Should show testteller-X.Y.Z-py3-none-any.whl
```

## Building and Distribution

The centralized version system is fully compatible with:

- **PyPI uploads via `twine`**
- **`pip install` from source**
- **`python -m build` for wheel/sdist creation**
- **Docker builds**
- **Development installs (`pip install -e .`)**

### Example Build Process

```bash
# 1. Update version
python bump_version.py bump minor

# 2. Build distributions
python -m build

# 3. Upload to PyPI
twine upload dist/*
```

## Benefits

1. **Single Point of Truth**: Version only needs to be changed in one place
2. **No Sync Issues**: Impossible to have mismatched versions across files
3. **Build Tool Compatible**: Works with all standard Python packaging tools
4. **Easy Automation**: Simple to automate version bumping in CI/CD
5. **Developer Friendly**: Clear and obvious where to update the version

## Troubleshooting

### If imports fail:
- Ensure `testteller/_version.py` exists and has `__version__ = "X.Y.Z"`
- Check for circular import issues
- Verify Python path includes the project root

### If builds fail:
- Ensure `setup.py` can find `testteller/_version.py`
- Check that `pyproject.toml` has correct dynamic version configuration
- Verify regex pattern in `setup.py` matches the version format

### For PyPI uploads:
- Build wheel and sdist before uploading: `python -m build`
- Verify version in built files: `ls dist/`
- Test installation: `pip install dist/testteller-X.Y.Z-py3-none-any.whl` 