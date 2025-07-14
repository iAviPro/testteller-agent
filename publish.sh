#!/bin/bash

# TestTeller Package Publishing Script using Twine
set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

info() { echo -e "${BLUE}[INFO]${NC} $1"; }
success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ] || [ ! -f "testteller/_version.py" ]; then
    error "Please run this script from the project root directory"
    exit 1
fi

# Show current version
VERSION=$(python -c "from testteller._version import __version__; print(__version__)")
info "Current version: $VERSION"

# Install required packages
info "Installing/updating build tools..."
pip install --upgrade build twine

# Clean previous builds
info "Cleaning previous build artifacts..."
rm -rf build/ dist/ *.egg-info/ testteller.egg-info/

# Build the package
info "Building package with setuptools..."
python -m build

if [ $? -ne 0 ]; then
    error "Build failed"
    exit 1
fi

success "Package built successfully"
ls -la dist/

# Check package with twine
info "Checking package integrity..."
python -m twine check dist/*

if [ $? -ne 0 ]; then
    error "Package check failed"
    exit 1
fi

success "Package integrity check passed"

# Function to publish
publish_to_repository() {
    local repo=$1
    local repo_name=$2
    
    info "Publishing to $repo_name..."
    
    if [ "$repo" = "testpypi" ]; then
        python -m twine upload --repository testpypi dist/*
    else
        python -m twine upload dist/*
    fi
    
    if [ $? -eq 0 ]; then
        success "Successfully published to $repo_name"
        
        if [ "$repo" = "testpypi" ]; then
            info "Test installation command:"
            echo "pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ testteller==$VERSION"
        else
            info "Installation command:"
            echo "pip install testteller==$VERSION"
        fi
    else
        error "Failed to publish to $repo_name"
        return 1
    fi
}

# Parse arguments
if [ "$1" = "--test-only" ] || [ "$1" = "-t" ]; then
    # Test PyPI only
    publish_to_repository "testpypi" "Test PyPI"
elif [ "$1" = "--prod-only" ] || [ "$1" = "-p" ]; then
    # PyPI only
    warning "Publishing directly to PyPI!"
    read -p "Are you sure? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        publish_to_repository "pypi" "PyPI"
    else
        info "Cancelled"
    fi
elif [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -t, --test-only    Publish to Test PyPI only"
    echo "  -p, --prod-only    Publish to PyPI only"
    echo "  -h, --help         Show this help"
    echo ""
    echo "Default: Publish to Test PyPI first, then ask for PyPI"
else
    # Default: Test PyPI first, then PyPI
    publish_to_repository "testpypi" "Test PyPI"
    
    echo ""
    warning "Publish to production PyPI?"
    read -p "Continue to PyPI? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        publish_to_repository "pypi" "PyPI"
    else
        info "Stopped at Test PyPI"
    fi
fi

success "Publishing process completed!"





######################################## STEPS TO RUN THE SCRIPT ########################################

#cd /Users/aviral/code/github/testteller-rag-agent

#chmod +x publish.sh

## Test PyPI first, then prompt for PyPI
#./publish.sh

# Test PyPI only
#./publish.sh --test-only

# PyPI only (production)
#./publish.sh --prod-only

# Help
#./publish.sh --help