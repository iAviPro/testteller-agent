# setup.py for the testteller-rag-agent package
"""
setup.py
This script sets up the TestTeller RAG Agent package, including dependencies,
entry points, and metadata. It uses setuptools for packaging.
"""
import pathlib
import re
from setuptools import setup, find_packages

# Function to read the requirements from requirements.txt


def parse_requirements(filename="requirements.txt"):
    """Load requirements from a pip requirements file."""
    with open(filename, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


def get_version():
    """Extract version from testteller/_version.py"""
    version_file = pathlib.Path(__file__).parent / "testteller" / "_version.py"
    with open(version_file, "r", encoding="utf-8") as f:
        content = f.read()

    version_match = re.search(
        r"^__version__ = ['\"]([^'\"]*)['\"]", content, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError(
        "Unable to find version string in testteller/_version.py")

   # The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# Get version from single source of truth
VERSION = get_version()


setup(
    name="testteller",
    version=VERSION,
    description="TestTeller : A versatile RAG AI agent for generating test cases from project documentation and code, supporting multiple LLM providers including Gemini, OpenAI, Claude, and Llama.",
    long_description=README,
    long_description_content_type="text/markdown",
    author="Aviral Nigam",
    url="https://github.com/iAviPro/testteller-rag-agent",
    project_urls={
        "Bug Tracker": "https://github.com/iAviPro/testteller-rag-agent/issues"
    },
    license="Apache License 2.0",
    entry_points={
        # The entry point should point to the app_runner function
        # which handles the initial logging setup before calling app().
        "console_scripts": [
            "testteller=testteller.main:app_runner"
        ]
    },
    packages=find_packages(
        exclude=["tests", "tests.*", ".github", ".github.*"]),
    include_package_data=True,
    install_requires=parse_requirements(),
    python_requires=">=3.11",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Information Technology",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Testing",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords=["testing", "rag", "llm", "generative ai", "test case generation", "qa", "automation", "testcase",
              "testteller", "ai testing", "rag agent", "knowledge base", "document ingestion", "code ingestion", "testteller-rag-agent", "testteller rag agent", "testteller_rag_agent"]

)
