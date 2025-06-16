# setup.py for the testteller-rag-agent package
"""
setup.py
This script sets up the TestTeller RAG Agent package, including dependencies,
entry points, and metadata. It uses setuptools for packaging.
"""
import pathlib
from setuptools import setup, find_packages

# Function to read the requirements from requirements.txt


def parse_requirements(filename="requirements.txt"):
    """Load requirements from a pip requirements file."""
    with open(filename, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


   # The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()


# Package version (should ideally be managed in one place, e.g., a version.py or __init__.py)
# For simplicity here, hardcoded. In a real setup, you might read it from testteller_rag_agent/__init__.py
VERSION = "0.1.2"  # Make sure this matches your intended version


setup(
    name="testteller",
    version=VERSION,
    description="TestTeller : A versatile RAG AI agent for generating test cases from project documentation (PRDs, Contracts, Design Docs, etc.) and project code, leveraging LLMs.",
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
