# setup.py for the testteller-rag-agent package
"""
setup.py
This script sets up the TestTeller RAG Agent package, including dependencies,
entry points, and metadata. It uses setuptools for packaging.
"""
import pathlib
from setuptools import setup, find_packages


# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

setup(
    name="testteller-rag-agent",
    version="0.1.0-alpha",
    description="AI Test Case Generation Agent (Production-Ready Version).",
    long_description=README,
    long_description_content_type="text/markdown",
    author="Aviral Nigam",
    packages=find_packages(exclude=["tests", "tests.*"]),
    install_requires=[
        "aiofiles>=23.1.0",
        "aiohttp>=3.8.1",
        "asyncio>=3.4.3",
        "docx2txt>=0.8",
        "fitz>=1.18.19",
        "openpyxl>=3.0.9",
        "typer>=0.4.0",
        "pydantic>=1.10.2",
        "requests>=2.26.0",
        "python-dotenv>=0.19.2",
        "styling==0.1.0-alpha",  # Ensure this matches your local version
        "constants==0.1.0-alpha"  # Ensure this matches your local version
    ],
    entry_points={
        "console_scripts": [
            "testteller-rag-agent=testteller_rag_agent.main:app"
        ]
    },
    include_package_data=True,
    python_requires=">=3.8",
    license="MIT",  # Specify your license
)
