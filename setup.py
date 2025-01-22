from setuptools import setup, find_packages
import pathlib

def read_requirements(filename):
    """Parse a requirements.txt file into a list of dependencies."""
    with open(filename, encoding="utf-8") as f:
        return f.read().splitlines()

current_dir = pathlib.Path(__file__).parent.resolve()
long_description = (current_dir / "README.md").read_text(encoding="utf-8")

setup(
    name="gemastik",
    version="0.1.0",
    author="Fauzan Ghaza Madani",
    author_email="contact@fauzanghaza.com",
    description="A project for efficient data loading and processing.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ghazafm/gelatik-datamining-2025",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.7",
    install_requires=read_requirements("requirements.txt"),
    extras_require={
        "dev": [
            "pytest>=7.0",
            "black>=22.0",
            "flake8>=4.0",
            "mypy>=0.9",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx_rtd_theme>=1.0",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    keywords="deep-learning, data-processing, pytorch",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/deep-learning/issues",
        "Source Code": "https://github.com/yourusername/deep-learning",
    },
    # entry_points={
    #     "console_scripts": [
    #         "my_tool=my_package.module:main_function",
    #     ],
    # },
    include_package_data=True,
    zip_safe=False,
)