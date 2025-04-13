from setuptools import setup, find_packages

setup(
    name="causal_survey_analyzer",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.0.0",
        "numpy>=1.18.0",
        "matplotlib>=3.1.0",
        "statsmodels>=0.11.0",
        "scipy>=1.4.0",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="An object-oriented package for analyzing survey data to measure causal effects across various experimental designs",
    long_description=open("README.md").read() if hasattr(__file__, "parent") else "",
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/causal_survey_analyzer",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.6",
) 