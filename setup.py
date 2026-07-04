from setuptools import setup, find_packages

setup(
    name="deep_search",
    version="0.1.0",
    packages=find_packages(include=['deep_search', 'deep_search.*']),
    install_requires=[],
    author="",
    author_email="",
    description="Search tool with GPT integration",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
