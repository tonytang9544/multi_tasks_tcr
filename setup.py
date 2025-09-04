from setuptools import setup, find_packages

setup(
    name="multi_tasks_tcr",          # Package name
    version="0.1.0",                   # Version number
    author="Your Name",
    author_email="your.email@example.com",
    description="A short description of your package",
    long_description=open("README.md").read(),  # Detailed description from README
    long_description_content_type="text/markdown",
    url="https://github.com/",
    packages=find_packages(),          # Automatically find packages in your project
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  # License info
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[                 # List your package dependencies here
        "numpy",
        "torch",
        "pandas",
    ],
    entry_points={                    # If your package includes command-line scripts
        "console_scripts": [
            "your-command=your_package.module:function",
        ],
    },
)
