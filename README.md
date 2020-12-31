# graspe
Graphs in Space: Graph Embeddings for Machine Learning on Complex Data

This repository contains code for Graph Embedding evaluation for the GRASP project.

## Structure

    graspe
    ├── conda-env.yml       -> conda package manager environment definition
    ├── data                -> dataset files
    ├── LICENSE             -> project license file
    ├── models              -> saved models that can be loaded from disk
    ├── notebooks           -> jupyter notebooks
    ├── README.md           -> project readme
    ├── reports             -> documentation, instructions, manuals, reports, plots, etc.
    ├── requirements.txt    -> pip package manager environment definition
    └── src/{graspe,tests}  -> source code files, and test files

## Requirements

## Installation

## Examples

Running tests

    cd src/graspe && pytest -rP

Generate pydoc

    cd src/graspe
    pydoc -w `find . -name '*.py'`
    mv *.html ../../reports/doc

## Authors

(c) 2020 UNSPMF

## License and Acknowledgements
