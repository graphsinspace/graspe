# graspe
Graphs in Space: Graph Embeddings for Machine Learning on Complex Data

This repository contains code for Graph Embedding evaluation for the [GRASP](https://graphsinspace.net/) project.

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

## Installation

   We recommend using this library by installing conda via conda-env.yml file. You can do that in two steps:
   - a. Conda: conda env create -f conda-env.yml && conda activate graspe
   - b. pip install -r requirements.txt (In `clear` environment)

## Examples

Running tests

    cd src/graspe && pytest -rP

Generate pydoc

    cd src/graspe
    pydoc -w `find . -name '*.py'`
    mv *.html ../../reports/doc

Graph Auto Encoders

    python src/graspe embed gae -g karate_club_graph -d 10 
    python src/graspe embed gae -g karate_club_graph -d 10 --variational # to use VAE 

Example of graph embedding file
    
    out.embedding

## Authors

(c) 2020 UNSPMF

## License and Acknowledgements

- GNU General Public License v3.0
- This research (library) is supported by the Science Fund of the Republic of Serbia, \#6518241, AI -- GRASP.
