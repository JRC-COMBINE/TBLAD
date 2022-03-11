## Tree-Based Learning on Amperometric Time Series Data (TBLAD)

TBLAD is a software package implemented in Python for analysis of diverse amperometric datasets using well-established data-driven approaches in computational science. Specifically, it uses a standard Machine Learning pipeline, wherein several hundred features calculated from entire amperometric time series data enables very high accuracy predictions and novel insights into the datasets. This includes an end-to-end systematic machine learning workflow for amperometric time series datasets consisting of pre-processing; feature extraction; model identification; training and testing; followed by feature importance evaluation. Given that tree-based classifiers emerged as the top family of supervised learning models in our analysis, across different experiments, chemical conditions and cell types, this implementation uses Random Forest as a default classifier. The choice of the classifier was also motivated by the highly parallel structure, where each tree can be independently trained and evaluated, leading to an optimal training due to the simplicity of the constitutent decision tree classifiers. Further, the classifier relies completely on the characteristics of the datasets and makes few or no assumptions on the underlying exocytosis process. TBLAD provides an universal method for classification and to our knowledge, is one of the first packages that propose a scheme for supervised learning on full amperometry time series datasets.

## How do I use TBLAD?


## Module Descriptions


## OS Compatibility

The TBLAD package is OS-independent. A fully functioning Python environment is sufficient to use this package. In case you run into incompatibility issues, please check the dependency list below.

## Dependencies

## Installation Instructions

## License

This is an open source software and is licensed under LGPL.

## Getting help

For queries regarding the software write to: zeyu.lian@rwth-aachen.de/ krishnan@aices.rwth-aachen.de

