## Tree-Based Learning on Amperometric Time Series Data (TBLAD)

TBLAD is a software package implemented in Python for analysis of diverse amperometric datasets using well-established data-driven approaches in computational science. Specifically, it uses a standard Machine Learning pipeline, wherein several hundred features calculated from entire amperometric time series data enables very high accuracy predictions and novel insights into the datasets. This includes an end-to-end systematic machine learning workflow for amperometric time series datasets consisting of pre-processing; feature extraction; model identification; training and testing; followed by feature importance evaluation. Given that tree-based classifiers emerged as the top family of supervised learning models in our analysis, across different experiments, chemical conditions and cell types, this implementation uses Random Forest as a default classifier. The choice of the classifier was also motivated by the highly parallel structure, where each tree can be independently trained and evaluated, leading to an optimal training due to the simplicity of the constitutent decision tree classifiers. Further, the classifier relies completely on the characteristics of the datasets and makes few or no assumptions on the underlying exocytosis process. TBLAD provides an universal method for classification and to our knowledge, is one of the first packages that propose a scheme for supervised learning on full amperometry time series datasets.

**Why should I use TBLAD?**

Firstly, a key aspect of the amperometry datasets is their sub-millisecond temporal resolution of the current recordings which usually leads to several gigabytes of high-quality data. This makes it an excellent use case for Machine Learning models. Secondly, state-of-the-art analysis tools in the amperometry community make inferences on these datasets largely based on spike properties alone. However, transients between spikes, trace baselines and other advanced features extracted from temporal structure of spikes carry rich information about the datasets. 

TBLAD was tested on heterogeneous amperometric time series datasets generated using different experiemental approaches, chemical stimulations, electrode types and varying recording times. We found that using full time-series features as compared to just spike features from these datasets boosts the classification accuracy from 30% to 95%. Therefore, an effective feature representation of amperometric time series requires the full time series and the important features identified by this implementation may aid in uncovering partially unknown exocytosis mechanisms. Lastly, all libraries used in this package are easy-to-use and open-source. 

**How should I use TBLAD?**

The classification scripts should be executed in the following order:

* `preprocessor` splits the raw data into chunks of length chunk_size and assign each chunk with a label. The output .pkl files are stored unter `./dataframe/`
* `train` extracts `tsfresh` features from each chunk, the output is written to `./Features/`. Note that this script requires high computational power, recommended to run on the server.
* `evaluation` trains the decision-tree classifiers (including random forest, extra trees and xgboost) and returns the following: (a) accuracy and standard deviation, (b) top features with importance values, (c) heat map of cross-correlation, (d) bar plot of the feature importances, (e) a visualization of one of the decision trees, (f) list of relevant features and their importance values as `csv` file under `Features/` 

`params` has  global settings for the classification. Most important parameters are: data path, chunk size (please select this based on your computational capacity), number of forests (to provide information on the standard deviation of the accuracy), number of decision trees per forest(required for RF and ET). `visualization` provides an visualization of the raw time series data. The path name should be changes by future users under `params`.
   
**OS Compatibility**

The TBLAD package is OS-independent. A fully functioning Python environment is sufficient to use this package. In case you run into incompatibility issues, please check the dependency list below.

**Dependencies**

* [Numpy](https://numpy.org/install/)
* [Scipy](https://scipy.org/install/)
* [Pandas](https://pypi.org/project/pandas/)
* [Matplotlib](https://matplotlib.org/stable/users/installing/index.html)
* [Scikit-Learn](https://scikit-learn.org/stable/)
* [Seaborn](https://seaborn.pydata.org/installing.html)
* [Tsfresh](https://tsfresh.readthedocs.io/en/latest/)
* [Lazypredict](https://pypi.org/project/lazypredict/)

**Installation Instructions**

* Clone the repository to your local machine or the cluster using `git clone `
* Install dependency packages as necessary

**License**

This is an open source software and is licensed under LGPL.

**Getting help**

For queries regarding the software write to: zeyu.lian@rwth-aachen.de/ krishnan@aices.rwth-aachen.de

