# Multi Modal Time Series Classification
This repository contains the code for my master's thesis.

**Abstract**

Throughout human history, patterns that repeat or follow a trend over time have always been a source of curiosity and study. Over the years, the development of sciences, computational power, and the availability of temporal data have advanced the use of time series for many purposes. One example of this is the study of time series classification (TSC), which is considered one of the most challenging problems in data mining. The goal is to relate time series to specific labels or categories. Given the richness of details that natural phenomena can contain, it is unlikely that a single modality can describe them completely or satisfactorily. As the availability of databases containing different modalities and the context of multimodal learning have grown, the study of how to fuse different modalities within a neural network and handle the differences between them has become increasingly necessary. This work aims to study various aspects of time series classification (TSC) in the context of multimodality. Specifically, we seek to answer the following questions: Can fusion models improve TSC if multivariate time series from different sources are treated as distinct modalities? What types of time series fusion can enhance model performance?

**Research Goals**

This research focuses on enhancing the classification of multivariate time series through multimodal deep learning approaches, guided by a main hypotheses: treating each variable in a multivariate time series as a distinct modality can improve classification model performance. The primary objectives of this research include developing a framework to adapt multivariate time series for classification by treating different series as separate modalities. Additionally, the research aims to compare the performance of leading multivariate time series classification methods with those of models utilizing multimodality. Another key objective is to evaluate the performance of these classification models when integrating data at various stages, early, intermediate, and late fusion, in a multimodal context.


**Dataset**

In this research, we are utilizing the UEA multivariate time series classification archive, a collaborative effort between researchers at the University of East Anglia (UEA) and the University of California, Riverside (UCR), aimed at building an archive for multivariate time series classification similar to UCR's univariate archive. This archive contains 30 multivariate time series datasets with a wide range of cases, dimensions, and series lengths, including Human Activity Recognition, Motion Classification, ECG Classification, EEG/MEG Classification, among others.


**Code**

The code is organized as follows:
- Each experiment has its own `main_.py` file
- The code execute the function `training_nn_for_seeds` that executes the training for a defined number of random seeds.
