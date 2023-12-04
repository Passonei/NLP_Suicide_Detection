# NLP_Suicide_Detection
A repository focusing on natural language processing techniques applied to posts labeled "suicide" and "non-suicide." The project includes data analysis, pre-processing, feature engineering and implementation of a classification model.

##### Analysis
The database comes from kaggle (https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch/data)  
contains about 250,000 posts belonging to one of two classes "suicide" or "non_suicide," the data is balanced.

##### Classification
A number of experiments were conducted with different classifiers, preprocessing methods, hyperparameters and additional features.  
Classifiers tested: Naive Bayes, KNN, SVM, Random Forest, Logistic Regression. The best performance in terms of accuracy with reasonable running time was shown by the Logistic Regression model. The experiments also identified the most promising preprocessing procedure. A favorable effect on the quality of classification of new created features such as length and sentiment was also noted.

Based on the experiments, an optimal pipeline was created:
1. Preprocessing:
- lowercase, 
- removal of punctuation marks, 
- removal of links and numbers, 
- converting emoji to text, 
- tokenization, 
- remove stopwords, 
- lemmatization.
2. Feature extraction: 
- calculation of document length,
- calculation of document sentiment,
- text vectorization.
3. Classification.

###### Results
The classifier achieves 94.19% accuracy on the test set. Which is a higher score than those used on kaggle by other users of more advanced models like recurrent neural networks (RNNs).
                precision   recall   f1-score  support

           0       0.94      0.95      0.94     23301
           1       0.95      0.93      0.94     23104

    accuracy                           0.94     46405

