# NLP_Suicide_Detection
A repository focusing on natural language processing techniques applied to posts labeled "suicide" and "non-suicide." The project includes data analysis, pre-processing, feature engineering and implementation of a classification model.

##### Analysis
The database comes from kaggle (https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch/data)  
contains about 250,000 posts belonging to one of two classes "suicide" or "non_suicide," the data is balanced.

__Length__  
Texts suggesting suicidal thoughts are characterized by longer posts, which may indicate a more comprehensive expression of emotions and thoughts. In the 'non-suicide' class, posts are much shorter, suggesting simpler and more concise communication.

__Sentiment__  
Suicide-related texts show a higher level of negative sentiment (26.53%) compared to non-suicide-related texts (15.47%). This suggests that texts related to suicide contain more negative content. Texts unrelated to suicide have a slightly higher level of neutral sentiment (62.95%) compared to texts related to suicide (54.07%). Texts related to suicide contain more neutral content. The level of positive sentiment is similar for both categories.  
Total Indicator (Compound): The average composite index is positive for non-suicide-related texts (12.41%), indicating a slightly positive trend. In contrast, for texts related to suicide, the index is significantly negative (-35.38%), suggesting a higher level of overall negative sentiment.  
Sentiment analysis shows that texts related to suicide contain more negative content compared to unrelated texts, which show a slightly higher level of positive sentiment and a lower level of overall negative sentiment.

__Thematics__  
Texts related to suicide:
1. Emotional themes: These texts are dominated by emotions of desperation, hopelessness, requests for help and mentions of death.
2. Specific vocabulary: Contains words related to suicide methods, medications, and mental and emotional symptoms associated with suicidal thoughts.  
3. Emptiness and desperation: They often mention emptiness, lack of meaning in life, and the desire to end life.

Texts not related to suicide:
1. Cultural Topics: Focus on internet culture, humor, interpersonal relationships, and popular trends.
2. No mentions of suicide: They do not contain specific words related to suicidal thoughts or emotional aspects related to this topic.
3. Everyday life and entertainment: They cover a variety of areas, from technology to everyday conversations and Internet interests.

Sentiment and thematic analysis confirm that texts related to suicide contain more negative content than those unrelated to it. Texts unrelated to suicide show slightly higher levels of positive sentiment and lower overall negative sentiment. The topics of suicide-related texts focus on emotions, suicide-related vocabulary and desperation, while unrelated texts focus on everyday life and online culture.

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

