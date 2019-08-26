# About the Project
The main goal of this project was to predict average response time of the Los Angeles Fire Department. The evaluation metric was MSE.

# Methodology
* Imported external data on district information from [LA Times](https://www.latimes.com/) to get district information of LA
* Engineered new features using regular expression, aggregation and etc.
* Selected features through repeatedly adding features to baseline model and see which one contributed the most
* Implemented regression XGBoost with selected 10 features
  * 6 were original features, 4 were newly created
* Tuned hyperparameter with parallel mapping (hyperparameter tuned: eta, nrounds, max_depth)
* 10-fold cross validation in parallel to reduce overfitting
* Postprocessed predictions by removing negative values
* **Ranked 3rd/92 teams**

# Further Details
For more information: 
- [Markdown](https://github.com/jhuo831alex/Kaggle-Competition-101C/blob/master/Project_Report.pdf) 
- [Deck](https://github.com/jhuo831alex/Kaggle-Competition-101C/blob/master/Presentation_Deck.pdf)


