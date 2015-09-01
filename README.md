# Kaggle - Caterpillar Tube Assembly Pricing Prediction

### Final Standing
- 133/1323 (top 10%) on the private leaderboard

### Problem Statement
- The goal of this competition was to predict the pricing quotes by suppliers for various tube assemblies. Caterpillar relies on a variety of suppliers to manufacture tube assemblies, with each supplier having their own unique pricing model. 
- The objective was to predict the supplier price quote for a tube assembly given the detailed tube, component, and annual volume data.

### Files 
- DWrangling_EDA.Rmd - Table joins, long form to wide form feature transformation of tube components, specs, and bill of materials, end form transformation etc
- regressor_v2.py - TFIDF transform + Truncated SVD on wide form sparse features, gradient boosting regressor model on engineered features 
- XGBoost_regressor.Rmd - Gradient Boosting with the XGBoost model.
- 14model_avg_v2.py - Post processing script to perform weighted average prediction of 14 different models from the boosted model library.

### Models and Tuning
- Since there were a lot of different tables, data wrangling involved a number of table joins, discarding some id columns and transforming long form to wide form features to improve prediction
- Best performing model was a Gradient Boosting Decision Tree regressor among the different algorithms tested
- Key insight was to use different subset of features for each model to improve model diversity and reduce errors for difficult to predict datapoints by ensembling

### Feature Engineering
- Engineered feature to determine volume of tube material based on length, inner and outer radius
- Engineered components, spec, and bill of materials count features per tube 
- Engineered long form to wide form sparse features followed by TFIDF + Truncated SVD to reduce number of features for training and prediction
- Built different models with different subset of features to improve model diversity
- Some models built separate boosted trees for bracketed and non-bracketed pricing data to further improve model diversity

### Final Model 
- Weighted average of 14 gradient boosted tree models 
    - Weights are based on public leaderboard submission scores
- Each model was built on different subset of features to improve model diversity 
    - Ensemble provided a 5% boost over single best performing model as a result

### What didn't work
- Random Forest Regressor and Support Vector Regressor models gave very poor predictions and were rejected for the final ensemble

### Could have tried
- Due to the size of the dataset, number of features and laptop hardware limitations, I could not perform principled K-Fold validation with grid search for model hyper parameter tuning.
     - Doing so might have improved final ranking by further 1-2% possibly
 



