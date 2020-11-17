# High Risk Credit Analysis :credit_card:
### Using Machine Learning to Predict Good and Bad Loans 

## Overview of the Credit Risk Analysis

The purpose of this analysis is to create various machine learning models to predict credit risk levels. Credit card risk has historically had unbalanced classification problems, so this analysis contains imbalanced-learn and scikit libraries are used to train models to level out unbalanced data. The first part of the analysis involves RandomOverSampler and SMOTE algorithms to oversample the data. Then these results are compared to a ClusterCentroids model to undersample the data. Lastly, we compare all the models with the SMOTEEN model that provides a combination of under and oversampling to see if this is more accurate at predicting credit risk. In the second part of the analysis, we use BalancedRandomForestClassifier and EasyEnsembleClassifier models that help to reduce bias within the dataset to estimate credit risk. 

The tools used in this analysis are: Python with Machine Learning Models in Jupyter Notebook. 

## Results 

1. Naive Random Oversampling 

In an oversampling model, sample data points are randomly selected and duplicated to balance the number of points between the larger and smaller classes. In this instance, the smaller class isn't under represented, but evenly balanced with the larger class. 
The first model - Naive Random Oversampling - does not provide very strong credit risk predictions. Overall, it only has a 63% accuracy score and 2% F1 score for high risk credit scores. It has low recall (sensitivity) and precision values as well, which means it is not accurate or reliable at predicting a high credit risk. 

-- insert screen shot

2. Smote Oversampling 

In the Smote Oversampling model, the logic is similar to the Naive Random Oversampling model, however it puts more value on which data points are oversampled in order to balance out the classes. It takes data points that are closer to the middle, rather than taking any outlier data points. This helps centralize the data samples and avoid any outliers than can skew the data. 

-- insert screen shot

Unfortunately, using the Smote Oversampling model does not improve the accuracy levels. Overall, it remains at 63% accuracy level and 2% F1 score for high risk credit levels. The recall or sensitivity level did increase from 59% to 62%, meaning that it does slightly improve the number of high credit risks that are accurately predicted. 

3. Cluster Centroids Undersampling 

In the third model, we use the Cluster Centroids Undersampling algorithm to predict credit risks. Undersampling models differ from the Naive Random and Smote Oversampling models because they actually just use real data points in the data set. Instead of increasing the smaller class like above, this model reduces the amount of the points in the larger class to equalize it with the smaller data. 

-- insert screen shot 

Even though this model only uses real data points in the dataset, the accuracy level significantly decreases from the oversampling methods. The accuracy level for the Cluster Centroids model dropped to 52% and a high risk F1 score of 1%. This is the worst algorithm to predict credit risk. 

4. Smoteenn Oversampling and Undersampling 

The fourth model of our machine learning analysis is the Smoteenn algorithm that combines over and undersampling. It oversamples like the Smote model and eliminates points that are outliers, and it also undersamples as it drops points that are too close to both classes. This model tries to further define boundaries between the two classes. 

-- insert screen shot 

This model also doesn't significantly improve the accuracy levels. This falls in line with the Naive Random Oversampling and Smote Oversampling levels at 64% accuracy rate. 

5. Balanced Random Forest

The following two models - Easy Ensemble Classifier and Balanced Random Forest - help to reduce bias. 

-- insert screen shot 

This model does increase accuracy over the other four. The level jumped to an 82% accuracy level. While this is much better than the previous models, an 82% is still a mediocre level for a machine learning model. Most would argue an 95% or higher would be the standard level for these models. 

6. Easy Ensemble Classifier

-- insert screen shot

The Easy Ensemble Classifier model is the best of all the models! The accuracy level of this algorithm comes in at 93%! Still, not the most accurate model - but significantly better than all other five options we used in our analysis. This model also increased the high risk recall level to 91%, which is much improved. 

## Summary 

In conclusion, the Easy Ensemble Classifier model is the best at predicting high credit risk. With a 93% accuracy level, it outperforms all of the other models. It is slightly below the 95% standards that most models are held against, but it is definitely more successful at predicting risk. 
