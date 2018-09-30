##################################################
############### Predictive Models ################
##################################################

rm(list = ls())

setwd("/Users/matthewkeys/Desktop/CRES/Data/SurgeryPrognostics/Cleaned/")

library(caret)        # Predictive Modelling  
library(ggplot2)
library(plm)          # Fixed Effects regression 
library(lfe)          # Fixed Effects regression 
library(gridExtra)
library(grid)
library(lattice)
library(ggpubr)
library(sjPlot)
library(missForest)
library(xgboost)
library(readr)
library(stringr)
library(car)
library(feather)      # Super efficient data transfer from python to R (not meant for long term storage)
library(tidyverse)    # General utility functions

# Load data
knee_wider <- read_feather("knee_wider_wdd.feather")
hip_wider <- read_feather("hip_wider_wdd.feather")
groin_wider <- read_feather("groin_wider_wdd.feather")

knee_wide <- read_feather("knee_wide_wdd.feather")
hip_wide <- read_feather("hip_wide_wdd.feather")
groin_wide <- read_feather("groin_wide_wdd.feather")

knee_reduced <- read_feather("knee_reduced_wdd.feather")
hip_reduced <- read_feather("hip_reduced_wdd.feather")
groin_reduced <- read_feather("groin_reduced_wdd.feather")


############  Wider Predictive Models ############

#### Final stage preprocessing 

# Drop Irrelevant variables for prediction (fixed effects, time indicators or alternative measures of output e.g. Post-Op Score)
knee_pwr <- knee_wdd[ , !(names(knee_wdd) %in% c("PostOp_Q_EQ5D_Index", "PreOp_Q_EQ5D_Index_Profile",
                                                 "Provider_Code", "Knee_Replacement_PostOp_Q_Score",
                                                 "Year", "Year_2014", "Year_2015", "Year_2016",
                                                 "Patient_ID"))]

groin_pwr <- groin_wdd[ , !(names(groin_wdd) %in% c("PostOp_Q_EQ5D_Index", "PreOp_Q_EQ5D_Index_Profile",
                                                    "Provider_Code", "Knee_Replacement_PostOp_Q_Score",
                                                    "Year", "Year_2014", "Year_2015", "Year_2016",
                                                    "Patient_ID"))]

hip_pwr <- hip_wdd[ , !(names(hip_wdd) %in% c("PostOp_Q_EQ5D_Index", "PreOp_Q_EQ5D_Index_Profile", 
                                              "Provider_Code", "Knee_Replacement_PostOp_Q_Score",
                                              "Year", "Year_2014", "Year_2015", "Year_2016",
                                              "Patient_ID"))]

#### Feature Engineering

# Time elapsed since introduction of PROMs data (2009)


# PCA 

####::::: No feature engineering at the moment 


### Test-Train Split
# Get training indexes (75% training, 25% testing)
kneeTrainIndex <- createDataPartition(knee_pwr$EQ5D_Index_Diff, p=0.75, list=FALSE)
hipTrainIndex <- createDataPartition(hip_pwr$EQ5D_Index_Diff, p=0.75, list=FALSE)
groinTrainIndex <- createDataPartition(groin_pwr$EQ5D_Index_Diff, p=0.75, list=FALSE)

# Formumlate training and test sets 
kneeTrain = knee_pwr[kneeTrainIndex, ] 
hipTrain = hip_pwr[hipTrainIndex, ] 
groinTrain = groin_pwr[groinTrainIndex, ] 

kneeTest = knee_pwr[-kneeTrainIndex, ] 
hipTest = hip_pwr[-hipTrainIndex, ] 
groinTest = groin_pwr[-groinTrainIndex, ]

## Seperate inputs and labels & convert to matrices 
# Knee
kneeTrain_data = data.matrix(subset(kneeTrain, select = -c(EQ5D_Index_Diff)))
kneeTrain_labels = data.matrix(kneeTrain["EQ5D_Index_Diff"])

kneeTest_data = data.matrix(subset(kneeTest, select = -c(EQ5D_Index_Diff)))
kneeTest_labels = data.matrix(kneeTest["EQ5D_Index_Diff"])

# Hip
hipTrain_data = data.matrix(subset(hipTrain, select = -c(EQ5D_Index_Diff)))
hipTrain_labels = data.matrix(hipTrain["EQ5D_Index_Diff"])

hipTest_data = data.matrix(subset(hipTest, select = -c(EQ5D_Index_Diff)))
hipTest_labels = data.matrix(hipTest["EQ5D_Index_Diff"])

# Groin
groinTrain_data = data.matrix(subset(groinTrain, select = -c(EQ5D_Index_Diff)))
groinTrain_labels = data.matrix(groinTrain["EQ5D_Index_Diff"])

groinTest_data = data.matrix(subset(groinTest, select = -c(EQ5D_Index_Diff)))
groinTest_labels = data.matrix(groinTest["EQ5D_Index_Diff"])

# Convert training & test data to XGboost optimised sparse matrices
knee_dtrain <- xgb.DMatrix(data = kneeTrain_data , label = kneeTrain_labels)
hip_dtrain <- xgb.DMatrix(data = hipTrain_data , label = hipTrain_labels)
groin_dtrain <- xgb.DMatrix(data = groinTrain_data , label = groinTrain_labels)

knee_dtest <- xgb.DMatrix(data = kneeTest_data , label = kneeTest_labels)
hip_dtest <- xgb.DMatrix(data = hipTest_data , label = hipTest_labels)
groin_dtest <- xgb.DMatrix(data = groinTest_data , label = groinTest_labels)

####### XGBoost #######

# Convert training & test data to XGboost optimised sparse matrices
knee_dtrain <- xgb.DMatrix(data = kneeTrain_data , label = kneeTrain_labels)
hip_dtrain <- xgb.DMatrix(data = hipTrain_data , label = hipTrain_labels)
groin_dtrain <- xgb.DMatrix(data = groinTrain_data , label = groinTrain_labels)

knee_dtrain <- xgb.DMatrix(data = kneeTest_data , label = kneeTest_labels)
hip_dtrain <- xgb.DMatrix(data = hipTest_data , label = hipTest_labels)
groin_dtrain <- xgb.DMatrix(data = groinTest_data , label = groinTest_labels)

### Knee 

## Set up grid search 
##########::: Come back to this

## Optimise 
xgb_knee <- xgboost(data = knee_dtrain,
                    eta = 0.1,
                    max_depth = 15, 
                    nround=25, 
                    subsample = 0.5,
                    colsample_bytree = 0.5,
                    seed = 1,
                    eval_metric = "merror",
                    objective = "multi:softprob",
                    num_class = 12,
                    nthread = 3
)

## Predict 
knee_pred <- predict(xgb_knee, data.matrix(X_test[,-1]))

## Present & Save Results 


###

