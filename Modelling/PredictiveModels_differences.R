################################################################
############### Predictive Models (Differences) ################
################################################################

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
kneeTrain <- read_feather("kneeTrain.feather")
kneeTest <- read_feather("kneeTest.feather")

hipTrain <- read_feather("hipTrain.feather")
hipTest <- read_feather("hipTest.feather")

groinTrain <- read_feather("groinTrain.feather")
groinTest <- read_feather("groinTest.feather")

## Drop Irrelevant variables for prediction (fixed effects, time indicators or alternative measures of output e.g. Post-Op Score)

# knee data
kneeTrain <- kneeTrain[ , !(names(kneeTrain) %in% c("PostOp_Q_EQ5D_Index", "PreOp_Q_EQ5D_Index_Profile", "EQ5D_Change_Discrete",  
                                                 "Provider_Code", "Knee_Replacement_PostOp_Q_Score",
                                                 "Year", "Year_2014", "Year_2015", "Year_2016",
                                                 "Patient_ID", "PreOp_Q_Symptom_Period_1", "PreOp_Q_Symptom_Period_2",
                                                 "PreOp_Q_Symptom_Period_3", "PreOp_Q_Symptom_Period_4",
                                                 "Knee_Replacement_Participation_Rate", "Knee_Replacement_Linkage_Rate",
                                                 "Knee_Replacement_Issue_Rate", "Knee_Replacement_Response_Rate"))]

kneeTest <- kneeTest[ , !(names(kneeTest) %in% c("PostOp_Q_EQ5D_Index", "PreOp_Q_EQ5D_Index_Profile", "EQ5D_Change_Discrete",  
                                                    "Provider_Code", "Knee_Replacement_PostOp_Q_Score",
                                                    "Year", "Year_2014", "Year_2015", "Year_2016",
                                                    "Patient_ID", "PreOp_Q_Symptom_Period_1", "PreOp_Q_Symptom_Period_2",
                                                    "PreOp_Q_Symptom_Period_3", "PreOp_Q_Symptom_Period_4",
                                                    "Knee_Replacement_Participation_Rate", "Knee_Replacement_Linkage_Rate",
                                                    "Knee_Replacement_Issue_Rate", "Knee_Replacement_Response_Rate"))]


# groin data
groinTrain <- groinTrain[ , !(names(groinTrain) %in% c("PostOp_Q_EQ5D_Index", "PreOp_Q_EQ5D_Index_Profile", "EQ5D_Change_Discrete", 
                                                    "Provider_Code", "Groin_Hernia_PostOp_Q_Score",
                                                    "Year", "Year_2014", "Year_2015", "Year_2016",
                                                    "Patient_ID", "PreOp_Q_Symptom_Period_1", "PreOp_Q_Symptom_Period_2",
                                                    "PreOp_Q_Symptom_Period_3", "PreOp_Q_Symptom_Period_4",
                                                    "Groin_Hernia_Participation_Rate", "Groin_Hernia_Linkage_Rate",
                                                    "Groin_Hernia_Issue_Rate", "Groin_Hernia_Response_Rate"))]

groinTest <- groinTest[ , !(names(groinTest) %in% c("PostOp_Q_EQ5D_Index", "PreOp_Q_EQ5D_Index_Profile", "EQ5D_Change_Discrete", 
                                                       "Provider_Code", "Groin_Hernia_PostOp_Q_Score",
                                                       "Year", "Year_2014", "Year_2015", "Year_2016",
                                                       "Patient_ID", "PreOp_Q_Symptom_Period_1", "PreOp_Q_Symptom_Period_2",
                                                       "PreOp_Q_Symptom_Period_3", "PreOp_Q_Symptom_Period_4",
                                                       "Groin_Hernia_Participation_Rate", "Groin_Hernia_Linkage_Rate",
                                                       "Groin_Hernia_Issue_Rate", "Groin_Hernia_Response_Rate"))]

# hip data                       
hipTrain <- hipTrain[ , !(names(hipTrain) %in% c("PostOp_Q_EQ5D_Index", "PreOp_Q_EQ5D_Index_Profile", "EQ5D_Change_Discrete", 
                                              "Provider_Code", "Hip_Replacement_PostOp_Q_Score",
                                              "Year", "Year_2014", "Year_2015", "Year_2016",
                                              "Patient_ID", "PreOp_Q_Symptom_Period_1", "PreOp_Q_Symptom_Period_2",
                                              "PreOp_Q_Symptom_Period_3", "PreOp_Q_Symptom_Period_4",
                                              "Hip_Replacement_Participation_Rate", "Hip_Replacement_Linkage_Rate",
                                              "Hip_Replacement_Issue_Rate", "Hip_Replacement_Response_Rate"))]

hipTest <- hipTest[ , !(names(hipTest) %in% c("PostOp_Q_EQ5D_Index", "PreOp_Q_EQ5D_Index_Profile", "EQ5D_Change_Discrete", 
                                                 "Provider_Code", "Hip_Replacement_PostOp_Q_Score",
                                                 "Year", "Year_2014", "Year_2015", "Year_2016",
                                                 "Patient_ID", "PreOp_Q_Symptom_Period_1", "PreOp_Q_Symptom_Period_2",
                                                 "PreOp_Q_Symptom_Period_3", "PreOp_Q_Symptom_Period_4",
                                                 "Hip_Replacement_Participation_Rate", "Hip_Replacement_Linkage_Rate",
                                                 "Hip_Replacement_Issue_Rate", "Hip_Replacement_Response_Rate"))]


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

### Set up Evaluation functions 
R2 <- function(y_pred, y_actual) {
  
  R2 = 1 - (sum((y_actual - y_pred)**2))/(sum((y_actual-mean(y_actual))**2))
  
  return(R2)
  
}

AdjR2 <- function(R2, p, N) {
  
  AdjR2 = 1 - ((1-R2)*(N-1))/(N - p - 1)
  
  return(AdjR2)
  
}

#######################
####### XGBoost #######
#######################


### Now with extensive hyperparameter optimisation ###

# Set up cross validation 
xgboostGrid <- expand.grid(nrounds = c(50, 100, 300),
                           subsample = c(1), 
                           eta = c(0.1, 0.3),
                           gamma = c(0, 0.5),
                           colsample_bytree = c(0.5, 1),
                           min_child_weight = c(1),
                           max_depth = c(3, 6))

# Set up cross validation within the training set. 10 folds, 4 repeats. 
xgboostControl = trainControl(method = "repeatedcv",
                              number = 10,
                              repeats = 4,
                              search = "grid")


# Knee 
xgboost_knee <- train(EQ5D_Index_Diff ~ .,
                      data = kneeTrain,
                      method = "xgbTree",
                      trControl = xgboostControl,
                      tuneGrid = xgboostGrid,
                      verbose = TRUE,
                      objective = 'reg:linear',
                      metric = 'rmse')

print(xgboost_knee)
summary(xgboost_knee)

# Model Results 
print(xgboost_knee)
summary(xgboost_knee)

# Validation 
knee_predict = predict(xgboost_knee, kneeTest_data)
knee_R2 = R2(knee_predict, kneeTest_labels)
knee_AdjR2 = AdjR2(knee_R2, ncol(kneeTest_data), nrow(kneeTest_data))
knee_RMSE = rmse(kneeTest_labels, knee_predict)

print(c(knee_R2, knee_AdjR2, knee_RMSE))

# Hip 
xgboost_hip <- train(EQ5D_Index_Diff ~ .,
                     data = hipTrain,
                     method = "xgbTree",
                     trControl = xgboostControl,
                     tuneGrid = xgboostGrid,
                     verbose = TRUE,
                     objective = 'reg:linear',
                     metric = 'rmse')

# Model Results
print(xgboost_hip)
summary(xgboost_hip)

# Validation 
hip_predict = predict(xgboost_hip, hipTest_data)
hip_R2 = R2(hip_predict, hipTest_labels)
hip_AdjR2 = AdjR2(hip_R2, ncol(hipTest_data), nrow(hipTest_data))
hip_RMSE = rmse(hipTest_labels, hip_predict)

print(c(hip_R2, hip_AdjR2, hip_RMSE))

# Groin
xgboost_groin <- train(EQ5D_Index_Diff ~ .,
                      data = groinTrain,
                      method = "xgbTree",
                      trControl = xgboostControl,
                      tuneGrid = xgboostGrid,
                      verbose = TRUE,
                      objective = 'reg:linear',
                      metric = 'rmse')

# Model Results 
print(xgboost_groin)
summary(xgboost_groin)

# Validation 
groin_predict = predict(xgboost_groin, groinTest_data)
groin_R2 = R2(groin_predict, groinTest_labels)
groin_AdjR2 = AdjR2(groin_R2, ncol(groinTest_data), nrow(groinTest_data))
groin_RMSE = rmse(groinTest_labels, groin_predict)

print(c(groin_R2, groin_AdjR2, groin_RMSE))

############### Formatting Results ###############



