###################################################################
############### Predictive Models (Classification) ################
#################################################################

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
kneeTrain <- kneeTrain[ , !(names(kneeTrain) %in% c("EQ5D_Index_Diff", "PreOp_Q_EQ5D_Index_Profile", "PostOp_Q_EQ5D_Index",  
                                                    "Provider_Code", "Knee_Replacement_PostOp_Q_Score",
                                                    "Year", "Year_2014", "Year_2015", "Year_2016",
                                                    "Patient_ID", "PreOp_Q_Symptom_Period_1", "PreOp_Q_Symptom_Period_2",
                                                    "PreOp_Q_Symptom_Period_3", "PreOp_Q_Symptom_Period_4",
                                                    "Knee_Replacement_Participation_Rate", "Knee_Replacement_Linkage_Rate",
                                                    "Knee_Replacement_Issue_Rate", "Knee_Replacement_Response_Rate"))]

kneeTest <- kneeTest[ , !(names(kneeTest) %in% c("EQ5D_Index_Diff", "PreOp_Q_EQ5D_Index_Profile", "PostOp_Q_EQ5D_Index",  
                                                 "Provider_Code", "Knee_Replacement_PostOp_Q_Score",
                                                 "Year", "Year_2014", "Year_2015", "Year_2016",
                                                 "Patient_ID", "PreOp_Q_Symptom_Period_1", "PreOp_Q_Symptom_Period_2",
                                                 "PreOp_Q_Symptom_Period_3", "PreOp_Q_Symptom_Period_4",
                                                 "Knee_Replacement_Participation_Rate", "Knee_Replacement_Linkage_Rate",
                                                 "Knee_Replacement_Issue_Rate", "Knee_Replacement_Response_Rate"))]


# groin data
groinTrain <- groinTrain[ , !(names(groinTrain) %in% c("EQ5D_Index_Diff", "PreOp_Q_EQ5D_Index_Profile", "PostOp_Q_EQ5D_Index", 
                                                       "Provider_Code", "Groin_Hernia_PostOp_Q_Score",
                                                       "Year", "Year_2014", "Year_2015", "Year_2016",
                                                       "Patient_ID", "PreOp_Q_Symptom_Period_1", "PreOp_Q_Symptom_Period_2",
                                                       "PreOp_Q_Symptom_Period_3", "PreOp_Q_Symptom_Period_4",
                                                       "Groin_Hernia_Participation_Rate", "Groin_Hernia_Linkage_Rate",
                                                       "Groin_Hernia_Issue_Rate", "Groin_Hernia_Response_Rate"))]

groinTest <- groinTest[ , !(names(groinTest) %in% c("EQ5D_Index_Diff", "PreOp_Q_EQ5D_Index_Profile", "PostOp_Q_EQ5D_Index", 
                                                    "Provider_Code", "Groin_Hernia_PostOp_Q_Score",
                                                    "Year", "Year_2014", "Year_2015", "Year_2016",
                                                    "Patient_ID", "PreOp_Q_Symptom_Period_1", "PreOp_Q_Symptom_Period_2",
                                                    "PreOp_Q_Symptom_Period_3", "PreOp_Q_Symptom_Period_4",
                                                    "Groin_Hernia_Participation_Rate", "Groin_Hernia_Linkage_Rate",
                                                    "Groin_Hernia_Issue_Rate", "Groin_Hernia_Response_Rate"))]

# hip data                       
hipTrain <- hipTrain[ , !(names(hipTrain) %in% c("EQ5D_Index_Diff", "PreOp_Q_EQ5D_Index_Profile", "PostOp_Q_EQ5D_Index", 
                                                 "Provider_Code", "Hip_Replacement_PostOp_Q_Score",
                                                 "Year", "Year_2014", "Year_2015", "Year_2016",
                                                 "Patient_ID", "PreOp_Q_Symptom_Period_1", "PreOp_Q_Symptom_Period_2",
                                                 "PreOp_Q_Symptom_Period_3", "PreOp_Q_Symptom_Period_4",
                                                 "Hip_Replacement_Participation_Rate", "Hip_Replacement_Linkage_Rate",
                                                 "Hip_Replacement_Issue_Rate", "Hip_Replacement_Response_Rate"))]

hipTest <- hipTest[ , !(names(hipTest) %in% c("EQ5D_Index_Diff", "PreOp_Q_EQ5D_Index_Profile", "PostOp_Q_EQ5D_Index", 
                                              "Provider_Code", "Hip_Replacement_PostOp_Q_Score",
                                              "Year", "Year_2014", "Year_2015", "Year_2016",
                                              "Patient_ID", "PreOp_Q_Symptom_Period_1", "PreOp_Q_Symptom_Period_2",
                                              "PreOp_Q_Symptom_Period_3", "PreOp_Q_Symptom_Period_4",
                                              "Hip_Replacement_Participation_Rate", "Hip_Replacement_Linkage_Rate",
                                              "Hip_Replacement_Issue_Rate", "Hip_Replacement_Response_Rate"))]


## Seperate inputs and labels & convert to matrices 
# Knee
kneeTrain_data = data.matrix(subset(kneeTrain, select = -c(EQ5D_Change_Discrete)))
kneeTrain_labels = data.matrix(kneeTrain["EQ5D_Change_Discrete"])

kneeTest_data = data.matrix(subset(kneeTest, select = -c(EQ5D_Change_Discrete)))
kneeTest_labels = data.matrix(kneeTest["EQ5D_Change_Discrete"])

# Hip
hipTrain_data = data.matrix(subset(hipTrain, select = -c(EQ5D_Change_Discrete)))
hipTrain_labels = data.matrix(hipTrain["EQ5D_Change_Discrete"])

hipTest_data = data.matrix(subset(hipTest, select = -c(EQ5D_Change_Discrete)))
hipTest_labels = data.matrix(hipTest["EQ5D_Change_Discrete"])

# Groin
groinTrain_data = data.matrix(subset(groinTrain, select = -c(EQ5D_Change_Discrete)))
groinTrain_labels = data.matrix(groinTrain["EQ5D_Change_Discrete"])

groinTest_data = data.matrix(subset(groinTest, select = -c(EQ5D_Change_Discrete)))
groinTest_labels = data.matrix(groinTest["EQ5D_Change_Discrete"])

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

# Convert training & test data to XGboost optimised sparse matrices
knee_dtrain <- xgb.DMatrix(data = kneeTrain_data , label = kneeTrain_labels)
hip_dtrain <- xgb.DMatrix(data = hipTrain_data , label = hipTrain_labels)
groin_dtrain <- xgb.DMatrix(data = groinTrain_data , label = groinTrain_labels)

knee_dtest <- xgb.DMatrix(data = kneeTest_data , label = kneeTest_labels)
hip_dtest <- xgb.DMatrix(data = hipTest_data , label = hipTest_labels)
groin_dtest <- xgb.DMatrix(data = groinTest_data , label = groinTest_labels)

##### First runs (Postop level outcome) #####
xgb_knee <- xgboost(data = knee_dtrain, nrounds=50)
xgb_hip <- xgboost(data = hip_dtrain, nrounds=100)
xgb_groin <- xgboost(data = groin_dtrain, nrounds=100)

knee_pred <- predict(xgb_knee, knee_dtest)
hip_pred <- predict(xgb_hip, hip_dtest)
groin_pred <- predict(xgb_groin, groin_dtest)

# Knee results 
knee_test_R2 = R2(knee_pred, kneeTest_labels)
knee_test_AdjR2 = AdjR2(knee_test_R2, ncol(kneeTest_data), nrow(kneeTest_data))

print(knee_test_R2)
print(knee_test_AdjR2)

confusionMatrix(factor(knee_pred), )


# Hip Results 
hip_test_R2 = R2(hip_pred, hipTest_labels)
hip_test_AdjR2 = AdjR2(hip_test_R2, ncol(hipTest_data), nrow(hipTest_data))

print(hip_test_R2)
print(hip_test_AdjR2)

# Groin Results 
groin_test_R2 = R2(groin_pred, groinTest_labels)
groin_test_AdjR2 = AdjR2(groin_test_R2, ncol(groinTest_data), nrow(groinTest_data))

print(groin_test_R2)
print(groin_test_AdjR2)

### Now with extensive hyperparameter optimisation ###

### Set up cross validation 
xgboostGrid <- expand.grid(nrounds = c(50, 70, 100, 150, 300, 600, 1000),
                           eta = c(0.05, 0.1, 0.3),
                           gamma = 1,
                           subsample = c(0.5, 1), 
                           colsample_bytree = c(0.5, 1),
                           max_depth = c(3, 6, 12),
                           min_child_weight = c(1, 4))

xgboostControl = trainControl(method = "repeatedcv",
                              number = 10,
                              repeats = 2,
                              search = "grid")

### Knee 
xgboost_knee <- train(PostOp_Q_EQ5D_Index ~ .,
                      data = kneeTrain,
                      method = "xgbTree",
                      trControl = xgboostControl,
                      tuneGrid = xgboostGrid,
                      verbose = TRUE,
                      metric = 'Rsquared')

# Model Results 
print(xgboost_knee)
summary(xgboost_knee)

# Validation 
knee_predict = predict(xgboost_knee, kneeTest_data)
knee_R2 = R2(knee_predict, kneeTest_labels)
knee_AdjR2 = AdjR2(knee_R2, ncol(kneeTest_data), nrow(kneeTest_data))

cat('The adjusted R^2 error of the knee model on the test data is ', knee_AdjR2,'\n')
cat('The optimal hyperparameters found for the knee model are', knee_optimal_hp,'\n')

### Hip 
xgboost_hip <- train(PostOp_Q_EQ5D_Index ~ .,
                     data = hipTrain,
                     method = "xgbTree",
                     trControl = xgboostControl,
                     tuneGrid = xgboostGrid,
                     verbose = TRUE,
                     metric = 'Rsquared')

print(xgboost_hip)
summary(xgboost_hip)

# Model Results 
print(xgboost_hip)
summary(xgboost_hip)

# Validation 
hip_predict = predict(xgboost_hip, hipTest_data)
hip_R2 = R2(hip_predict, hipTest_labels)
hip_AdjR2 = AdjR2(hip_R2, ncol(hipTest_data), nrow(hipTest_data))

cat('The adjusted R^2 error of the hip model on the test data is ', hip_AdjR2,'\n')
cat('The optimal hyperparameters found for the hip model are', hip_optimal_hp,'\n')

# Groin
xgboost_knee <- train(PostOp_Q_EQ5D_Index ~ .,
                      data = groinTrain,
                      method = "xgbTree",
                      trControl = xgboostControl,
                      tuneGrid = xgboostGrid,
                      verbose = TRUE,
                      metric = 'Rsquared')

print(xgboost_groin)
summary(xgboost_groin)

# Model Results 
print(xgboost_groin)
summary(xgboost_groin)

# Validation 
groin_predict = predict(xgboost_hip, hipTest_data)
groin_R2 = R2(hip_predict, hipTest_labels)
groin_AdjR2 = AdjR2(hip_R2, ncol(hipTest_data), nrow(hipTest_data))

cat('The adjusted R^2 error of the groin model on the test data is ', groin_AdjR2,'\n')
cat('The optimal hyperparameters found for the groin model are ', groin_optimal_hp,'\n')

############### Formatting Results ###############

