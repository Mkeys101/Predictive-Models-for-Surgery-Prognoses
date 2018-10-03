###########################################################
############### Predictive Models (Levels) ################
###########################################################

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
library(Metrics) 

# Load data
knee_wider <- read_feather("knee_wider_wdd.feather")
hip_wider <- read_feather("hip_wider_wdd.feather")
groin_wider <- read_feather("groin_wider_wdd.feather")

############  Wider Predictive Models ############

#### Final stage preprocessing 

# Make the year variable predictive friendly: PROMs initialisation difference. (i.e. how long data has been collected for)
knee_wider$Year_diff = knee_wider$Year - 2009 
hip_wider$Year_diff = hip_wider$Year - 2009 
groin_wider$Year_diff = groin_wider$Year - 2009 

# Drop Irrelevant variables for prediction (fixed effects, time indicators or alternative measures of output e.g. Post-Op Score)
knee_pwr <- knee_wider[ , !(names(knee_wider) %in% c("EQ5D_Index_Diff", "PreOp_Q_EQ5D_Index_Profile",
                                                 "Provider_Code", "Knee_Replacement_PostOp_Q_Score",
                                                 "Year", "Year_2014", "Year_2015", "Year_2016",
                                                 "Patient_ID", "PreOp_Q_Symptom_Period_1", "PreOp_Q_Symptom_Period_2",
                                                 "PreOp_Q_Symptom_Period_3", "PreOp_Q_Symptom_Period_4",
                                                 "Knee_Replacement_Participation_Rate", "Knee_Replacement_Linkage_Rate",
                                                 "Knee_Replacement_Issue_Rate", "Knee_Replacement_Response_Rate"))]

groin_pwr <- groin_wider[ , !(names(groin_wider) %in% c("EQ5D_Index_Diff", "PreOp_Q_EQ5D_Index_Profile",
                                                    "Provider_Code", "Groin_Hernia_PostOp_Q_Score",
                                                    "Year", "Year_2014", "Year_2015", "Year_2016",
                                                    "Patient_ID", "PreOp_Q_Symptom_Period_1", "PreOp_Q_Symptom_Period_2",
                                                    "PreOp_Q_Symptom_Period_3", "PreOp_Q_Symptom_Period_4",
                                                    "Groin_Hernia_Participation_Rate", "Groin_Hernia_Linkage_Rate",
                                                    "Groin_Hernia_Issue_Rate", "Groin_Hernia_Response_Rate"))]
                      
hip_pwr <- hip_wider[ , !(names(hip_wider) %in% c("EQ5D_Index_Diff", "PreOp_Q_EQ5D_Index_Profile", 
                                              "Provider_Code", "Hip_Replacement_PostOp_Q_Score",
                                              "Year", "Year_2014", "Year_2015", "Year_2016",
                                              "Patient_ID", "PreOp_Q_Symptom_Period_1", "PreOp_Q_Symptom_Period_2",
                                              "PreOp_Q_Symptom_Period_3", "PreOp_Q_Symptom_Period_4",
                                              "Hip_Replacement_Participation_Rate", "Hip_Replacement_Linkage_Rate",
                                              "Hip_Replacement_Issue_Rate", "Hip_Replacement_Response_Rate"))]

#### Feature Engineering


# PCA 

#### Test-Train Split
# Get training indexes (75% training, 25% testing)
kneeTrainIndex <- createDataPartition(knee_pwr$PostOp_Q_EQ5D_Index, p=0.75, list=FALSE)
hipTrainIndex <- createDataPartition(hip_pwr$PostOp_Q_EQ5D_Index, p=0.75, list=FALSE)
groinTrainIndex <- createDataPartition(groin_pwr$PostOp_Q_EQ5D_Index, p=0.75, list=FALSE)

# Formumlate training and test sets 
kneeTrain = knee_pwr[kneeTrainIndex, ] 
hipTrain = hip_pwr[hipTrainIndex, ] 
groinTrain = groin_pwr[groinTrainIndex, ] 

kneeTest = knee_pwr[-kneeTrainIndex, ] 
hipTest = hip_pwr[-hipTrainIndex, ] 
groinTest = groin_pwr[-groinTrainIndex, ]

## Seperate inputs and labels & convert to matrices 
# Knee
kneeTrain_data = data.matrix(subset(kneeTrain, select = -c(PostOp_Q_EQ5D_Index)))
kneeTrain_labels = data.matrix(kneeTrain["PostOp_Q_EQ5D_Index"])

kneeTest_data = data.matrix(subset(kneeTest, select = -c(PostOp_Q_EQ5D_Index)))
kneeTest_labels = data.matrix(kneeTest["PostOp_Q_EQ5D_Index"])

# Hip
hipTrain_data = data.matrix(subset(hipTrain, select = -c(PostOp_Q_EQ5D_Index)))
hipTrain_labels = data.matrix(hipTrain["PostOp_Q_EQ5D_Index"])

hipTest_data = data.matrix(subset(hipTest, select = -c(PostOp_Q_EQ5D_Index)))
hipTest_labels = data.matrix(hipTest["PostOp_Q_EQ5D_Index"])

# Groin
groinTrain_data = data.matrix(subset(groinTrain, select = -c(PostOp_Q_EQ5D_Index)))
groinTrain_labels = data.matrix(groinTrain["PostOp_Q_EQ5D_Index"])

groinTest_data = data.matrix(subset(groinTest, select = -c(PostOp_Q_EQ5D_Index)))
groinTest_labels = data.matrix(groinTest["PostOp_Q_EQ5D_Index"])

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

hip_test_R2 = R2(hip_pred, hipTest_labels)
hip_test_AdjR2 = AdjR2(hip_test_R2, ncol(hipTest_data), nrow(hipTest_data))

print(hip_test_R2)
print(hip_test_AdjR2)

groin_test_R2 = R2(groin_pred, groinTest_labels)
groin_test_AdjR2 = AdjR2(groin_test_R2, ncol(groinTest_data), nrow(groinTest_data))

print(groin_test_R2)
print(groin_test_AdjR2)

### Now with extensive hyperparameter optimisation ###

# Set up grid to search 
xgboostGrid <- expand.grid(nrounds = c(50, 100, 200, 800),
                           eta = c(0.1, 0.3, 0.5),
                           gamma = c(0, 0.5, 1)
                           subsample = c(0.5, 1), 
                           colsample_bytree = c(0.5, 1),
                           max_depth = c(3, 6, 12),
                           min_child_weight = c(1, 3),
                           lambda = c(1, 4), 
                           alpha = c(0, 1))

# Set up cross validation within the training set. 10 folds, 4 repeats. 
xgboostControl = trainControl(method = "repeatedcv",
                              number = 10,
                              repeats = 4,
                              search = "grid")

### Knee 
xgboost_knee <- train(PostOp_Q_EQ5D_Index ~ .,
                      data = kneeTrain,
                      method = "xgbTree",
                      trControl = xgboostControl,
                      tuneGrid = xgboostGrid,
                      verbose = TRUE,
                      objective = 'reg:linear',
                      metric = c('Rsquared', 'rmse'))

# Model Results 
print(xgboost_knee)
summary(xgboost_knee)

# Validation 
knee_predict = predict(xgboost_knee, kneeTest_data)
knee_R2 = R2(knee_predict, kneeTest_labels)
knee_AdjR2 = AdjR2(knee_R2, ncol(kneeTest_data), nrow(kneeTest_data))
knee_RMSE = rmse(kneeTest_labels, knee_predict)

print(c(knee_R2, knee_AdjR2, knee_RMSE))

### Hip 
xgboost_hip <- train(PostOp_Q_EQ5D_Index ~ .,
                      data = hipTrain,
                      method = "xgbTree",
                      trControl = xgboostControl,
                      tuneGrid = xgboostGrid,
                      verbose = TRUE,
                      objective = 'reg:linear',
                      metric = c('Rsquared', 'rmse'))

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
xgboost_groin <- train(PostOp_Q_EQ5D_Index ~ .,
                      data = groinTrain,
                      method = "xgbTree",
                      trControl = xgboostControl,
                      tuneGrid = xgboostGrid,
                      verbose = TRUE,
                      eval = 'reg:linear',
                      metric = c('Rsquared', 'rmse'))
                      
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


