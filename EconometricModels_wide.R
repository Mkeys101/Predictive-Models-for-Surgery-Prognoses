#####################################################
############### Econometric Models ##################
#####################################################

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
library(stargazer)

### stargazer 

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

##################################################
############## Econometric Exercises #############
##################################################

###### Wide dataset (less features, more observations) #######
# Note: we have dropped the analysis with the widest dataset. Makes no sense to include the time fixed comorbidities. 

### Groin Hernia Repair 
groin_wide_fe_regression = plm(EQ5D_Index_Diff ~ PreOp_Q_Activity + PreOp_Q_Anxiety + 
                               PreOp_Q_Discomfort +  PreOp_Q_Disability + PreOp_Q_EQ5D_Index + PreOp_Q_Mobility + PreOp_Q_SelfCare +           
                               PreOp_Q_Assisted + PreOp_Q_Assisted_By + Age_Band_30_to_39 + Age_Band_50_to_59 +
                               Age_Band_60_to_69 + Age_Band_70_to_79 + Age_Band_80_to_89 + 
                               Groin_Hernia_Participation_Rate + Groin_Hernia_Linkage_Rate + Groin_Hernia_Issue_Rate + 
                               Groin_Hernia_Response_Rate + Gender_1 + Year_2014 + Year_2015 + Year_2016 +
                               PreOp_Q_Symptom_Period_1 + PreOp_Q_Symptom_Period_2, 
                               data = groin_wide,
                               index=c('Provider_Code','Patient_ID'), 
                               model="within")

summary(groin_wide_fe_regression)

### Knee Replacement Surgery  
knee_wide_fe_regression = plm(EQ5D_Index_Diff ~ Knee_Replacement_PreOp_Q_Score + PreOp_Q_Activity + PreOp_Q_Anxiety + 
                              PreOp_Q_Discomfort + PreOp_Q_EQ5D_Index + PreOp_Q_Mobility + PreOp_Q_SelfCare +           
                              Knee_Replacement_PreOp_Q_Confidence + Knee_Replacement_PreOp_Q_Kneeling +
                              Knee_Replacement_PreOp_Q_Limping + Knee_Replacement_PreOp_Q_Night_Pain + Knee_Replacement_PreOp_Q_Pain +
                              Knee_Replacement_PreOp_Q_Shopping + Knee_Replacement_PreOp_Q_Stairs + Knee_Replacement_PreOp_Q_Standing +
                              Knee_Replacement_PreOp_Q_Transport + Knee_Replacement_PreOp_Q_Walking + Knee_Replacement_PreOp_Q_Washing +
                              Knee_Replacement_PreOp_Q_Work + Age_Band_50_to_59 + Age_Band_60_to_69 + Age_Band_70_to_79 + Age_Band_80_to_89 +
                              Age_Band_90_to_120 + PreOp_Q_Symptom_Period_1 + PreOp_Q_Symptom_Period_2 + PreOp_Q_Symptom_Period_3 + 
                              PreOp_Q_Symptom_Period_4 + Year_2014 + Year_2015 + Year_2016 + Knee_Replacement_Participation_Rate + 
                              Knee_Replacement_Linkage_Rate + Knee_Replacement_Issue_Rate + Knee_Replacement_Response_Rate +
                              Gender_1,
                              data = knee_wide,
                              index=c('Provider_Code','Patient_ID'), 
                              model="within")

summary(knee_wide_fe_regression)

### Hip Replacement Surgery 
hip_wide_fe_regression = plm(EQ5D_Index_Diff ~ Hip_Replacement_PreOp_Q_Score + PreOp_Q_Activity + PreOp_Q_Anxiety + 
                             PreOp_Q_Discomfort + PreOp_Q_EQ5D_Index + PreOp_Q_Mobility + PreOp_Q_SelfCare +           
                             Hip_Replacement_PreOp_Q_Dressing + Hip_Replacement_PreOp_Q_Sudden_Pain +
                             Hip_Replacement_PreOp_Q_Limping + Hip_Replacement_PreOp_Q_Night_Pain + Hip_Replacement_PreOp_Q_Pain +
                             Hip_Replacement_PreOp_Q_Shopping + Hip_Replacement_PreOp_Q_Stairs + Hip_Replacement_PreOp_Q_Standing +
                             Hip_Replacement_PreOp_Q_Transport + Hip_Replacement_PreOp_Q_Walking + Hip_Replacement_PreOp_Q_Washing +
                             Hip_Replacement_PreOp_Q_Work + Age_Band_20_to_29 + Age_Band_30_to_39 + Age_Band_50_to_59 + 
                             Age_Band_60_to_69 + Age_Band_70_to_79 + Age_Band_80_to_89 +
                             Age_Band_90_to_120 + PreOp_Q_Symptom_Period_1 + PreOp_Q_Symptom_Period_2 + PreOp_Q_Symptom_Period_3 + 
                             PreOp_Q_Symptom_Period_4 + Year_2014 + Year_2015 + Year_2016 + Hip_Replacement_Participation_Rate + 
                             Hip_Replacement_Linkage_Rate + Hip_Replacement_Issue_Rate + Hip_Replacement_Response_Rate +
                             Gender_1,
                             data = hip_wider,
                             index=c('Provider_Code','Patient_ID'), 
                             model="within")

summary(hip_wide_fe_regression)
