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

knee_wide <- read_feather("knee_wide_wdd.feather")
hip_wide <- read_feather("hip_wide_wdd.feather")
groin_wide <- read_feather("groin_wide_wdd.feather")

##################################################
############## Econometric Exercises #############
##################################################
# Note: we have dropped the analysis with the widest dataset. Focusing on wide dataset for these exercises. 
# Makes no sense to include the time fixed comorbidities at the expense of observations 


####### Standard Regressions - No fixed Effects ########

### Groin Hernia Repair 
groin_regression = lm(PostOp_Q_EQ5D_Index ~ PreOp_Q_Activity + PreOp_Q_Anxiety + 
                      PreOp_Q_Discomfort +  PreOp_Q_Disability + PreOp_Q_EQ5D_Index + PreOp_Q_Mobility + PreOp_Q_SelfCare +           
                      PreOp_Q_Assisted + PreOp_Q_Assisted_By + Age_Band_30_to_39 + Age_Band_50_to_59 +
                      Age_Band_60_to_69 + Age_Band_70_to_79 + Age_Band_80_to_89 + 
                      Groin_Hernia_Participation_Rate + Groin_Hernia_Linkage_Rate + Groin_Hernia_Issue_Rate + 
                      Groin_Hernia_Response_Rate + Gender_1 + Year_2014 + Year_2015 + Year_2016 +
                      PreOp_Q_Symptom_Period_1 + PreOp_Q_Symptom_Period_2, 
                      data = groin_wide)

summary(groin_regression)

### Knee Replacement Surgery  
knee_regression = lm(PostOp_Q_EQ5D_Index ~ Knee_Replacement_PreOp_Q_Score + PreOp_Q_Activity + PreOp_Q_Anxiety + 
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
                     data = knee_wide)

summary(knee_regression)

### Hip Replacement Surgery 
hip_regression = lm(PostOp_Q_EQ5D_Index ~ Hip_Replacement_PreOp_Q_Score + PreOp_Q_Activity + PreOp_Q_Anxiety + 
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
                    data = hip_wider)

summary(hip_regression)

####### Regressions with patient fixed effects only ########

### Groin Hernia Repair 
groin_fe_pa_regression = plm(PostOp_Q_EQ5D_Index ~ PreOp_Q_Activity + PreOp_Q_Anxiety + 
                             PreOp_Q_Discomfort +  PreOp_Q_Disability + PreOp_Q_EQ5D_Index + PreOp_Q_Mobility + PreOp_Q_SelfCare +           
                             PreOp_Q_Assisted + PreOp_Q_Assisted_By + Age_Band_30_to_39 + Age_Band_50_to_59 +
                             Age_Band_60_to_69 + Age_Band_70_to_79 + Age_Band_80_to_89 + 
                             Groin_Hernia_Participation_Rate + Groin_Hernia_Linkage_Rate + Groin_Hernia_Issue_Rate + 
                             Groin_Hernia_Response_Rate + Gender_1 + Year_2014 + Year_2015 + Year_2016 +
                             PreOp_Q_Symptom_Period_1 + PreOp_Q_Symptom_Period_2, 
                             data = groin_wide,
                             index=c('Patient_ID'), 
                             model="within")

summary(groin_fe_pa_regression)

### Knee Replacement Surgery  
knee_fe_pa_regression = plm(PostOp_Q_EQ5D_Index ~ Knee_Replacement_PreOp_Q_Score + PreOp_Q_Activity + PreOp_Q_Anxiety + 
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
                              index=c('Patient_ID'), 
                              model="within")

summary(knee_fe_pa_regression)

### Hip Replacement Surgery 
hip_fe_pa_regression = plm(PostOp_Q_EQ5D_Index ~ Hip_Replacement_PreOp_Q_Score + PreOp_Q_Activity + PreOp_Q_Anxiety + 
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
                           index=c('Patient_ID'), 
                           model="within")

summary(hip_fe_pa_regression)

####### Regressions with provider fixed effects only ########

### Groin Hernia Repair 
groin_fe_pr_regression = plm(PostOp_Q_EQ5D_Index ~ PreOp_Q_Activity + PreOp_Q_Anxiety + 
                             PreOp_Q_Discomfort +  PreOp_Q_Disability + PreOp_Q_EQ5D_Index + PreOp_Q_Mobility + PreOp_Q_SelfCare +           
                             PreOp_Q_Assisted + PreOp_Q_Assisted_By + Age_Band_30_to_39 + Age_Band_50_to_59 +
                             Age_Band_60_to_69 + Age_Band_70_to_79 + Age_Band_80_to_89 + 
                             Groin_Hernia_Participation_Rate + Groin_Hernia_Linkage_Rate + Groin_Hernia_Issue_Rate + 
                             Groin_Hernia_Response_Rate + Gender_1 + Year_2014 + Year_2015 + Year_2016 +
                             PreOp_Q_Symptom_Period_1 + PreOp_Q_Symptom_Period_2, 
                             data = groin_wide,
                             index=c('Provider_Code'), 
                             model="within")

summary(groin_fe_pr_regression)

### Knee Replacement Surgery  
knee_fe_pr_regression = plm(PostOp_Q_EQ5D_Index ~ Knee_Replacement_PreOp_Q_Score + PreOp_Q_Activity + PreOp_Q_Anxiety + 
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
                            index=c('Provider_Code'), 
                            model="within")

summary(knee_fe_pr_regression)

### Hip Replacement Surgery 
hip_fe_pr_regression = plm(PostOp_Q_EQ5D_Index ~ Hip_Replacement_PreOp_Q_Score + PreOp_Q_Activity + PreOp_Q_Anxiety + 
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
                           index=c('Provider_Code'), 
                           model="within")

summary(hip_fe_pr_regression)


####### Regressions with both patient and provider fixed effects ########

### Groin Hernia Repair 
groin_fe_prpa_regression = plm(PostOp_Q_EQ5D_Index ~ PreOp_Q_Activity + PreOp_Q_Anxiety + 
                               PreOp_Q_Discomfort +  PreOp_Q_Disability + PreOp_Q_EQ5D_Index + PreOp_Q_Mobility + PreOp_Q_SelfCare +           
                               PreOp_Q_Assisted + PreOp_Q_Assisted_By + Age_Band_30_to_39 + Age_Band_50_to_59 +
                               Age_Band_60_to_69 + Age_Band_70_to_79 + Age_Band_80_to_89 + 
                               Groin_Hernia_Participation_Rate + Groin_Hernia_Linkage_Rate + Groin_Hernia_Issue_Rate + 
                               Groin_Hernia_Response_Rate + Gender_1 + Year_2014 + Year_2015 + Year_2016 +
                               PreOp_Q_Symptom_Period_1 + PreOp_Q_Symptom_Period_2, 
                               data = groin_wide,
                               index=c('Provider_Code','Patient_ID'), 
                               model="within")

summary(groin_fe_prpa_regression)

### Knee Replacement Surgery  
knee_fe_prpa_regression = plm(PostOp_Q_EQ5D_Index ~ Knee_Replacement_PreOp_Q_Score + PreOp_Q_Activity + PreOp_Q_Anxiety + 
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

summary(knee_fe_prpa_regression)

### Hip Replacement Surgery 
hip_fe_prpa_regression = plm(PostOp_Q_EQ5D_Index ~ Hip_Replacement_PreOp_Q_Score + PreOp_Q_Activity + PreOp_Q_Anxiety + 
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

summary(hip_fe_prpa_regression)

###### Regressions with unique individual input coefficients (Random effects model?) #######

### Groin Hernia Repair 
groin_fe_prpa_regression = plm(PostOp_Q_EQ5D_Index ~ PreOp_Q_Activity + PreOp_Q_Anxiety + 
                                 PreOp_Q_Discomfort +  PreOp_Q_Disability + PreOp_Q_EQ5D_Index + PreOp_Q_Mobility + PreOp_Q_SelfCare +           
                                 PreOp_Q_Assisted + PreOp_Q_Assisted_By + Age_Band_30_to_39 + Age_Band_50_to_59 +
                                 Age_Band_60_to_69 + Age_Band_70_to_79 + Age_Band_80_to_89 + 
                                 Groin_Hernia_Participation_Rate + Groin_Hernia_Linkage_Rate + Groin_Hernia_Issue_Rate + 
                                 Groin_Hernia_Response_Rate + Gender_1 + Year_2014 + Year_2015 + Year_2016 +
                                 PreOp_Q_Symptom_Period_1 + PreOp_Q_Symptom_Period_2, 
                               data = groin_wide,
                               index=c('Provider_Code','Patient_ID'), 
                               model="random")

summary(groin_fe_prpa_regression)

### Knee Replacement Surgery  
knee_fe_prpa_regression = plm(PostOp_Q_EQ5D_Index ~ Knee_Replacement_PreOp_Q_Score + PreOp_Q_Activity + PreOp_Q_Anxiety + 
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
                              model="random")

summary(knee_fe_prpa_regression)

### Hip Replacement Surgery 
hip_fe_prpa_regression = plm(PostOp_Q_EQ5D_Index ~ Hip_Replacement_PreOp_Q_Score + PreOp_Q_Activity + PreOp_Q_Anxiety + 
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
                             model="random")

summary(hip_fe_prpa_regression)

############ Presentation of Results ##############

### Tables
stargazer(knee_regression, knee_fe_pr_regression, knee_fe_pa_regression, knee_fe_prpa_regression,
          title="Knee Replacement PostOp EQ5D - Variance Explaining Exercises", align=TRUE,
          keep.stat = c('n', 'adj.rsq', 'rsq', 'ser'))

stargazer(hip_regression, hip_fe_pr_regression, hip_fe_pa_regression, hip_fe_prpa_regression,
          title="Hip Replacement PostOp EQ5D Difference - Variance Explaining Exercises", align=TRUE,
          keep.stat = c('n', 'adj.rsq', 'rsq', 'ser'))

stargazer(groin_regression, groin_fe_pr_regression, groin_fe_pa_regression, groin_fe_prpa_regression,
          title="Groin Hernia PostOp EQ5D Difference - Variance Explaining Exercises", align=TRUE,
          keep.stat = c('n', 'adj.rsq', 'rsq', 'ser'))

### Plots 
Models = c("Regression", "Regression w/ Provider FE", 
           "Regression w/ Patient FE", 
           "Regression w/ Patient & Provider FE")

Knee_AdjR2s = c(results_knee_regression$adj.r.squared, results_knee_fe_pr_regression$adj.r.squared,
                results_knee_fe_pa_regression$adj.r.squared2, results_knee_fe_prpa_regression$adj.r.squared)

Hip_AdjR2s = c(results_hip_regression$adj.r.squared, results_hip_fe_pr_regression$adj.r.squared,
               results_hip_fe_pa_regression$adj.r.squared2, results_hip_fe_prpa_regression$adj.r.squared)

Groin_AdjR2s = c(results_groin_regression$adj.r.squared, results_groin_fe_pr_regression$adj.r.squared,
                 results_groin_fe_pa_regression$adj.r.squared2, results_groin_fe_prpa_regression$adj.r.squared)

AdjR2_matrix = cbind(Models, Knee_AdjR2s, Hip_AdjR2s, Groin_AdjR2s) 


