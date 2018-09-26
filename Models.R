#############################################################
######## Predictive Models for Surgery Prognostics ##########
#############################################################

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
library(feather)      # Super efficient data transfer from python to R 

### Load Data 
reduced_groin_data <- read_feather("reduced_groin_data.feather")
reduced_hip_data <- read_feather("reduced_hip_data.feather")
reduced_knee_data <- read_feather("reduced_knee_data.feather")

wide_groin_data <- read_feather("large_groin_data.feather")
wide_hip_data <- read_feather("large_hip_data.feather")
wide_knee_data <- read_feather("large_knee_data.feather")

wider_groin_data <- read_feather("larger_groin_data.feather")
wider_hip_data <- read_feather("larger_hip_data.feather")
wider_knee_data <- read_feather("larger_knee_data.feather")

#####################################
######## Final Preprocessing ########
#####################################

####### Groin Data #######

# Create patient fixed effects
wider_groin_data$`Patient ID` = c(1:nrow(wider_groin_data))

# Compute EQ-5DL differences 
wider_groin_data$`EQ5D Index Diff` = wider_groin_data$`Post-Op Q EQ5D Index` - wider_groin_data$`Pre-Op Q EQ5D Index` 

### Transform columns to have appropriate numbers

# Gender = 1 is male, 0 female
wider_groin_data['Gender'] = (wider_groin_data['Gender'] == 2)*1

# Cormbidity = 1 means disease, 0 no disease or unknown
for (column in c('Arthritis', 'Cancer', 'Circulation', 'Depression', 
                'Diabetes', 'Heart Disease', 'High Bp', 'Kidney Disease', 'Liver Disease', 
                'Lung Disease', 'Nervous System', 'Stroke')) {
  
  wider_groin_data[column] = (wider_groin_data[column] == 1)*1 

}

### One-hot Encoding  
hip_cat_columns = c('Age Band', 'Gender', 'Arthiritis', 'Cancer', 'Circulation', 'Depression', 
                    'Diabetes', 'Heart Disease', 'High Bp', 'Kidney Disease', 'Liver Disease', 
                    'Lung Disease', 'Nervous System', 'Stroke') 

knee_cat_columns = c('Age Band', 'Gender', 'Arthiritis', 'Cancer', 'Circulation', 'Depression', 
                      'Diabetes', 'Heart Disease', 'High Bp', 'Kidney Disease', 'Liver Disease', 
                      'Lung Disease', 'Nervous System', 'Stroke') 

groin_cat_columns = c('Age Band', 'Gender', 'Arthiritis', 'Cancer', 'Circulation', 'Depression', 
                      'Diabetes', 'Heart Disease', 'High Bp', 'Kidney Disease', 'Liver Disease', 
                      'Lung Disease', 'Nervous System', 'Stroke') 


knee_cat_dummies = fastDummies::dummy_cols(wider_knee_data,
                                            select_columns = knee_cat_columns)

hip_cat_dummies = fastDummies::dummy_cols(wider_hip_data,
                                            select_columns = hip_cat_columns) 

groin_cat_dummies = fastDummies::dummy_cols(wider_groin_data,
                                            select_columns = groin_cat_columns) 

# Remove the initial categorical columns 
knee_cat_dummies = knee_cat_dummies[ , !(names(knee_cat_dummies) %in% knee_cat_columns)]
hip_cat_dummies = hip_cat_dummies[ , !(names(hip_cat_dummies) %in% hip_cat_columns)]
groin_cat_dummies = groin_cat_dummies[ , !(names(groin_cat_dummies) %in% groin_cat_columns)]

# Drop the columns for the first dummy (=0) 
indicators_0 = names(cat_dummies)[grepl(paste(c("_0", "Band"), collapse = "|"), names(cat_dummies))]
  
cat_dummies = cat_dummies[, !(names(cat_dummies) %in% indicators_0)]

# Reformat column names 
names(cat_dummies) <- gsub("-", "", names(cat_dummies))
names(cat_dummies) <- gsub(" ", "_", names(cat_dummies))
names(cat_dummies) <- gsub("__", "_", names(cat_dummies))

##################################################
############## Econometric Exercises #############
##################################################

#### Groin Hernia Repair 
groin_fe_regression = plm(EQ5D_Index_Diff ~ Arthritis + PreOp_Q_Activity + PreOp_Q_Anxiety + 
                            PreOp_Q_Discomfort +  PreOp_Q_Disability + PreOp_Q_EQ5D_Index + PreOp_Q_Mobility + PreOp_Q_SelfCare +           
                            PreOp_Q_Symptom_Period + PreOp_Q_Assisted + PreOp_Q_Assisted_By +
                            Groin_Hernia_Participation_Rate + Groin_Hernia_Linkage_Rate + 
                            Groin_Hernia_Issue_Rate + Groin_Hernia_Response_Rate +
                            Gender_1 + Cancer_1 + Circulation_1 + Depression_1 + Diabetes_1 + Heart_Disease_1 +   
                            High_Bp_1 + Kidney_Disease_1 + Liver_Disease_1 + Lung_Disease_1 +                  
                            Nervous_System_1 + Stroke_1,   
                          data = cat_dummies,
                          index=c('Provider_Code',"Patient_ID"), 
                          model="within")

summary(groin_fe_regression)

#### Knee Replacement Surgery  
knee_fe_regression = plm(Post-Op Q EQ5D Index ~ .,
                         data = larger_knee_data,
                         index=c("",""), 
                         model="within")

summary(knee_fe_regression)

### Hip Replacement Surgery 

hip_fe_regression = plm(Post-Op Q EQ5D Index ~ .,
                        data = larger_hip_data,
                        index=c("",""), 
                        model="within")

summary(hip_fe_regression)

### 

# Show individual fixed effect coefficeints 
fixef(groin_fe_regression)
fixef(knee_fe_regression)
fixef(hip_fe_regression)

# F-test for individual fixed effects. 
pFtest(groin_fe_regression, ols)
pFtest(knee_fe_regression, ols)
pFtest(hip_fe_regression, ols)


##################################################
############## Predictive Exercises ##############
##################################################

############## Reduced Models ###############

### One-hot encoding 
feature_matrix <- sparse.model.matrix(response ~ .-1, data = campaign)


### Feature Engineering

# Transformations

# PCA 






# Create test-training split 



### XGBoost ###
# Set up grid search 



# Optimise 
xgb <- xgboost(data = data.matrix(X[,-1]), 
               label = y, 
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

# Predict 
y_pred <- predict(xgb, data.matrix(X_test[,-1]))





### RForest ###
# Set up grid search 



# Optimise 


# Results 




############### Full Models #################

# Create test-traning split 


# 



############## Fuller Models ################












