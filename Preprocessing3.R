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
library(feather)      # Super efficient data transfer from python to R (not meant for long term storage)
library(tidyverse)    # General utility functions


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

####### Widest Dataset (with time constant factors) ########

# Create patient fixed effects
wider_groin_data$`Patient ID` = c(1:nrow(wider_groin_data))
wider_knee_data$`Patient ID` = c(1:nrow(wider_knee_data))
wider_hip_data$`Patient ID` = c(1:nrow(wider_hip_data))

# Compute EQ-5DL differences 
wider_groin_data$`EQ5D Index Diff` = wider_groin_data$`Post-Op Q EQ5D Index` - wider_groin_data$`Pre-Op Q EQ5D Index` 
wider_knee_data$`EQ5D Index Diff` = wider_knee_data$`Post-Op Q EQ5D Index` - wider_knee_data$`Pre-Op Q EQ5D Index` 
wider_hip_data$`EQ5D Index Diff` = wider_hip_data$`Post-Op Q EQ5D Index` - wider_hip_data$`Pre-Op Q EQ5D Index` 

### Transform columns to have appropriate numbers

# Gender = 1 is male, 0 female
wider_groin_data['Gender'] = (wider_groin_data['Gender'] == 2)*1
wider_knee_data['Gender'] = (wider_knee_data['Gender'] == 2)*1
wider_hip_data['Gender'] = (wider_hip_data['Gender'] == 2)*1

# Cormbidity = 1 means disease, 0 no disease or unknown
for (column in c('Arthritis', 'Cancer', 'Circulation', 'Depression', 
                'Diabetes', 'Heart Disease', 'High Bp', 'Kidney Disease', 'Liver Disease', 
                'Lung Disease', 'Nervous System', 'Stroke')) {
  
  wider_groin_data[column] = (wider_groin_data[column] == 1)*1 
  wider_knee_data[column] = (wider_knee_data[column] == 1)*1 
  wider_hip_data[column] = (wider_hip_data[column] == 1)*1 

}

### One-hot Encoding of categorical variables
cat_columns = c('Age Band', 'Gender', 'Arthritis', 'Cancer', 'Circulation', 'Depression', 
                'Diabetes', 'Heart Disease', 'High Bp', 'Kidney Disease', 'Liver Disease', 
                'Lung Disease', 'Nervous System', 'Stroke', 'Year', 'Pre-Op Q Symptom Period') 

# Datasets with binary dummies 
knee_wider_wd = fastDummies::dummy_cols(wider_knee_data,
                                            select_columns = cat_columns)

hip_wider_wd = fastDummies::dummy_cols(wider_hip_data,
                                            select_columns = cat_columns) 

groin_wider_wd = fastDummies::dummy_cols(wider_groin_data,
                                            select_columns = cat_columns) 

# Remove the initial categorical columns 
knee_wider_wd = knee_wider_wd[ , !(names(knee_wider_wd) %in% cat_columns)]
hip_wider_wd = hip_wider_wd[ , !(names(hip_wider_wd) %in% cat_columns)]
groin_wider_wd = groin_wider_wd[ , !(names(groin_wider_wd) %in% cat_columns)]

# Drop the columns for the first dummy (=0) 
indicators_0_groin = names(groin_wider_wd)[grepl(paste(c("_0", "40 to 49", "_2013"), collapse = "|"), names(groin_wider_wd))]
indicators_0_knee = names(knee_wider_wd)[grepl(paste(c("_0", "40 to 49", "_2013"), collapse = "|"), names(knee_wider_wd))]
indicators_0_hip = names(hip_wider_wd)[grepl(paste(c("_0", "40 to 49", "_2013"), collapse = "|"), names(hip_wider_wd))]

groin_wider_wdd = groin_wider_wd[, !(names(groin_wider_wd) %in% indicators_0_groin)]
knee_wider_wdd = knee_wider_wd[, !(names(knee_wider_wd) %in% indicators_0_knee)]
hip_wider_wdd = hip_wider_wd[, !(names(hip_wider_wd) %in% indicators_0_hip)]

# Convert obvious numerical columns to numeric 
convert_numerical_knee = c("Knee Replacement - Participation Rate", "Knee Replacement - Issue Rate",
                           "Knee Replacement - Response Rate", "Knee Replacement - Linkage Rate")

convert_numerical_hip = c("Hip Replacement - Participation Rate", "Hip Replacement - Issue Rate",
                          "Hip Replacement - Response Rate", "Hip Replacement - Linkage Rate")

convert_numerical_groin = c("Groin Hernia - Participation Rate", "Groin Hernia - Issue Rate",
                          "Groin Hernia - Response Rate", "Groin Hernia - Linkage Rate")

percent_to_numeric <- function(x) {
  s = as.numeric(sub("%", "", x))
  return(s)
}

knee_wider_wdd[, convert_numerical_knee] = sapply(knee_wider_wdd[, convert_numerical_knee], percent_to_numeric)
hip_wider_wdd[, convert_numerical_hip] = sapply(hip_wider_wdd[, convert_numerical_hip], percent_to_numeric)
groin_wider_wdd[, convert_numerical_groin] = sapply(groin_wider_wdd[, convert_numerical_groin], percent_to_numeric)

# Check NA counts - we lose a couple of thousand per dataset. 
nrow(knee_wider_wdd) - sum(complete.cases(knee_wider_wdd))
nrow(hip_wider_wdd) - sum(complete.cases(hip_wider_wdd))
nrow(groin_wider_wdd) - sum(complete.cases(groin_wider_wdd))

# Drop NAs 
knee_wider_wdd = knee_wider_wdd[complete.cases(knee_wider_wdd), ]
hip_wider_wdd = hip_wider_wdd[complete.cases(hip_wider_wdd), ]
groin_wider_wdd = groin_wider_wdd[complete.cases(groin_wider_wdd), ]

# Reformat column names 
names(knee_wider_wdd) <- gsub("-", "", names(knee_wider_wdd))
names(knee_wider_wdd) <- gsub(" ", "_", names(knee_wider_wdd))
names(knee_wider_wdd) <- gsub("__", "_", names(knee_wider_wdd))

names(hip_wider_wdd) <- gsub("-", "", names(hip_wider_wdd))
names(hip_wider_wdd) <- gsub(" ", "_", names(hip_wider_wdd))
names(hip_wider_wdd) <- gsub("__", "_", names(hip_wider_wdd))

names(groin_wider_wdd) <- gsub("-", "", names(groin_wider_wdd))
names(groin_wider_wdd) <- gsub(" ", "_", names(groin_wider_wdd))
names(groin_wider_wdd) <- gsub("__", "_", names(groin_wider_wdd))


##### Wide Dataset (without time constant factors) #####

# Create patient fixed effects
wide_groin_data$`Patient ID` = c(1:nrow(wide_groin_data))
wide_knee_data$`Patient ID` = c(1:nrow(wide_knee_data))
wide_hip_data$`Patient ID` = c(1:nrow(wide_hip_data))

# Compute EQ-5DL differences 
wide_groin_data$`EQ5D Index Diff` = wide_groin_data$`Post-Op Q EQ5D Index` - wide_groin_data$`Pre-Op Q EQ5D Index` 
wide_knee_data$`EQ5D Index Diff` = wide_knee_data$`Post-Op Q EQ5D Index` - wide_knee_data$`Pre-Op Q EQ5D Index` 
wide_hip_data$`EQ5D Index Diff` = wide_hip_data$`Post-Op Q EQ5D Index` - wide_hip_data$`Pre-Op Q EQ5D Index` 

### Transform columns to have appropriate numbers

# Gender = 1 is male, 0 female
wide_groin_data['Gender'] = (wide_groin_data['Gender'] == 2)*1
wide_knee_data['Gender'] = (wide_knee_data['Gender'] == 2)*1
wide_hip_data['Gender'] = (wide_hip_data['Gender'] == 2)*1

# Cormbidity = 1 means disease, 0 no disease or unknown

### One-hot Encoding of categorical variables
cat_columns = c('Age Band', 'Gender', 'Year', 'Pre-Op Q Symptom Period') 

# Datasets with binary dummies 
knee_wide_wd = fastDummies::dummy_cols(wide_knee_data,
                                  select_columns = cat_columns)

hip_wide_wd = fastDummies::dummy_cols(wide_hip_data,
                                 select_columns = cat_columns) 

groin_wide_wd = fastDummies::dummy_cols(wide_groin_data,
                                   select_columns = cat_columns) 

# Remove the initial categorical columns 
knee_wide_wd = knee_wide_wd[ , !(names(knee_wide_wd) %in% cat_columns)]
hip_wide_wd = hip_wide_wd[ , !(names(hip_wide_wd) %in% cat_columns)]
groin_wide_wd = groin_wide_wd[ , !(names(groin_wide_wd) %in% cat_columns)]

# Drop the columns for the first dummy (=0) 
indicators_0_groin = names(groin_wide_wd)[grepl(paste(c("_0", "40 to 49", "_2013"), collapse = "|"), names(groin_wide_wd))]
indicators_0_knee = names(knee_wide_wd)[grepl(paste(c("_0", "40 to 49", "_2013"), collapse = "|"), names(knee_wide_wd))]
indicators_0_hip = names(hip_wide_wd)[grepl(paste(c("_0", "40 to 49", "_2013"), collapse = "|"), names(hip_wide_wd))]

groin_wide_wdd = groin_wide_wd[, !(names(groin_wide_wd) %in% indicators_0_groin)]
knee_wide_wdd = knee_wide_wd[, !(names(knee_wide_wd) %in% indicators_0_knee)]
hip_wide_wdd = hip_wide_wd[, !(names(hip_wide_wd) %in% indicators_0_hip)]

# Convert obvious numerical columns to numeric 
knee_wide_wdd[, convert_numerical_knee] = sapply(knee_wide_wdd[, convert_numerical_knee], percent_to_numeric)
hip_wide_wdd[, convert_numerical_hip] = sapply(hip_wide_wdd[, convert_numerical_hip], percent_to_numeric)
groin_wide_wdd[, convert_numerical_groin] = sapply(groin_wide_wdd[, convert_numerical_groin], percent_to_numeric)

# Check NA counts
nrow(knee_wide_wdd) - sum(complete.cases(knee_wide_wdd))
nrow(hip_wide_wdd) - sum(complete.cases(hip_wide_wdd))
nrow(groin_wide_wdd) - sum(complete.cases(groin_wide_wdd))

# Drop NAs 
knee_wide_wdd = knee_wide_wdd[complete.cases(knee_wide_wdd),]
hip_wide_wdd = hip_wide_wdd[complete.cases(hip_wide_wdd),]
groin_wide_wdd = groin_wide_wdd[complete.cases(groin_wide_wdd),]

# Reformat column names 
names(knee_wide_wdd) <- gsub("-", "", names(knee_wide_wdd))
names(knee_wide_wdd) <- gsub(" ", "_", names(knee_wide_wdd))
names(knee_wide_wdd) <- gsub("__", "_", names(knee_wide_wdd))

names(hip_wide_wdd) <- gsub("-", "", names(hip_wide_wdd))
names(hip_wide_wdd) <- gsub(" ", "_", names(hip_wide_wdd))
names(hip_wide_wdd) <- gsub("__", "_", names(hip_wide_wdd))

names(groin_wide_wdd) <- gsub("-", "", names(groin_wide_wdd))
names(groin_wide_wdd) <- gsub(" ", "_", names(groin_wide_wdd))
names(groin_wide_wdd) <- gsub("__", "_", names(groin_wide_wdd))

#### Reduced Dataset (select pre-op characteristics, prediction only) ####

# Compute EQ-5DL differences 
reduced_groin_data$`EQ5D Index Diff` = reduced_groin_data$`Post-Op Q EQ5D Index` - reduced_groin_data$`Pre-Op Q EQ5D Index` 
reduced_knee_data$`EQ5D Index Diff` = reduced_knee_data$`Post-Op Q EQ5D Index` - reduced_knee_data$`Pre-Op Q EQ5D Index` 
reduced_hip_data$`EQ5D Index Diff` = reduced_hip_data$`Post-Op Q EQ5D Index` - reduced_hip_data$`Pre-Op Q EQ5D Index` 

### Transform columns to have appropriate numbers

# Gender = 1 is male, 0 female
reduced_groin_data['Gender'] = (reduced_groin_data['Gender'] == 2)*1
reduced_knee_data['Gender'] = (reduced_knee_data['Gender'] == 2)*1
reduced_hip_data['Gender'] = (reduced_hip_data['Gender'] == 2)*1

# Cormbidity = 1 means disease, 0 no disease or unknown

### One-hot Encoding of categorical variables
cat_columns = c('Age Band', 'Gender', 'Year', 'Pre-Op Q Symptom Period') 

# Datasets with binary dummies 
knee_reduced_wd = fastDummies::dummy_cols(reduced_knee_data,
                                       select_columns = cat_columns)

hip_reduced_wd = fastDummies::dummy_cols(reduced_hip_data,
                                      select_columns = cat_columns) 

groin_reduced_wd = fastDummies::dummy_cols(reduced_groin_data,
                                        select_columns = cat_columns) 

# Remove the initial categorical columns 
knee_reduced_wd = knee_reduced_wd[ , !(names(knee_reduced_wd) %in% cat_columns)]
hip_reduced_wd = hip_reduced_wd[ , !(names(hip_reduced_wd) %in% cat_columns)]
groin_reduced_wd = groin_reduced_wd[ , !(names(groin_reduced_wd) %in% cat_columns)]

# Drop the columns for the first dummy (=0) 
indicators_0_groin = names(groin_reduced_wd)[grepl(paste(c("_0", "40 to 49", "_2013"), collapse = "|"), names(groin_reduced_wd))]
indicators_0_knee = names(knee_reduced_wd)[grepl(paste(c("_0", "40 to 49", "_2013"), collapse = "|"), names(knee_reduced_wd))]
indicators_0_hip = names(hip_reduced_wd)[grepl(paste(c("_0", "40 to 49", "_2013"), collapse = "|"), names(hip_reduced_wd))]

groin_reduced_wdd = groin_reduced_wd[, !(names(groin_reduced_wd) %in% indicators_0_groin)]
knee_reduced_wdd = knee_reduced_wd[, !(names(knee_reduced_wd) %in% indicators_0_knee)]
hip_reduced_wdd = hip_reduced_wd[, !(names(hip_reduced_wd) %in% indicators_0_hip)]

# Check NA counts - we lose a couple of thousand per dataset. 
nrow(knee_reduced_wdd) - sum(complete.cases(knee_reduced_wdd))
nrow(hip_reduced_wdd) - sum(complete.cases(hip_reduced_wdd))
nrow(groin_reduced_wdd) - sum(complete.cases(groin_reduced_wdd))

# Drop NAs 
knee_reduced_wdd = knee_reduced_wdd[complete.cases(knee_reduced_wdd), ]
hip_reduced_wdd = hip_reduced_wdd[complete.cases(hip_reduced_wdd), ]
groin_reduced_wdd = groin_reduced_wdd[complete.cases(groin_reduced_wdd), ]

# Reformat column names 
names(knee_reduced_wdd) <- gsub("-", "", names(knee_reduced_wdd))
names(knee_reduced_wdd) <- gsub(" ", "_", names(knee_reduced_wdd))
names(knee_reduced_wdd) <- gsub("__", "_", names(knee_reduced_wdd))

names(hip_reduced_wdd) <- gsub("-", "", names(hip_reduced_wdd))
names(hip_reduced_wdd) <- gsub(" ", "_", names(hip_reduced_wdd))
names(hip_reduced_wdd) <- gsub("__", "_", names(hip_reduced_wdd))

names(groin_reduced_wdd) <- gsub("-", "", names(groin_reduced_wdd))
names(groin_reduced_wdd) <- gsub(" ", "_", names(groin_reduced_wdd))
names(groin_reduced_wdd) <- gsub("__", "_", names(groin_reduced_wdd))

####### Save Datasets #######

# Wider datasets (Econometric + Predictive exercises)
write_feather(knee_wider_wdd, 'knee_wider_wdd.feather')
write_feather(hip_wider_wdd, 'hip_wider_wdd.feather')
write_feather(groin_wider_wdd, 'groin_wider_wdd.feather')

# Wide datasets (Econometric + Predictive exercises)
write_feather(knee_wide_wdd, 'knee_wide_wdd.feather')
write_feather(hip_wide_wdd, 'hip_wide_wdd.feather')
write_feather(groin_wide_wdd, 'groin_wide_wdd.feather')

# Reduced datasets (Predictive exercises)
write_feather(knee_reduced_wdd, 'knee_reduced_wdd.feather')
write_feather(hip_reduced_wdd, 'hip_reduced_wdd.feather')
write_feather(groin_reduced_wdd, 'groin_reduced_wdd.feather')











