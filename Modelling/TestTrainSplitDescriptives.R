###############################################
###### Test/Train Splits & Descriptives #######
###############################################

rm(list = ls())

setwd("/Users/matthewkeys/Desktop/CRES/Data/SurgeryPrognostics/Cleaned/")

library(caret)        # Predictive Modelling  
library(feather)      # Super efficient data transfer from python to R (not meant for long term storage)

# Load data
knee_wider <- read_feather("knee_wider_wdd.feather")
hip_wider <- read_feather("hip_wider_wdd.feather")
groin_wider <- read_feather("groin_wider_wdd.feather")

# Make the year variable predictive friendly: PROMs initialisation difference. (i.e. how long data has been collected for)
knee_wider$Year_diff = knee_wider$Year - 2009 
hip_wider$Year_diff = hip_wider$Year - 2009 
groin_wider$Year_diff = groin_wider$Year - 2009 

## Make EQ5D_Index Diff Discrete 
knee_wider$EQ5D_Change_Discrete = cut(knee_wider$EQ5D_Index_Diff,
                                      breaks = c(-Inf, -0.1, 0.1, Inf),
                                      labels = c("Deterioration", "Minimal/No Change", "Improvement"))

hip_wider$EQ5D_Change_Discrete = cut(hip_wider$EQ5D_Index_Diff,
                                     breaks = c(-Inf, -0.1, 0.1, Inf),
                                     labels = c("Deterioration", "Minimal/No Change", "Improvement"))

groin_wider$EQ5D_Change_Discrete = cut(groin_wider$EQ5D_Index_Diff,
                                       breaks = c(-Inf, -0.1, 0.1, Inf),
                                       labels = c("Deterioration", "Minimal/No Change", "Improvement"))

### Test-Train Split
set.seed(37)

# Get training indexes (75% training, 25% testing)
kneeTrainIndex <- createDataPartition(knee_pwr$EQ5D_Index_Diff, p=0.75, list=FALSE)
hipTrainIndex <- createDataPartition(hip_pwr$EQ5D_Index_Diff, p=0.75, list=FALSE)
groinTrainIndex <- createDataPartition(groin_pwr$EQ5D_Index_Diff, p=0.75, list=FALSE)

# Formulate train & test sets 
kneeTrain = knee_wider[kneeTrainIndex, ] 
hipTrain = hip_wider[hipTrainIndex, ] 
groinTrain = groin_wider[groinTrainIndex, ] 

kneeTest = knee_wider[-kneeTrainIndex, ] 
hipTest = hip_wider[-hipTrainIndex, ] 
groinTest = groin_wider[-groinTrainIndex, ]

######## Descriptive Statistics ########



### Save datasets 
write_feather(kneeTrain, 'kneeTrain.feather')
write_feather(kneeTest, 'kneeTest.feather')

write_feather(hipTrain, 'hipTrain.feather')
write_feather(hipTest, 'hipTest.feather')

write_feather(groinTrain, 'groinTrain.feather')
write_feather(groinTest, 'groinTest.feather')
