
#running the neuralnet with parameter tunning in caret
library(tidyverse)
library(nnet)
library(caret)
library(doSNOW)


## ============================================================================================##
# train the neural net. With Caret
## ============================================================================================##
##load train data
BH_train <- read_rds(here::here("create_dataset", "BH_train.rds"))
BH_train <- BH_train %>% mutate(Higher = ifelse(Higher == 1, "Higher", "Lower")) %>% 
  mutate(Higher = factor(Higher))
BH_train <- BH_train %>% sample_frac(.0051)

fitControl <- trainControl(method = "repeatedcv",
                           number = 10,
                           repeats = 2,
                           search = "grid")

tune_grid <-  expand.grid(eta = c(0.05, 0.075, 0.1),
                          nrounds = c(50, 75, 100),
                          max_depth = 6:8,
                          min_child_weight = c(2.0, 2.25, 2.5),
                          colsample_bytree = c(0.3, 0.4, 0.5),
                          gamma = 0,
                          subsample = 1)

# tune_grid <- expand.grid(subsample = c(0.5, 0.6), 
#             colsample_bytree = c(0.5, 0.6),
#             max_depth = c(3, 4),
#             min_child = seq(1), 
#             eta = c(0.1))


cl <- makeCluster(4, type = "SOCK")
registerDoSNOW(cl)

xgboostFit <- train(Higher ~ ., 
                 data = BH_train,
                 method = "xgbTree",
                 tuneGrid = tune_grid,
                 trControl = fitControl)


stopCluster(cl)

#examine caret's processing results. 

# Make predictions on the test set using 
# found optimal hyperparameter values.
BH_test <- read_rds(here::here("create_dataset", "BH_test.rds")) %>% 
  mutate(Higher = ifelse(Higher == 1, "Higher", "Lower")) %>% 
  mutate(Higher = factor(Higher))


preds2 <- predict(xgboostFit, BH_test[2:60])


# Use caret's confusionMatrix() function to estimate the 
# effectiveness of this model on unseen, new data.
confusionMatrix(preds2, BH_test$Higher)


write_rds(xgboostFit, here::here("xgboost_caret", "xgboost_test_model_1500rows.rds"))



BH_test_id_lookup <- read_rds(here::here("create_dataset", "BH_test_id_lookup.rds"))
eval_test <- data.frame(BH_test_id_lookup$REF_ID, BH_test$Higher, preds2) %>% 
  rename(REF_ID = BH_test_id_lookup.REF_ID, acctaul = BH_test.Higher, predicted = preds2) %>%  
  mutate(acctaul = ifelse(acctaul == "Higher", 1, 0),
         predicted = ifelse(predicted == "Higher", 1,0)) %>% 
  group_by(REF_ID) %>%  
  summarize(acctaul = mean(acctaul), predicted = round(mean(predicted),4))



  
  
  
  
