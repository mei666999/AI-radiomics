library(openxlsx)  
library(readxl)
file_path <- "selected_features_after_u_test_data.xlsx"   
df <- read_excel(file_path) 
mydata <- df
library(glmnet)
y <- as.matrix(mydata[, 1])  
x <- as.matrix(mydata[, 2:65]) 
lasso_model <- glmnet(x, y, family = "binomial", 
                      alpha = 1) 
print(lasso_model) 
library(glmnet)
plot(lasso_model,
     xvar = "lambda")
cv_model <- cv.glmnet(x, y, family = "binomial",alpha = 1,nfolds = 10)
plot(cv_model)
lambda_min <- cv_model$lambda.min
lambda_min
lambda_1se <- cv_model$lambda.1se
lambda_1se
coef_lasso <- coef(lasso_model,
                   s = lambda_1se)
coef_lasso
library(ggplot2)    
coef_data <- as.data.frame(as.matrix(coef_lasso))  
coef_data$Feature <- rownames(coef_data)  
colnames(coef_data) <- c("Coefficient", "Feature")    
coef_data <- coef_data[coef_data$Feature != "(Intercept)", ]    
coef_data <- coef_data[order(abs(coef_data$Coefficient), decreasing = TRUE), ]  
print(coef_data)
write.csv(coef_data, "coef_output.csv", row.names = TRUE)  
ggplot(coef_data[coef_data$Coefficient != 0, ], aes(x = reorder(Feature, Coefficient), y = Coefficient)) +  
  geom_bar(stat = "identity", fill = "#0072B2") +    
  coord_flip() +    
  theme_minimal() +  
  labs(title = "Feature Coefficients from Lasso Regression",  
       x = "Features",  
       y = "Coefficient") +  
  theme(axis.text.y = element_text(size = 10))
library(tidymodels)
library(baguette)
library(discrim)
library(readxl)  
library(xgboost)  
library(kknn)  
library(ranger) 
library(yardstick) 
library(dplyr) 
df <- read_excel("significant_features_data_1.xlsx") %>%   
  mutate(yanzhong=factor(yanzhong)) 
rec <- recipe(yanzhong~.,df)
set.seed(123)
xgb_mod <- boost_tree() %>%             
  set_engine("xgboost") %>%   
  set_mode("classification")
dt_mod <- decision_tree() %>%             
  set_engine("rpart") %>%             
  set_mode("classification")
logistic_mod <- logistic_reg() %>%         
  set_engine('glm')
nnet_mod <- mlp() %>%            
  set_engine('nnet') %>%            
  set_mode('classification')       
naivebayes_mod <- naive_Bayes() %>%         
  set_engine('naivebayes')
knn_mod <- nearest_neighbor() %>% 
  set_engine('kknn') %>% 
  set_mode('classification') 
rf_mod <- rand_forest() %>%
  set_engine('ranger') %>% 
  set_mode('classification') 
svm_mod <- svm_rbf() %>% 
  set_engine('kernlab') %>% 
  set_mode('classification')
wf <- workflow_set(preproc=list(rec), 
                   models=list(xgb=xgb_mod, 
                               dt=dt_mod, 
                               log= logistic_mod, 
                               nb=naivebayes_mod, 
                               nnet=nnet_mod, 
                               knn=knn_mod, 
                               rf=rf_mod,
                               svm=svm_mod))
folds <- bootstraps(df,1000)
ctr <- control_resamples(save_pred = TRUE)
wf_res <- wf %>% 
  workflow_map("fit_resamples",               
               resamples=folds,  
               control=ctr)
predictions <- wf_res %>%   
  collect_predictions()
rank_results(wf_res,rank_metric = "accuracy") %>%   
  filter(.metric=="accuracy") %>%   
  select(model,mean)
rank_results(wf_res,rank_metric = "brier_class") %>%   
  filter(.metric=="brier_class") %>%   
  select(model,mean)
roc_auc_results <- rank_results(wf_res, rank_metric = "roc_auc") %>%  
  filter(.metric == "roc_auc") %>%  
  select(model, mean, std_err)    
roc_auc_results <- roc_auc_results %>%  
  mutate(  
    lower_ci = mean - 1.96 * std_err,  # 计算下限  
    upper_ci = mean + 1.96 * std_err   # 计算上限  
  )    
print(roc_auc_results)  
rank_results(wf_res,rank_metric = "roc_auc") %>%   
  filter(.metric=="roc_auc") %>%   
  select(model,mean)
model_names <- wf_res$wflow_id
sensitivity_list = list()
specificity_list = list()
for (model_name in model_names) {
  preds = wf_res %>%
    extract_workflow_set_result(model_name) %>%
    collect_predictions()
  sensitivity = preds %>%
    yardstick::sensitivity(truth = yanzhong, estimate =.pred_class)
  sensitivity_list[[model_name]] = sensitivity
  specificity = preds %>%
    yardstick::specificity(truth = yanzhong, estimate =.pred_class)
  specificity_list[[model_name]] = specificity
}
sensitivity_mean_results = data.frame(model = names(sensitivity_list), mean_sensitivity = NA)
for (i in 1:length(sensitivity_list)) {
  sensitivity_mean_results$mean_sensitivity[i] = mean(sensitivity_list[[i]]$.estimate)
}
print(sensitivity_mean_results)
specificity_mean_results = data.frame(model = names(specificity_list), mean_specificity = NA)
for (i in 1:length(specificity_list)) {
  specificity_mean_results$mean_specificity[i] = mean(specificity_list[[i]]$.estimate)
}
print(specificity_mean_results)
combined_results = merge(roc_auc_results, sensitivity_mean_results, by = "model")
combined_results = merge(combined_results, specificity_mean_results, by = "model")
