import os
os.getcwd()
os.chdir('E:')
import pandas as pd
data = pd.read_excel('.xlsx')
def min_max_normalize(column):
    min_val = column.min()
    max_val = column.max()
    if max_val - min_val == 0:  
        return column
    else:
        return (column - min_val) / (max_val - min_val)
for col in data.columns[2:1036]: 
    data[col] = min_max_normalize(data[col])
data.to_excel('normalized_data.xlsx', index=False)
import pandas as pd  
from scipy.stats import mannwhitneyu  
file_path = 'normalized_data.xlsx'   
data = pd.read_excel(file_path)   
group_column = 'fenzu'  
feature_columns = data.columns[2:1036]  
results = []  
for feature in feature_columns:    
    case_data = data[data[group_column] == 'PT'][feature]  
    control_data = data[data[group_column] == 'HC'][feature]    
    stat, p_value = mannwhitneyu(case_data, control_data, alternative='two-sided')    
    results.append({'Feature': feature, 'p_value': p_value})    
results_df = pd.DataFrame(results)    
significant_results = results_df[results_df['p_value'] < 0.05]    
significant_features = significant_results['Feature'].tolist()    
significant_data = data[[group_column] + significant_features]    
significant_data.to_excel('significant_features_data.xlsx', index=False)
import pandas as pd    
df = pd.read_excel('significant_features_data.xlsx', usecols="B:DD")     
df.columns = ['Feature' + str(i) for i in range(1, len(df.columns) + 1)]   
df.insert(0, 'fenzu', pd.read_excel('significant_features_data.xlsx', usecols="A").iloc[:, 0])  
import numpy as np    
corr_matrix = df.iloc[:, 1:].corr(method='spearman')    
high_corr_pairs = [(i, j) for i in corr_matrix.columns for j in corr_matrix.columns if i != j and corr_matrix.loc[i, j] > 0.9]
from scipy.stats import mannwhitneyu  
import pandas as pd    
results_list = []    
for pair in high_corr_pairs:  
    feature1, feature2 = pair    
    pt_group1 = df[df['fenzu'] == 'PT'][feature1]  
    hc_group1 = df[df['fenzu'] == 'HC'][feature1]  
    pt_group2 = df[df['fenzu'] == 'PT'][feature2]  
    hc_group2 = df[df['fenzu'] == 'HC'][feature2]  
    p_value1 = mannwhitneyu(pt_group1.dropna(), hc_group1.dropna(), alternative='two-sided').pvalue  
    p_value2 = mannwhitneyu(pt_group2.dropna(), hc_group2.dropna(), alternative='two-sided').pvalue    
    selected_feature = feature1 if p_value1 < p_value2 else feature2  
    results_list.append({'Feature1': feature1, 'Feature2': feature2,   
                          'P_value_Feature1': p_value1, 'P_value_Feature2': p_value2,   
                          'Selected_Feature': selected_feature})    
results = pd.DataFrame(results_list)    
results.to_excel('selected_features_after_u_test.xlsx', index=False)
