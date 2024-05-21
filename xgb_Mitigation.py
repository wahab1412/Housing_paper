from Data_info import *
import Mitigation as mt
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


data_df_orginal = pd.read_csv("../Ml_ready_all_percent.csv")
data_class = Data_info(data_df_orginal)
eps_list = [0.001]
constraint = "DP"  
loss = "square" 
X_train, y_train , X_val, y_val = pd.DataFrame(data_class.X_train_scaled),  pd.Series(data_class.y_percentage_train_scaled), pd.DataFrame(data_class.X_val_scaled), pd.Series(data_class.y_percentage_val_scaled)
print('----------------')
print('Theta', len(mt.Theta))
learner = mt.XGB_Regression_Learner(mt.Theta)


info = str('; loss: ' + loss + '; eps list: '+str(eps_list)) + '; Solver: '+ learner.name



a1 = pd.Series(data_class.protected_white_vs_other)
print('Starting experiment. ' + info)
result= mt.fair_train_test(X_train, a1, y_train , X_val, a1, y_val,eps_list,learner ,constraint=constraint, loss=loss)
mt.read_result_list([result])  
results_df = pd.DataFrame()
results_df['0.001'] = result['test_eval'][0.001]['pred'].mean(1)
results_df.to_csv('results/XGB_predection_white_vs_other.csv', index=False)


a1 = pd.Series(data_class.protected_Hispanic_vs_other)
print('Starting experiment. ' + info)
result= mt.fair_train_test(X_train, a1, y_train , X_val, a1, y_val,eps_list,learner ,constraint=constraint, loss=loss)
mt.read_result_list([result]) 
results_df = pd.DataFrame()
results_df['0.001'] = result['test_eval'][0.001]['pred'].mean(1)
results_df.to_csv('results/XGB_predection_Hispanic_vs_other.csv', index=False)




a1 = pd.Series(data_class.protected_white_none_Hispanic_vs_other)
print('Starting experiment. ' + info)
result= mt.fair_train_test(X_train, a1, y_train , X_val, a1, y_val,eps_list,learner ,constraint=constraint, loss=loss)
mt.read_result_list([result]) 
results_df = pd.DataFrame()
results_df['0.001'] = result['test_eval'][0.001]['pred'].mean(1)
results_df.to_csv('results/XGB_predection_white_none_Hispanic_vs_other.csv', index=False)
