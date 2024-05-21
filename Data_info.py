from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import  mean_squared_error
import pandas as pd
import numpy   as np


class Data_info:
    def __init__(self, dataframe):

        dataframe = dataframe.loc[:, ~dataframe.columns.str.contains('Tract', case=False)]
        dataframe = dataframe.drop(['lat_redfin','long_redfin','Zip Code_x'], axis=1)
        dataframe['appraised_val_2023_percent'] = (dataframe['appraised_val_2023'] - dataframe['appraised_val_2022']) / dataframe['appraised_val_2022'] * 100
        dataframe['appraised_val_2022_percent'] = (dataframe['appraised_val_2022'] - dataframe['appraised_val_2021']) / dataframe['appraised_val_2021'] * 100
        dataframe['appraised_val_2021_percent'] = (dataframe['appraised_val_2021'] - dataframe['appraised_val_2020']) / dataframe['appraised_val_2020'] * 100
        c1 = len(dataframe)

        Q1 = dataframe['appraised_val_2023_percent'].quantile(0.25)
        Q3 = dataframe['appraised_val_2023_percent'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 -  2 * IQR
        upper_bound = Q3 +  2 * IQR
        # Filter rows that are within the bounds
        dataframe = dataframe[(dataframe['appraised_val_2023_percent'] >= lower_bound) & (dataframe['appraised_val_2023_percent'] <= upper_bound)]

        Q1 = dataframe['appraised_val_2022_percent'].quantile(0.25)
        Q3 = dataframe['appraised_val_2022_percent'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 -  2 * IQR
        upper_bound = Q3 +  2 * IQR
        # Filter rows that are within the bounds
        dataframe = dataframe[(dataframe['appraised_val_2022_percent'] >= lower_bound) & (dataframe['appraised_val_2022_percent'] <= upper_bound)]

        Q1 = dataframe['appraised_val_2021_percent'].quantile(0.25)
        Q3 = dataframe['appraised_val_2021_percent'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 -  2 * IQR
        upper_bound = Q3 +  2 * IQR
        # Filter rows that are within the bounds
        dataframe = dataframe[(dataframe['appraised_val_2021_percent'] >= lower_bound) & (dataframe['appraised_val_2021_percent'] <= upper_bound)]
        dataframe = dataframe.drop('appraised_val_2021_percent', axis=1)
        dataframe = dataframe.drop('appraised_val_2022_percent', axis=1)
        dataframe = dataframe.drop('appraised_val_2023_percent', axis=1)
        c2 = len(dataframe)
        filtered_df = dataframe.dropna()
        print('Reomve outliers based on appraised_val_2023.', c1 - c2,' Data removed' )
        data_df = filtered_df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
        

        data_df['house_age_train'] = (2020 - data_df['year_x']) / 250
        data_df['house_age_test'] = (2021 - data_df['year_x']) / 250
        data_df['house_age_val'] = (2022 - data_df['year_x']) / 250
        data_df = data_df.drop('year_x', axis=1)


        for year in range(2015,2021):
            if year > 2018:
                data_df.rename(columns={f"{year}_Houses_Values_Estimate!!Total:!!$1,000,000 to $1,499,999": f'{year}_Houses_Values_Estimate!!Total!!$1,000,000 to $1,499,999'}, inplace=True)
                data_df.rename(columns={f"{year}_Houses_Values_Estimate!!Total:!!$1,500,000 to $1,999,999": f'{year}_Houses_Values_Estimate!!Total!!$1,500,000 to $1,999,999'}, inplace=True)
                data_df.rename(columns={f"{year}_Houses_Values_Estimate!!Total:!!$2,000,000 or more": f'{year}_Houses_Values_Estimate!!Total!!$2,000,000 or more'}, inplace=True)
            
            data_df[f'{year}_Houses_Values_Estimate!!Total!!$1,000,000 to $1,499,999'] = data_df[f'{year}_Houses_Values_Estimate!!Total!!$1,000,000 to $1,499,999'] + data_df[f'{year}_Houses_Values_Estimate!!Total!!$1,500,000 to $1,999,999'] + data_df[f'{year}_Houses_Values_Estimate!!Total!!$2,000,000 or more']
            data_df.rename(columns={f'{year}_Houses_Values_Estimate!!Total!!$1,000,000 to $1,499,999': f'{year}_Houses_Values_Estimate!!Total!!$1,000,000 or more'}, inplace=True)
            data_df = data_df.drop([f'{year}_Houses_Values_Estimate!!Total!!$1,500,000 to $1,999,999',f'{year}_Houses_Values_Estimate!!Total!!$2,000,000 or more'], axis=1)

            # Definfing sensitive features Based on 2020 race data.
        cols = ['2020_Race_Data_Percent!!RACE!!One race!!White',
                '2020_Race_Data_Percent!!RACE!!One race!!Black or African American',
                '2020_Race_Data_Percent!!RACE!!One race!!American Indian and Alaska Native',
                '2020_Race_Data_Percent!!RACE!!One race!!Asian',
                '2020_Race_Data_Percent!!RACE!!One race!!Native Hawaiian and Other Pacific Islander',
                '2020_Race_Data_Percent!!RACE!!One race!!Some other race',
                '2020_Race_Data_Percent!!HISPANIC OR LATINO AND RACE!!Hispanic or Latino (of any race)']


        # Hispanic: if 'Hispanic' >= 50% ,else 'None Hispanic'.
        print('First sensitive features')
        protected_Hispanic_vs_other = ['Hispanic' if value >= 0.5 else 'None Hispanic' for value in data_df[cols[-1]]]
        print('Hispanic: ', protected_Hispanic_vs_other.count('Hispanic'))
        print('None Hispanic: ', protected_Hispanic_vs_other.count('None Hispanic'))
        protected_Hispanic_vs_other = np.array(protected_Hispanic_vs_other)

        print('')
        ####################

        print('Second sensitive features')
        # 'White': if majorty is white
        protected_white_vs_other = data_df[cols].apply(lambda row: 'White' if row.iloc[0] >= row.iloc[1:-1].sum() else 'None White', axis=1)
        print('White: ', list(protected_white_vs_other).count('White'))
        print('None White: ', list(protected_white_vs_other).count('None White'))
        protected_white_vs_other = np.array(protected_white_vs_other)



        ##############
        print('')
        print('Third sensitive features')
        protected_four_Hispanic_White = []
        for i in range(len(data_df)):
            row = data_df.iloc[i][cols]
            if row.iloc[-1] < 0.5 and row.iloc[0] >= row.iloc[1:-1].sum():
                protected_four_Hispanic_White.append('White None Hispanic')
            else:
                protected_four_Hispanic_White.append('Other')


        print('White None Hispanic: ',list(protected_four_Hispanic_White).count('White None Hispanic'))
        print('Other: ',list(protected_four_Hispanic_White).count('Other'))
        print('')
        protected_four_Hispanic_White = np.array(protected_four_Hispanic_White)




        data_df = data_df.loc[:, ~data_df.columns.str.contains('Race', case=False)]



        train_data = data_df.loc[:, ~data_df.columns.str.contains('2019_', case=False)]
        train_data = train_data.loc[:, ~train_data.columns.str.contains('2020_', case=False)]
        X_train = train_data.drop(['appraised_val_2023','appraised_val_2022','appraised_val_2021','house_age_test','house_age_val'], axis=1)
        y_train = data_df['appraised_val_2021'].values


        test_data = data_df.loc[:, ~data_df.columns.str.contains('2010_', case=False)]
        test_data = test_data.loc[:, ~test_data.columns.str.contains('2020_', case=False)]
        X_test = test_data.drop(['appraised_val_2023','appraised_val_2022','appraised_val_2020','house_age_val','house_age_train'], axis=1)
        y_test = data_df['appraised_val_2022'].values


        val_data = data_df.loc[:, ~data_df.columns.str.contains('2010_', case=False)]
        val_data = val_data.loc[:, ~val_data.columns.str.contains('2011_', case=False)]
        X_val = val_data.drop(['appraised_val_2023','appraised_val_2020','appraised_val_2021','house_age_test','house_age_train'], axis=1)
        y_val = data_df['appraised_val_2023'].values


        for_plus_train = X_train.values[:,-2:-1].flatten()
        for_plus_test = X_test.values[:,-2:-1].flatten()
        for_plus_val = X_val.values[:,-2:-1].flatten()


        
        y_percentage_train = (y_train - for_plus_train ) / for_plus_train
        y_percentage_test = (y_test - for_plus_test ) / for_plus_test
        y_percentage_val = (y_val - for_plus_val ) / for_plus_val


        X_train_columns_to_scale = X_train.columns[X_train.gt(1).any()]
        X_test_columns_to_scale = X_test.columns[X_test.gt(1).any()]
        X_val_columns_to_scale = X_val.columns[X_val.gt(1).any()]

        # Normalizing the  data
        scaler_X = MinMaxScaler()
        X_train_normlized = scaler_X.fit_transform(X_train[X_train_columns_to_scale].values)
        X_test_normlized = scaler_X.transform(X_test[X_test_columns_to_scale].values)
        X_val_normlized = scaler_X.transform(X_val[X_val_columns_to_scale].values)

        # Normalizing the y data
        scaler_y_percent = MinMaxScaler()
        y_percentage_train_scaled = (scaler_y_percent.fit_transform(y_percentage_train.reshape(-1,1))).flatten()
        y_percentage_val_scaled = (scaler_y_percent.transform(y_percentage_val.reshape(-1,1))).flatten()
        y_percentage_test_scaled = (scaler_y_percent.transform(y_percentage_test.reshape(-1,1))).flatten()

        X_train_scaled = X_train.copy()
        X_train_scaled[X_train_columns_to_scale] = X_train_normlized
        X_train_scaled = X_train_scaled.values


        X_test_scaled = X_test.copy()
        X_test_scaled[X_test_columns_to_scale] = X_test_normlized
        X_test_scaled = X_test_scaled.values

        
        X_val_scaled = X_val.copy()
        X_val_scaled[X_val_columns_to_scale] = X_val_normlized
        X_val_scaled = X_val_scaled.values


        self.X_train = X_train
        self.X_test = X_test
        self.X_val = X_val

        self.X_train_scaled = X_train_scaled
        self.X_test_scaled = X_test_scaled
        self.X_val_scaled = X_val_scaled


        self.y_percentage_train = y_percentage_train
        self.y_percentage_test = y_percentage_test
        self.y_percentage_val = y_percentage_val

        self.y_percentage_train_scaled = y_percentage_train_scaled
        self.y_percentage_test_scaled = y_percentage_test_scaled
        self.y_percentage_val_scaled = y_percentage_val_scaled
        

        self.scaler_y_percent = scaler_y_percent

        self.protected_white_vs_other = protected_white_vs_other
        self.protected_Hispanic_vs_other = protected_Hispanic_vs_other
        self.protected_white_none_Hispanic_vs_other = protected_four_Hispanic_White
        

    

    # MSE calculation
   
    def calc_mse_percent_val(self, y_pred):
        y_pred_descalled = self.scaler_y_percent.inverse_transform(y_pred.reshape(-1,1)).flatten()
        return  np.sqrt(mean_squared_error(self.y_percentage_val, y_pred_descalled))

    def calc_mse_percent_test(self, y_pred):
        y_pred_descalled = self.scaler_y_percent.inverse_transform(y_pred.reshape(-1,1)).flatten()
        return  np.sqrt(mean_squared_error(self.y_percentage_test, y_pred_descalled))

    def get_pred_descalled_from_perecent(self, y_pred):
        y_pred_descalled = self.scaler_y_percent.inverse_transform(y_pred.reshape(-1,1)).flatten()
        return  y_pred_descalled

