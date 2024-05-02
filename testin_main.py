import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv('C:/Users/manya/OneDrive/Documents/Medical/Medical_insurance.csv', encoding='latin')
print('Shape before duplicating values :',df.shape)
df=df.drop_duplicates()
print('shape after duplicates get removed:',df.shape)


print(df.head(10))


# Converting the categorical variables into numeric using get_dummies
# Perform one-hot encoding
df_encoded = pd.get_dummies(df, columns=['region'],dtype= int)

# Print the encoded DataFrame
print(df_encoded)

charges = df['charges']

# describing dataset
df.head()
df.tail()
df.info()
# Looking at the descriptive statistics of the data
df.describe()
# Finding unique values for each column
# TO understand which column is categorical and which one is Continuous
# Typically if the number of unique values are < 20 then the variable is likely to be a category otherwise continuous
df.nunique()

#Replacing outliers for 'bmi'
# Finding nearest values to 55 mark
df['bmi'][df['bmi']<55].sort_values(ascending=False)
# Replacing outliers with nearest possible value
df['bmi'][df['bmi']>55] =51.13
# Finding how many missing values are there for each column
df.isnull().sum()



# Defining a function to find the statistical relationship with all the categorical variables
def FunctionAnova(inpData, TargetVariable, CategoricalPredictorList):
    from scipy.stats import f_oneway

    # Creating an empty list of final selected predictors
    SelectedPredictors=[]

    print('##### ANOVA Results ##### \n')
    for predictor in CategoricalPredictorList:
        CategoryGroupLists=inpData.groupby(predictor)[TargetVariable].apply(list)
        AnovaResults = f_oneway(*CategoryGroupLists)

        # If the ANOVA P-Value is <0.05, that means we reject H0
        if (AnovaResults[1] < 0.05):
            print(predictor, 'is correlated with', TargetVariable, '| P-Value:', AnovaResults[1])
            SelectedPredictors.append(predictor)
        else:
            print(predictor, 'is NOT correlated with', TargetVariable, '| P-Value:', AnovaResults[1])

    return(SelectedPredictors)

#Calling the function to check which categorical variables are correlated with target
CategoricalPredictorList=['smoker', 'region','sex','children']
FunctionAnova(inpData=df, TargetVariable='charges', CategoricalPredictorList=CategoricalPredictorList)


# STEP 16
SelectedColumns=['age','sex','smoker','region']

# Selecting final columns
DataForML=df[SelectedColumns]
DataForML.head()

#Converting the nominal variable to numeric using get_dummies()
# Treating all the nominal variables at once using dummy variables
DataForML_Numeric=pd.get_dummies(DataForML,dtype=int)

# Adding Target Variable to the data
DataForML_Numeric['charges']=df['charges']

# Printing sample rows
DataForML_Numeric.head()



####################################################################################################
######   MACHINE LEARNING ##########################################################################
# Printing all the column names for our reference
DataForML_Numeric.columns
#Separate Target Variable and Predictor Variables
TargetVariable='charges'
Predictors=['age', 'sex_female', 'sex_male', 'smoker_no', 'smoker_yes',
       'region_northeast', 'region_northwest', 'region_southeast',
       'region_southwest']

#
X=DataForML_Numeric[Predictors].values
y=DataForML_Numeric[TargetVariable].values

# Split the data into training and testing set
from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=428) # original value 428

### Standardization of data ###
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# MinMax SCALER
PredictorScaler=MinMaxScaler()

# Storing the fit object for later reference
PredictorScalerFit=PredictorScaler.fit(X)

# Generating the standardized values of X
X=PredictorScalerFit.transform(X)

# Split the data into training and testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Sanity check for the sampled data
# print(X_train.shape)
# print(y_train.shape)
# print(X_test.shape)
# print(y_test.shape)


#Multiple Linear Regression
from sklearn.linear_model import LinearRegression

# Printing all the parameters of Linear regression
RegModel = LinearRegression()
# print(RegModel)

# Creating the model on Training Data
LREG=RegModel.fit(X_train,y_train)
prediction=LREG.predict(X_test)

from sklearn import metrics
# Measuring Goodness of fit in Training data
# print('R2 Value:',metrics.r2_score(y_train, LREG.predict(X_train)))

print('\n##### Model Validation and Accuracy Calculations ##########')
# Printing some sample values of prediction
TestingDataResults=pd.DataFrame(data=X_test, columns=Predictors)
TestingDataResults[TargetVariable]=y_test
TestingDataResults[('Predicted'+TargetVariable)]=np.round(prediction)

# Printing sample prediction values
# print(TestingDataResults.head())
# Calculating the error for each row
TestingDataResults['APE']=100 * ((abs(TestingDataResults['charges']-TestingDataResults['Predictedcharges']))/TestingDataResults['charges'])
MAPE=np.mean(TestingDataResults['APE'])
MedianMAPE=np.median(TestingDataResults['APE'])

Accuracy =100 - MAPE
MedianAccuracy=100- MedianMAPE
print('Mean Accuracy on test data:', Accuracy) # Can be negative sometimes due to outlier
print('Median Accuracy on test data:', MedianAccuracy)


# Defining a custom function to calculate accuracy
# Make sure there are no zeros in the Target variable if you are using MAPE
def Accuracy_Score(orig,pred):
    MAPE = np.mean(100 * (np.abs(orig-pred)/orig))
    #print('#'*70,'Accuracy:', 100-MAPE)
    return(100-MAPE)

# Custom Scoring MAPE calculation
from sklearn.metrics import make_scorer
custom_Scoring=make_scorer(Accuracy_Score, greater_is_better=True)

# Custom Scoring MAPE calculation
from sklearn.metrics import make_scorer
custom_Scoring=make_scorer(Accuracy_Score, greater_is_better=True)

# Importing cross validation function from sklearn
from sklearn.model_selection import cross_val_score
#Running 10-fold cross-validation on a given algorithm
# Passing full data X and y because the K-fold will split the data and automatically choose train/test
Accuracy_Values=cross_val_score(RegModel, X , y, cv=10, scoring=custom_Scoring)
print('\nAccuracy values for 10-fold Cross Validation:\n',Accuracy_Values)
print('\nFinal Average Accuracy of the model:', round(Accuracy_Values.mean(),2))


##################################################################################################################
########### GUI 

import tkinter as tk
from tkinter import messagebox, ttk
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

class MedicalInsurancePredictionApp:
    def __init__(self, master):
        self.master = master
        self.master.title('Medical Insurance Charges Prediction')
        self.data = pd.read_csv('C:/Users/manya/OneDrive/Documents/Medical/Medical_insurance.csv')
        self.widgets = []

        # Drop duplicates
        self.data = self.data.drop_duplicates()

        # Converting categorical variables into numeric using one-hot encoding
        self.data_encoded = pd.get_dummies(self.data, columns=['region', 'sex', 'smoker'], dtype=int)

        # Selecting predictor columns
        self.selected_columns = ['age', 'bmi', 'sex_female', 'sex_male', 'children','smoker_no', 'smoker_yes', 'region_northeast', 'region_northwest', 'region_southeast', 'region_southwest']
        self.X = self.data_encoded[self.selected_columns].values
        self.y = self.data_encoded['charges'].values

        # Splitting data into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

        # Fitting the model
        self.model = LinearRegression()
        self.model.fit(self.X_train, self.y_train)

        self.create_widgets()

    def create_widgets(self):
        for i, column in enumerate(self.data.columns[:-1]):
            label = tk.Label(self.master, text=column + ': ')
            label.grid(row=i, column=0)
            
            if self.data[column].dtype == 'object':  # Handling categorical variables
                values = self.data[column].unique()
                combobox = ttk.Combobox(self.master, values=values, state="readonly")
                combobox.grid(row=i, column=1)
                self.widgets.append((combobox, column))
            else:  # Handling numerical variables
                current_val_label = tk.Label(self.master, text='0.0')
                current_val_label.grid(row=i, column=2)
                slider = ttk.Scale(self.master, from_=self.data[column].min(), to=self.data[column].max(), orient="horizontal", command=lambda val, label=current_val_label: label.config(text=f'{float(val):.2f}'))
                slider.grid(row=i, column=1)
                self.widgets.append((slider, current_val_label, column))

        predict_button = tk.Button(self.master, text='Predict Charges', command=self.predict_charges)
        predict_button.grid(row=len(self.data.columns[:-1]), columnspan=3)

    def predict_charges(self):
        inputs = [0] * len(self.selected_columns)  # Initialize inputs with zeros

        for widget in self.widgets:
            if isinstance(widget[0], ttk.Combobox):  # Handling categorical variables
                value = widget[0].get().strip("[]'").replace('"', '')
                if widget[1] == "sex":
                    inputs[self.selected_columns.index(f'sex_{value}')] = 1
                elif widget[1] == "smoker":
                    inputs[self.selected_columns.index(f'smoker_{value}')] = 1
                else:  # For 'region'
                    inputs[self.selected_columns.index(f'region_{value}')] = 1
            else:  # Handling numerical variables
                if widget[2] == "bmi":  # Do not convert 'bmi' to integer
                    inputs[self.selected_columns.index(widget[2])] = float(widget[0].get())
                else:
                    inputs[self.selected_columns.index(widget[2])] = int(float(widget[0].get()))  # Convert slider value to integer

        price = self.model.predict([inputs])
        messagebox.showinfo('Predicted Charges', f'The predicted insurance charges is ${price[0]:.2f}')

if __name__ == '__main__':
    root = tk.Tk()
    app = MedicalInsurancePredictionApp(root)
    root.mainloop()


           

        