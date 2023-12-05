# Importing the required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

from onnxmltools.convert import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType
from onnxruntime import InferenceSession

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# Load the data
data = "data/Data_MaanCummings_2011_ AmNat52931_ByPop.txt"
df = pd.read_table(data)

#  Display the first five rows
print(df.head())

# Print out the column names
print(df.columns)


# Rename columns

columns = ['pop', 'pop_number', 'n_tox', 'toxicity', 'toxicity_se', 
'toxicity_normalised', 'toxicity_norm_se', 'viewer', 'background', 'noise', 
'v_or_d', 'vs_lumin_cont', 'vs_chrom_cont', 'vs_conspic', 'vi_brightness', 'vi_bright_se', 
'vi_bright_normalised', 'bird_lumin_cont', 'bird_chrom_cont', 'bird_conspic_cont']

df.columns = columns
print(df.head())

# Create a data subset
df = df[['pop', 'n_tox', 'toxicity', 'viewer', 'background', 
'noise', 'v_or_d', 'vs_lumin_cont', 'vs_chrom_cont', 'vs_conspic','vi_brightness']]

print(df.head())

# Obtain data type information
print(df.info())

# Describe basic statistics
print(df.describe())

# Check for missing values
print(df.isnull().sum())

# Drop the missing values & recalculate the sums
df = df.dropna()
print("missing values count \n")
print(df.isnull().sum())

# Check for duplicate values
print("number of duplicated values: " , df.duplicated().sum())

# Plot a histogram of the target variable
sns.histplot(df.toxicity)
plt.show()

# Obtain correlation values
correlation = df[["n_tox", "toxicity", "vs_lumin_cont", "vs_chrom_cont", "vs_conspic", "vi_brightness"]].corr()
print(correlation)

# Plot a correlation matrix
sns.heatmap(correlation, cmap="crest", annot=True)
plt.show()

# Split the data into train and test
df_train, df_test = train_test_split(df, test_size=0.2, shuffle=True, 
random_state=2)

# Print out the shape of the new DataFrames
print("Train:", df_train.shape[0])
print("Test:", df_test.shape[0])

# Isolate and transform the target variable
y_train = np.log1p(df_train['toxicity'].values)
y_test = np.log1p(df_test['toxicity'].values)

# Delete the target variable from DataFrames
del df_train['toxicity']
del df_test['toxicity']

# Encode the data
x_train = pd.get_dummies(df_train, dtype="int")
x_test = pd.get_dummies(df_test, dtype="int")

# Create the DMatrices
dtrain = xgb.DMatrix(x_train, label=y_train)
dtest = xgb.DMatrix(x_test, label=y_test)

# Define XGBoost parameters
xgb_params = {'eval_metric': 'rmse'}

# Fit the model
model = xgb.train(xgb_params, dtrain, num_boost_round=500)

# Obtain predictions
y_pred = model.predict(dtest)

# Calculate RMSE
score = np.sqrt(mean_squared_error(y_test, y_pred))
print(score)

# Update the parameter dictionary
xgb_params = {
 'colsample_bytree': 0.8,
 'eval_metric': 'rmse',
 'learning_rate': 0.1,
 'max_depth': 3,
 'min_child_weight': 5,
 'objective': 'reg:squarederror',
 'random_state': 10
}

# Retrain the model
model = xgb.train(xgb_params, dtrain, num_boost_round=86)

# Obtain updated predictions
y_pred2 = model.predict(dtest)

# Calculate and display RMSE
score2 = np.sqrt(mean_squared_error(y_test, y_pred2))
print(score2)

# Create the first plot
sns.scatterplot(x=range(len(y_test)), y=y_test)
sns.scatterplot(x=range(len(y_pred)), y=y_pred)

plt.legend(["Testing data", "Baseline prediction"], loc="lower left", bbox_to_anchor=(0, 1))
plt.show()

# Create the second plot

sns.scatterplot(x= range(len(y_test)), y=y_test)
sns.scatterplot(x= range(len(y_pred)), y=y_pred)
sns.scatterplot(x= range(len(y_pred2)), y=y_pred2)

plt.legend(["Testing data", "Baseline prediction", "Tuned prediction"], loc="lower left", bbox_to_anchor=(0, 1))
plt.show()

# Define input shape
tensor = FloatTensorType([None, x_train.shape[1]])

# Create input list
initial_type = [('float_input', tensor)]

# Rename model features
model.feature_names = [f"f{num}" for num in range(len(model.feature_names))]
print(model.feature_names)


# Convert the model
onnx_model = convert_xgboost(model, initial_types=initial_type)

# Write the final file
with open("model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

# Start inference session
session = InferenceSession("model.onnx")

# Obtain input and output names
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name


# Display input and output names
print(input_name, output_name)