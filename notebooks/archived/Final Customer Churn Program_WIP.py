#!/usr/bin/env python
# coding: utf-8

# # Customer Churn - C744 Perfomance Assessment

# ## Importing required libraries, Data Collection & Data Exploration

# ### 1. Importing Required Libraries

# In[1]:


# Data processesing, CSV I/O 
import pandas as pd
import numpy as np

# Data Visualizations
import matplotlib as mt
import matplotlib.pyplot as plt
import seaborn as sns
import dexplot as dx
import itertools
import warnings

## Data Pre-processing, Model Building and Assessments

from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn import under_sampling, over_sampling

import statsmodels.api as sm  # Stats model for ANOVA

from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, roc_auc_score, roc_curve

# get_ipython().run_line_magic('matplotlib', 'inline')


# ### 2. Data Collection

# In[2]:


# Notebook settings
pd.set_option('display.max_columns', 999)
pd.set_option("display.max_rows", 100)
warnings.simplefilter(action='ignore', category=FutureWarning)

# In[3]:


telecom = pd.read_csv("../data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
telecom.head()

# In[4]:


# Get the overview of data
telecom.info()

## Obs: 
##  1. Out of all features, tenure, MonthlyCharges and TotalCharges looks like continuous variables. 
##        Need to explore more to decide type (categorical vs numerical) of each feature
##   1.1. Object type of TotalCharges is object. Need to change type to float for proper processing. 
##   1.2. SeniorCitizen is int64, but is a categorical feature (confirmed later by unique values)
##  2. 7043 observations and 21 features
##   2.1. 20 features (Predictor Variables) and 1 Dependent/Response variable - Churn -> 
##           since we need to predict finally if the customer would be retained or Churned based on features
## Need to check the individual statistics of the features and response variables


# In[5]:


# Convert the type of TotalCharges & check for Missing Values
telecom["TotalCharges"] = telecom["TotalCharges"].apply(pd.to_numeric, downcast="float", errors="coerce")
telecom.isnull().sum()

# In[6]:


# Fix Missing entries in TotalCharges
telecom["TotalCharges"] = telecom["TotalCharges"].apply(pd.to_numeric, downcast="float", errors="coerce").fillna(0)
telecom.isnull().sum()

# In[7]:


telecom["SeniorCitizen"] = telecom.SeniorCitizen.astype("str")

# In[8]:


telecom.mean()

# In[9]:


# List number of unique entries in each of the feature
telecom.nunique()

## Obs: It is confirmed that tenure, MonthlyCharges and TotalCharges are numeric/continuous variables
##      The customerID feature is an identifier columns for each observation, with unique value for each observation.
##      Remaining 17 variables are categorical in nature.


# In[10]:


# List unique entries in each of the features
telecom_columns = telecom.columns.tolist()
telecom_columns

for i in telecom_columns:
    print(i, ": ", telecom[i].unique())

# In[ ]:


# ## Feature Engineering
# ##### Obs: Some categorical levels can be condensed to reduced the # of levels. e.g OnlineSecurity => No internet service - > can be converted to No

# In[11]:


# Some categorical feature levels can be refined for a cleaner classification between catergory levels

# 1. Change "No internet service" to "No"
l_reduce_nointernet = ["OnlineSecurity", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
                       "StreamingTV", "StreamingMovies"]

for cols in l_reduce_nointernet:
    telecom[cols] = telecom[cols].replace({"No internet service": "No"})

# 2. Change "No phone service" to "No"
telecom["MultipleLines"] = telecom["MultipleLines"].replace({"No phone service": "No"})

# 3. Shorten PaymentMethod details
telecom["PaymentMethod"] = telecom["PaymentMethod"].replace({"Electronic check": "ECheck",
                                                             "Mailed check": "MCheck",
                                                             "Bank transfer (automatic)": "AutoBankTransfer",
                                                             "Credit card (automatic)": "AutoCreditCard"})

# 4. Shorten InternetService details
telecom["InternetService"] = telecom["InternetService"].replace({"Fiber optic": "FiberOptic"})

# 5. Shorten Contract details
telecom["Contract"] = telecom["Contract"].replace({"Month-to-month": "Monthly", "One year": "1yr", "Two year": "2yr"})


# In[12]:


# Convert Tenure into categorical feature levels
def tenure_conversion(df):
    if df["tenure"] <= 12:
        return "t_0-1"
    if (df["tenure"] > 12) & (df["tenure"] <= 24):
        return "t_1-2"
    if (df["tenure"] > 24) & (df["tenure"] <= 48):
        return "t_2-4"
    if (df["tenure"] > 48) & (df["tenure"] <= 60):
        return "t_4-5"
    if df["tenure"] > 60:
        return "t_5plus"


telecom["tenuregroup"] = telecom.apply(lambda x: tenure_conversion(x), axis=1)

# In[13]:


telecom.head()

# In[14]:


# Check for Null values again to make sure all obersvations are okay
telecom.isnull().sum()

# ## Get Summary Statistics of the Data

# In[15]:


# Get summary stats for all continious variables
telecom.describe()

## Obs: 


# In[16]:


## Box-Plot to understand Outliers on continuous variables

# 1. Tenure
telecom["tenure"].plot(kind="box", color="r")

## Obs: No Outliers


# In[17]:


# 2. MonthlyCharges
telecom["MonthlyCharges"].plot(kind="box", color="g")

## Obs: No Outliers


# In[18]:


# 3. TotalCharges
telecom["TotalCharges"].plot(kind="box", color="b")

## Obs: No Outliers


# ## Check the summary & distribution of data

# ### Summary of Features & Response Variables

# In[20]:


sns.countplot(data=telecom, x="Churn")

count_no = len(telecom[telecom["Churn"] == "No"])
count_yes = len(telecom[telecom["Churn"] == "Yes"])

print(f"Percentage of Retained Customers: {count_no / (count_no + count_yes) :.2%}")
print(f"Percentage of Churn Customers: {count_yes / (count_no + count_yes) :.2%}")

# looks like an imbalaced data set - Need to handle this future steps during model building


# In[21]:


# sns.distplot(telecom["MonthlyCharges"], kde = True)

# split categorical, numerical features, excluding customerID and Target Variable

cat_columns = telecom.nunique()[telecom.nunique() < 6].keys().tolist()
cat_columns = [col for col in cat_columns if col not in ["customerID"] + ["Churn"]]

num_columns = [col for col in telecom.columns if col not in cat_columns + ["customerID"] + ["Churn"]]

# In[22]:


# Count Plot for all Categorical Features

fig, ax = plt.subplots(figsize=(5, 7))
plt.close("all")
sns.set(font_scale=1)

for col in cat_columns:
    plt.subplots(figsize=(10, 5))
    # plt.figure()
    plt.tight_layout()
    sns.countplot(x=col, data=telecom, order=telecom[col].value_counts().index).set_title(f"{col}: Data Summary")

# Will be plotted in Descending order of the value count by feature levels    


# In[23]:


# Histogram and Density Plots for continuous features

# 1. Histograms

for col in num_columns:
    # plt.subplots(figsize=(10,5))
    plt.figure(figsize=(10, 5))
    plt.tight_layout()
    sns.distplot(telecom[col], kde=False, hist=True).set_title(f"{col}: Data Summary")
    # plt.xlabel(col)
    plt.ylabel(f"Sum of {col}")

# In[24]:


# 2. Density Plots

for col in num_columns:
    plt.figure(figsize=(10, 5))
    plt.tight_layout()
    sns.distplot(telecom[col], kde=True, hist=True).set_title(f"{col}: Data Summary")
    plt.ylabel(f"Density of {col}")

# In[25]:


# Check Percentage of Retained vs Churned Customers from the Data Set
retain_count = len(telecom[telecom["Churn"] == "No"])
churn_count = len(telecom[telecom["Churn"] == "Yes"])

print(f"Percentage of Retained Customers is: {(retain_count / (retain_count + churn_count)) * 100}")
print(f"Percentage of Churned Customers is: {(churn_count / (retain_count + churn_count)) * 100}")

## Obs: Classes are imbalanced and the ratio of retain to churn is 73:26


# In[26]:


## Check Means with respect to the Response variable
telecom.groupby("Churn").mean()  # Ignore SeniorCitizen

# Obs:
# 1. Avg. tenure for churn customer is lower than retained customers by 20 months
# 2. Avg. Monthly Charges for churned customer is higher than retained customers by 10.2
# 3. Avg. Total Charges for Churned customer is lower than retained customers by 1018.12 
#      => ties in since avg tenure for churn is lower


# In[ ]:


# dx.count("Churn", split="InternetService", data=telecom, stacked=True, split_order=telecom.InternetService.unique().tolist())


# In[ ]:


# Calculate Categorical for other categorical variables 


# In[27]:


telecom.groupby("SeniorCitizen").mean()
# Obs: The averages between SeniorCitizen and Young Customers are almost similar, with an exception of TotalCharges
#       which could mean that SeniorCitizen opt of more # of services than younger customers. 


# In[28]:


telecom.groupby("Partner").mean()  # Ignore SeniorCitizen
# Obs: The averages are higher for Partners than non-Partners.  


# In[29]:


# Similar observations can be made for rest of the categorical means
for col in cat_columns:
    print(f"\033[1m {col} Avg: \033[0;0m\n {telecom.groupby(col).mean()} \n")

# ## Visualizations - Feature by each Response Variable class

# In[30]:


## Distribution of Frequency by each class of Response variable, for all Categorical Features

churn_customers = telecom[telecom["Churn"] == "Yes"]
nochurn_customers = telecom[telecom["Churn"] == "No"]

# In[31]:


# Use Matplotlib to plot the pie and histogram (with overlay) graphs
# Display frequency of each of the categorical feature by Churn level

sns.set(font_scale=2)
colors = ["lightskyblue", "lightcoral", "royalblue", "red", "green", "yellowgreen", "gold", ]


def category_plots(colname):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12))
    churn_dist = ax1.pie(churn_customers[colname].value_counts().values.tolist(), autopct="%1.1f%%", startangle=90,
                         labels=churn_customers[colname].unique().tolist(), textprops={'size': 'small'}, colors=colors)

    nochurn_dist = ax2.pie(nochurn_customers[colname].value_counts().values.tolist(), autopct="%1.1f%%", startangle=90,
                           labels=nochurn_customers[colname].unique().tolist(), textprops={'size': 'small'},
                           colors=colors)

    fig.suptitle(f"{colname} Churn vs Retention Distribution", verticalalignment="baseline")
    fig = plt.gcf()

    ax1.set_title("Churned Customers", fontdict={'verticalalignment': "top"}, loc="center", pad=-10)
    ax2.set_title("Retained Customers", fontdict={'verticalalignment': "top"}, loc="center", pad=-10)

    plt.tight_layout()
    plt.show()


# In[32]:


for col in cat_columns:
    category_plots(col)

# In[33]:


sns.set(font_scale=1)


def continuous_plots(colname):
    churn = churn_customers[colname]
    nochurn = nochurn_customers[colname]

    # Stack the data
    plt.figure(figsize=(10, 5))
    plt.hist([churn, nochurn], bins=10, stacked=False, density=True, label=["Churn", "Retained"], histtype="step")

    plt.title(f"Density of {colname} by Churn ", {'verticalalignment': 'baseline', 'horizontalalignment': "center"})
    plt.xlabel(f"{colname} by Churn")
    plt.ylabel("Density")
    plt.legend(loc='upper right')
    plt.tight_layout()

    plt.show()


for col in num_columns:
    continuous_plots(col)

# ## Bivariate Summary Statistics & Visualizations

# In[34]:


# Create a Scatter Matrix
scatter_matrix_df = telecom[["tenure", "MonthlyCharges", "TotalCharges", "Churn", "tenuregroup"]]
scatter_matrix_df

# In[35]:


sns.pairplot(scatter_matrix_df, hue="Churn", diag_kind="kde",
             plot_kws={"alpha": 0.2, "s": 80, "edgecolor": "k"},
             height=5)

plt.tight_layout()

## Obs: There is a strong relationship between:
##       1. MonthlyCharges and TotalCharges
##       2. tenure and TotalCharges
##      The KDEs suggest the density by Churn for each of the continuous variables, which confirms above plots


# In[36]:


# Statistics by tenuregroup
churned_tenuregroup = churn_customers.tenuregroup.value_counts().reset_index()
churned_tenuregroup.columns = ["churn_tenuregroup", "frequency"]
# churned_tenuregroup["pct"] = churned_tenuregroup["frequency"]/(churned_tenuregroup["frequency"].sum())*100

churned_tenuregroup

# In[37]:


retained_tenuregroup = nochurn_customers.tenuregroup.value_counts().reset_index()
retained_tenuregroup.columns = ["retain_tenuregroup", "frequency"]
# retained_tenuregroup["pct"] = retained_tenuregroup["frequency"]/(retained_tenuregroup["frequency"].sum())*100
retained_tenuregroup

# In[38]:


## Plot Churn by Tenure Group

fig, ax1 = plt.subplots(figsize=(10, 6))
sns.set_color_codes('pastel')
sns.barplot(x='retain_tenuregroup', y='frequency', data=retained_tenuregroup,
            label='Retained by Tenure Group', color='r', edgecolor='w')
sns.set_color_codes('muted')
sns.barplot(x='churn_tenuregroup', y='frequency', data=churned_tenuregroup,
            label='Churned by Tenure Group', color='r', edgecolor='w')
ax1.legend(ncol=2, loc='upper right')
plt.xlabel("Tenure Group")
plt.ylabel("Frequency")
sns.despine(left=True, bottom=True)
plt.show()

## Obs: The newest customers have the Highest Churn by frequency.


# In[39]:


telecom.corr()

# In[40]:


telecom_corr = telecom.corr(method="pearson")

# In[41]:


corr_features = telecom.columns.tolist()
corr_array = np.array(telecom_corr)

# Plot Heatmap with annotations using seaborn

sns.set()
f, ax = plt.subplots(figsize=(20, 10))
sns.heatmap(telecom_corr, annot=True, linewidth=1, ax=ax)

# In[42]:


corr_array = np.array(telecom_corr)
corr_array

## Obs: Confirms the observations made in Scatter Plot matrix for continuous features


# ## Data Pre-processing for further analysis

# ##### Since the continuous variables have a wide range and different units of measure, it needs to be transformed to a similar Scale for better comparison
# ##### Can choose between StandardScaler or MinMaxScaler, 
# #####       - Outliers have more impact on StandardScaler than MinMaxScaler, but since there are no outliers in the continuous variables Std. Scaler                     is used
# ##### But Logistic regression fits a model that use a weighted sum of input variables are affected by difference in scale
# https://machinelearningmastery.com/standardscaler-and-minmaxscaler-transforms-in-python/

# #### 1.1. Perform Label Encoding for binary categorical features
# #### 1.2. Perform One Hot Encoding for non-binary categorical features
# #### 1.3. Perform Data Standardization by Scaling continuous features

# In[43]:


telecom.head()

# In[44]:


original_telecom = telecom.copy(deep=True)  # Save original dataframe for future references

# In[45]:


# List Label and One Hot Encoding features

le_columns = []  # features with exactly 2 values
oh_columns = []  # features with more than 2 values and non-ordinal values
id_col = ["customerID"]
target_col = ["Churn"]

le_columns = telecom.nunique()[telecom.nunique() == 2].keys().tolist()
oh_columns = [x for x in cat_columns if x not in le_columns]

# In[46]:


# Confirm feature levels for the features
telecom[le_columns].nunique()

# In[47]:


telecom[oh_columns].nunique()

# ### Perform Transformations to important features

# In[48]:


# Perform transformations

le_encode = LabelEncoder()

for l in le_columns:
    telecom[l] = le_encode.fit_transform(telecom[l])

# get_dummies for one hot encoding

telecom = pd.get_dummies(data=telecom, columns=oh_columns)

# Scaling Numerical Columns
std = StandardScaler()
scaled = std.fit_transform(telecom[num_columns])
scaled = pd.DataFrame(scaled, columns=num_columns)

for col in scaled.columns:
    plt.figure(figsize=(10, 5))
    plt.tight_layout()
    sns.distplot(scaled[col], kde=True, hist=True).set_title(f"Scaled {col}: Density plot")
    plt.ylabel(f"Density of {col}")

## Obs: The density distribution for the scaled dataset is similar to the original continuous features
## Running this cell again will result in errors, since the telecom df is already transformed


# In[49]:


# Compare original and transformed dataset for an example observation
telecom[telecom["customerID"] == "7590-VHVEG"]

# In[50]:


original_telecom[original_telecom["customerID"] == "7590-VHVEG"]

# In[51]:


# Merge scaled transformed continuous features to the telecom data set & drop original continuous features
# 1. Drop old features
telecom = telecom.drop(columns=num_columns, axis=1)
# 2. Add scaled features
telecom = telecom.merge(scaled, left_index=True, right_index=True, how="left")

telecom.head()

# ### Create a correlation matrix for transformed data

# In[52]:


telecom_corr = telecom.corr(method="pearson")

# In[53]:


# Plot Heatmap with annotations using seaborn

sns.set(font_scale=1.2)
plt.figure(figsize=(30, 20))
sns.heatmap(telecom_corr, annot=True, linewidth=1, center=0, cmap="RdYlGn")
plt.title("Correlation Matrix for Transformed Dataset", fontsize=25)
plt.tight_layout()

# # Create Models using Sci-kit Learn for Advanced Analysis & Prediction

# ## Splitting Data into "Training" and "Test" sub-sets for Model Performance Assessments
# ### 1. Train-Test Split

# In[54]:


# Prepare data for split
telecom_y = []
telecom_y = telecom["Churn"].ravel()
telecom_x = telecom.copy(deep=True)
telecom_x = telecom.drop(columns=["Churn", "customerID"], axis=1)  # Customer ID is not relevant for current analysis

# Split Data
X_train, X_test, y_train, y_test = train_test_split(telecom_x, telecom_y, test_size=0.25, random_state=0, shuffle=True)

del telecom_y, telecom_x  # release memory since these dfs are no longer required

# In[55]:


np.nanmean(y_train), np.nanmean(y_test)


# Similar means = the split is not imbalanced


# In[56]:


# Define required functions for repeated executions

# 1. Get Model Scores

def model_scores(model_name, model, Xtest, ytest, predictions, idx):
    model_acc = model.score(Xtest, ytest)
    model_recall = recall_score(ytest, np.array(predictions))
    model_precision = precision_score(ytest, np.array(predictions))
    model_auc = roc_auc_score(ytest, np.array(predictions))
    model_f1 = f1_score(ytest, np.array(predictions))
    model_classreport = classification_report(ytest, np.array(predictions))
    model_cm = confusion_matrix(ytest, np.array(predictions))

    print("\033[1m Classification Report: \033[0;0m\n", model_classreport)
    print("\033[1m Accuracy Score: \033[0;0m\n", model_acc)
    print(f"\033[1m Precision: \033[0;0m{model_precision}\033[1m  & Recall Scores: \033[0;0m{model_recall}\n")
    print("\033[1m F1 Score: \033[0;0m\n", model_f1)
    print("\033[1m AUC: \033[0;0m\n", model_auc)

    comp_cols = ["ModelName", "AccuracyScore", "RecallScore", "PrecisionScore", "F1Score", "AreaUnderCurve"]

    df = []
    df = (pd.DataFrame({"ModelName": model_name,
                        "AccuracyScore": model_acc,
                        "RecallScore": model_recall,
                        "PrecisionScore": model_precision,
                        "F1Score": model_f1,
                        "AreaUnderCurve": model_auc
                        },
                       index=[idx])
          )

    plot_confusionmatrix(model_name, model_cm)

    fpr, tpr, thresholds = roc_curve(ytest, model.predict_proba(Xtest)[:, 1])
    plot_roc_auc(model_name, model_auc, fpr, tpr)

    return df


# 2. Plot Confusion Matrix for the models
def plot_confusionmatrix(algorithm, conf_matrix):
    classes = ["Retained Customers", "Churn Customers"]
    plt.figure(figsize=(7, 7))
    plt.imshow(conf_matrix, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(f"{algorithm}: Confusion Matrix")
    plt.colorbar(aspect=4)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, size=14)
    plt.yticks(tick_marks, classes, size=14)
    fmt = "d"
    thresh = conf_matrix.max() / 2

    # Labeling the plot
    for i, j in itertools.product(range(conf_matrix.shape[0]), range(conf_matrix.shape[1])):
        plt.text(j, i, format(conf_matrix[i, j], fmt), fontsize=15,
                 horizontalalignment="center",
                 color="white" if conf_matrix[i, j] > thresh else "black")
    plt.grid(None)
    plt.tight_layout()
    plt.ylabel('True label', size=15)
    plt.xlabel('Predicted label', size=15)


# 3. Plot ROC-AUC Curve
def plot_roc_auc(algorithm, roc_auc, fpr, tpr):
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, label="Model (area = %0.2f)" % roc_auc)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{algorithm} Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.show()


# ### 1. Create Dummy Baseline Model

# In[57]:


dummy_model = DummyClassifier(strategy="most_frequent", random_state=0)

# Train Dummy Baseline
dummy_model.fit(X_train, y_train)

# Generate Prediction
base_predictions = dummy_model.predict(X_test)

# Display Scores & Confusion Matrix
model_comparison = pd.DataFrame(model_scores("Dummy Baseline", dummy_model, X_test, y_test, base_predictions, 0),
                                columns=["ModelName",
                                         "AccuracyScore",
                                         "RecallScore",
                                         "PrecisionScore",
                                         "F1Score",
                                         "AreaUnderCurve"]
                                )
model_comparison.drop_duplicates()  # In case this cell is rerun

# In[ ]:


# ### Over Sampling the train sets using SMOTE (Synthetic Minority Oversampling Technique)
# ###   - Up sample "Churned Customers" in the training dataset

# In[58]:


x_cols = X_train.columns.tolist()
y_cols = ["Churn"]

# In[59]:


# oversampling minority class using smote

over_sample = SMOTE(random_state=0)

os_data_X, os_data_y = over_sample.fit_sample(X_train, y_train)

os_data_X = pd.DataFrame(data=os_data_X, columns=x_cols)
os_data_y = pd.DataFrame(data=os_data_y, columns=y_cols)

# Check the details after SMOTE runs
retain_count = len(os_data_y[os_data_y["Churn"] == 0])
churn_count = len(os_data_y[os_data_y["Churn"] == 1])

print(f"Total count of Oversampled data: {len(os_data_X)}")
print(f"Number of Retained Customer count in Oversample data: {retain_count}")
print(f"Number of Churned Customer count in Oversample data: {churn_count}")
print(f"Proportion of Retained Customer data: {retain_count / len(os_data_X)}")
print(f"Proportion of Churned Customer data: {churn_count / len(os_data_X)}")

# In[ ]:


## Obs: Imbalanced classes from the target variable has been addressed and these data sets are to be used for further modelling


# ### Dimension Reduction - Remove features which would not contribute to the model's performance
# ### Use RFE (Recursive Feature Elimination)
# ####   - Repeatedly construct a model and choose either the best or worst performing feature

# In[60]:


# Create RFE Model
# use x_cols and y_cols from above for column information

log_regr = LogisticRegression(solver="liblinear")

rfe = RFE(log_regr, 15)  # Select top 15 features from 31 -> try and reduce the # of features by 50% approx.
rfe.fit(os_data_X, os_data_y.values.ravel())
rfe_ranking = pd.DataFrame({"Features": X_train.columns.tolist(),
                            "rfe_support": rfe.support_,
                            "rfe_ranking": rfe.ranking_})
rfe_ranking.sort_values(by="rfe_ranking", inplace=True)
rfe_ranking

## Obs: Based on the RFE model, we know that out of 31 features, top 15 features are listed below. 
## These can be further reduced based on the p-values for each 


# In[61]:


# Output top 15 columns from RFE process
rfe_cols = rfe_ranking[rfe_ranking["rfe_support"] == True]["Features"].tolist()

# Build new training dataframes with the above selected columns
X = os_data_X[rfe_cols]
y = os_data_y

# In[ ]:


# ### Implementing Stats Model to get the p-values of each of the selected columns

# In[62]:


stats_logitmodel = sm.Logit(y, X)
result = stats_logitmodel.fit()
print("Summary: \n", result.summary2())

# In[63]:


# Remove columns with p-value more than significance value of 0.05
remove_cols = ["tenuregroup_t_1-2", "Contract_2yr", "tenuregroup_t_2-4", "PhoneService"]
final_cols = [i for i in X.columns if i not in remove_cols]

X = os_data_X[final_cols]
y = os_data_y["Churn"]

print(f"Total number of Observations in Oversampled X: {len(X)} & Oversampled y: {len(y)}")

# #### Make sure the p-values are correct for remaining features

# In[64]:


stats_logitmodel = sm.Logit(y, X)
result = stats_logitmodel.fit()
print("Summary: \n", result.summary2())

# # Build Logistic Regression Model with Default Parameters

# In[65]:


# Split the oversampled "balanced" training data set again, since the old training data was imbalanced.
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.25, random_state=0)

# Create Logistic Regression model with Default parameters
lr_model = LogisticRegression(solver="liblinear")

# Train the model
lr_model.fit(X_train, y_train)

# Predict the results from testing set & calculate model performance
lr_predictions = lr_model.predict(X_test)

model_comparison = model_comparison.append(model_scores(model_name="Logistic Regression Model - DefaultParams",
                                                        model=lr_model,
                                                        Xtest=X_test,
                                                        ytest=y_test,
                                                        predictions=lr_predictions,
                                                        idx=1),
                                           ignore_index=False)

model_comparison = model_comparison.drop_duplicates()  # Incase this cell is run again.
model_comparison

# In[ ]:


## Obs: The Scores are Far better than the dummy baseline model 


# ### Hyperparameter tuning for Logistic Regression Model

# In[66]:


class_weight = [{1: 0.5, 0: 0.5}, {1: 0.4, 0: 0.6}, {1: 0.6, 0: 0.4}, {1: 0.65, 0: 0.3}, {1: 0.7, 0: 0.3}]
parameters = {"C": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
              "penalty": ["l1", "l2"],
              "class_weight": class_weight,
              "solver": ['liblinear', 'saga']}

cv_model = GridSearchCV(lr_model,
                        param_grid=parameters,
                        cv=3,
                        scoring='roc_auc',
                        verbose=1,
                        n_jobs=-1)  # perform 3 fold validation

cv_result = cv_model.fit(X_train, y_train)

print(f"Best Parameters: {cv_model.best_params_} \n")
print(f"Best Score after HPO: {cv_model.best_score_} \n")
print(f"Best Model Accuracy Score after Hyperparameter Optimization: {cv_model.score(X_test, y_test)}")

# In[67]:


cv_predictions = cv_model.predict(X_test)
model_comparison = model_comparison.append(model_scores(model_name="Logistic Regression Hyperparameter Optimized",
                                                        model=cv_model,
                                                        Xtest=X_test,
                                                        ytest=y_test,
                                                        predictions=cv_predictions,
                                                        idx=2),
                                           ignore_index=False)

model_comparison.drop_duplicates()
model_comparison

# # Build Random Forest Model with Default Parameters

# In[68]:


from sklearn.ensemble import RandomForestClassifier

# Create the model with 100 trees
model = RandomForestClassifier(n_estimators=100,
                               bootstrap=True,
                               max_features='sqrt')
# Fit on training data
model.fit(X, y)

# In[91]:


# Prepare data for split
telecom_y = []
telecom_y = os_data_y["Churn"]
telecom_y = telecom_y.ravel()

telecom_x = os_data_X[final_cols]  # work with only important features

# Split Data
X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(telecom_x,
                                                                telecom_y,
                                                                test_size=0.25,
                                                                random_state=0,
                                                                shuffle=True)

# In[93]:


X_train_rf

# In[94]:


# Create the model with 100 trees
rf_model = RandomForestClassifier(n_estimators=100,
                                  bootstrap=True,
                                  max_features='sqrt')
# Fit on training data
rf_model.fit(X_train_rf, y_train_rf)

# In[95]:


rf_pred = rf_model.predict(X_test_rf)

# In[96]:


rf_model.feature_importances_

# In[ ]:


# y_test_e = le_encode.fit_transform(y_test_rf)
# rf_pred_e = le_encode.fit_transform(rf_pred)


# In[97]:


# model_comparison = model_comparison.drop(index=3, axis=0)
model_comparison = model_comparison.append(model_scores(model_name="Random Forest Default",
                                                        model=rf_model,
                                                        Xtest=X_test_rf,
                                                        ytest=y_test_rf,
                                                        predictions=rf_pred,
                                                        idx=3),
                                           ignore_index=False)
model_comparison.drop_duplicates()
model_comparison

# ### Hyperparameter tuning for Random Forest Model

# In[98]:


param_grid = {
    'bootstrap': [True],
    'max_depth': [80, 90, 100, 110],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300, 1000]
}

# Instantiate the grid search model
cv_rf_model = GridSearchCV(estimator=rf_model, param_grid=param_grid,
                           cv=3, n_jobs=-1, verbose=2)  # 3-fold validation

# In[99]:


# Train the best model; Takes around 10 mins! - Get some coffee :)
cv_rf_model.fit(X_train_rf, y_train_rf)
print("Best Parameters:", cv_rf_model.best_params_)

# In[100]:


best_rf_model = cv_rf_model.best_estimator_
cv_rf_pred = best_rf_model.predict(X_test_rf)

model_comparison = model_comparison.append(model_scores(model_name="Random Forest HyperTuned",
                                                        model=best_rf_model,
                                                        Xtest=X_test_rf,
                                                        ytest=y_test_rf,
                                                        predictions=cv_rf_pred,
                                                        idx=4),
                                           ignore_index=False)
model_comparison.drop_duplicates()
model_comparison

# In[ ]:


# List Feature Importances


# In[101]:


# Import tools needed for visualization
from sklearn import tree
from sklearn.tree import export_graphviz
from graphviz import Source
from IPython.display import display

# Pull out one tree from the forest
est_tree = best_rf_model.estimators_[5]

graph = Source(tree.export_graphviz(est_tree, out_file=None,
                                    rounded=True, proportion=False,
                                    feature_names=X_train_rf.columns.tolist(),
                                    precision=2,
                                    class_names=["Not churn", "Churn"],
                                    filled=True
                                    )
               )

display(graph)

# In[110]:


model_comparison.sort_values(by=["RecallScore", "AccuracyScore", "F1Score"], ascending=False)
# Recall scores are generally more focused than Precision Scores, to reduce false negatives on customers churning.


# # Final Conclusions on Models:

# In[155]:


best_df = pd.DataFrame(columns=["PerfParam", "BestParamVal", "Model"])

best_recall_score = model_comparison["RecallScore"].max()
best_recall_score_model = model_comparison[model_comparison["RecallScore"] == best_recall_score]["ModelName"].tolist()
best_df = best_df.append([{"PerfParam": "RecallScore",
                           "BestParamVal": best_recall_score,
                           "Model": best_recall_score_model[0]}],
                         ignore_index=True)

best_f1_score = model_comparison["F1Score"].max()
best_f1_score_model = model_comparison[model_comparison["F1Score"] == best_f1_score]["ModelName"].tolist()
best_df = best_df.append([{"PerfParam": "F1Score",
                           "BestParamVal": best_f1_score,
                           "Model": best_f1_score_model[0]}],
                         ignore_index=True)

best_accuracy_score = model_comparison["AccuracyScore"].max()
best_accuracy_score_model = model_comparison[model_comparison["AccuracyScore"] == best_accuracy_score][
    "ModelName"].tolist()
best_df = best_df.append([{"PerfParam": "AccuracyScore",
                           "BestParamVal": best_accuracy_score,
                           "Model": best_accuracy_score_model[0]}],
                         ignore_index=True)

best_auc_score = model_comparison["AreaUnderCurve"].max()
best_auc_score_model = model_comparison[model_comparison["AreaUnderCurve"] == best_auc_score]["ModelName"].tolist()
best_df = best_df.append([{"PerfParam": "AreaUnderCurve",
                           "BestParamVal": best_auc_score,
                           "Model": best_auc_score_model[0]}],
                         ignore_index=True)

best_precision_score = model_comparison["PrecisionScore"].max()
best_precision_score_model = model_comparison[model_comparison["PrecisionScore"] == best_precision_score][
    "ModelName"].tolist()
best_df = best_df.append([{"PerfParam": "PrecisionScore",
                           "BestParamVal": best_precision_score,
                           "Model": best_precision_score_model[0]}],
                         ignore_index=True)

# best_df

print(
    f"Best Recall score is for \033[1m {best_recall_score_model[0]}\033[0;0m with \033[1m{best_recall_score :.2%}\033[0;0m")
print(f"Best F1 score is for \033[1m{best_f1_score_model[0]}\033[0;0m with \033[1m{best_f1_score :.2%}\033[0;0m")
print(
    f"Best accuracy score is for \033[1m{best_accuracy_score_model[0]}\033[0;0m with \033[1m{best_accuracy_score :.2%}\033[0;0m")
print(f"Best AUC score is for \033[1m {best_auc_score_model[0]}\033[0;0m with \033[1m{best_auc_score :.2%}\033[0;0m")
print(
    f"Best Precision score is for \033[1m {best_precision_score_model[0]}\033[0;0m with \033[1m{best_precision_score :.2%}\033[0;0m")

# In[ ]:


# In[ ]:
