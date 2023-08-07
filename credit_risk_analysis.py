

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns
from sklearn import model_selection
from sklearn.metrics import auc, accuracy_score, confusion_matrix, roc_auc_score, classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("credit.csv.zip")
df.head()

df.describe().T

df.dropna(how='any',axis=0)

df.duplicated().sum()

df.drop_duplicates()

df.isnull().sum()

df.dropna()

df['person_emp_length'].fillna(df['person_emp_length'].mode()[0], inplace=True)
df['loan_int_rate'].fillna(df['loan_int_rate'].median(), inplace=True)

sns.countplot(df['loan_status'])

plt.figure(figsize = (10, 10))
df['person_age'].plot.hist(bins=50)

df['person_income'].plot.box()

np.log(df['person_income']).plot.hist()
df['log_income'] = np.log(df['person_income'])

df['person_home_ownership'].value_counts().plot.pie()

df['loan_intent'].value_counts().plot.pie()

sns.boxplot(x = 'loan_status', y='loan_int_rate', data = df)

sns.boxplot(x = 'loan_status', y='log_income', data = df)

cat_cols = [col for col in df.columns if df[col].dtypes == 'O']

for col in cat_cols:
    print(df[col].value_counts(), "\n\n")

num_cols = pd.DataFrame(df[df.select_dtypes(include=['float', 'int']).columns])
# print the numerical variebles
num_cols.columns

# get the categorical variables
cat_cols = pd.DataFrame(df[df.select_dtypes(include=['object']).columns])
cat_cols.columns

encoded_cat_cols = pd.get_dummies(cat_cols)
cat_cols_corr = pd.concat([encoded_cat_cols, df['loan_status']], axis=1)
corr = cat_cols_corr.corr().sort_values('loan_status', axis=1, ascending=False)
corr = corr.sort_values('loan_status', axis=0, ascending=True)
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask, k=1)] = True

with sns.axes_style("white"):
    f, ax = plt.subplots(figsize=(16, 10))
    ax = sns.heatmap(corr, mask=mask, vmin=corr.loan_status.min(),
                     vmax=corr.drop(['loan_status'], axis=0).loan_status.max(),
                     square=True, annot=True, fmt='.2f',
                     center=0, cmap='RdBu',annot_kws={"size": 10})

credit_df = pd.concat([num_cols, encoded_cat_cols], axis=1)

label = credit_df['loan_status'] # labels
features = credit_df.drop('loan_status',axis=1) # features
x_train, x_test, y_train, y_test =train_test_split(features, label,random_state=42, test_size=.30)
print('The train dataset has {} data\nThe test dataset has {} data'.
      format(x_train.shape[0], x_test.shape[0]))

def model_assess(model, name='Default'):
    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    preds_proba = model.predict_proba(x_test)
    print(name, '\n',classification_report(y_test, model.predict(x_test)))

#Logistic Regression
lg = LogisticRegression(random_state=42)
model_assess(lg, 'Logistic Regression')
