import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Loading the dataset
df = pd.read_csv('admission_predict (1).csv')

# Exploring the dataset
print("Dataset shape:", df.shape)
print(df.head())

print("\nColumns in dataset:")
print(df.columns)

print("\nDataset info:")
print(df.info())

print("\nBasic statistics:")
print(df.describe().T)

print("\nData types:")
print(df.dtypes)

print("\nChecking for null values:")
print(df.isnull().any())

# Renaming the columns with appropriate names
df = df.rename(columns={
    'GRE Score': 'GRE',
    'TOEFL Score': 'TOEFL',
    'LOR ': 'LOR',
    'Chance of Admit ': 'Probability'
})

print("\nColumns after renaming:")
print(df.columns)
print(df.head())

# Data Visualization
plt.hist(df['GRE'], rwidth=0.7)
plt.title("Distribution of GRE Scores")
plt.xlabel('GRE Scores')
plt.ylabel('Count')
plt.show()

plt.hist(df['TOEFL'], rwidth=0.7)
plt.title('Distribution of TOEFL Scores')
plt.xlabel('TOEFL Scores')
plt.ylabel('Count')
plt.show()

plt.hist(df['University Rating'], rwidth=0.7)
plt.title('Distribution of University Rating')
plt.xlabel('University Rating')
plt.ylabel('Count')
plt.show()

plt.hist(df['SOP'], rwidth=0.7)
plt.title('Distribution of SOP')
plt.xlabel('SOP Rating')
plt.ylabel('Count')
plt.show()

plt.hist(df['LOR'], rwidth=0.7)
plt.title('Distribution of LOR Rating')
plt.xlabel('LOR Rating')
plt.ylabel('Count')
plt.show()

plt.hist(df['CGPA'], rwidth=0.7)
plt.title('Distribution of CGPA')
plt.xlabel('CGPA')
plt.ylabel('Count')
plt.show()

plt.hist(df['Research'], rwidth=0.7)
plt.title('Distribution of Research Papers')
plt.xlabel('Research')
plt.ylabel('Count')
plt.show()

# Data Cleaning
# Removing the Serial No. column
df.drop('Serial No.', axis='columns', inplace=True)
print("\nData after dropping 'Serial No.':")
print(df.head())

# Replacing 0 values in selected columns with NaN
df_copy = df.copy(deep=True)
cols_to_replace = ['GRE', 'TOEFL', 'University Rating', 'SOP', 'LOR', 'CGPA']
df_copy[cols_to_replace] = df_copy[cols_to_replace].replace(0, np.NaN)
print("\nNull values after replacing 0 with NaN:")
print(df_copy.isnull().sum())

# Model Building
# Splitting the dataset into features and label
X = df_copy.drop('Probability', axis='columns')
y = df_copy['Probability']

# Creating a function to calculate best model for this problem
def find_best_model(X, y):
    models = {
        'linear_regression': {
            'model': make_pipeline(StandardScaler(), LinearRegression()),
            'parameters': {
                'linearregression__fit_intercept': [True, False]
            }
        },
        'lasso': {
            'model': make_pipeline(StandardScaler(), Lasso()),
            'parameters': {
                'lasso__alpha': [1, 2],
                'lasso__selection': ['random', 'cyclic']
            }
        },
        'svr': {
            'model': make_pipeline(StandardScaler(), SVR()),
            'parameters': {
                'svr__gamma': ['auto', 'scale']
            }
        },
        'decision_tree': {
            'model': DecisionTreeRegressor(),
            'parameters': {
                'criterion': ['squared_error', 'friedman_sse'],
                'splitter': ['best', 'random']
            }
        },
        'random_forest': {
            'model': RandomForestRegressor(),
            'parameters': {
                'n_estimators': [5, 10, 15, 20]
            }
        },
        'knn': {
            'model': make_pipeline(StandardScaler(), KNeighborsRegressor()),
            'parameters': {
                'kneighborsregressor__n_neighbors': [2, 5, 10, 20]
            }
        }
    }

    scores = []
    for model_name, model_params in models.items():
        gs = GridSearchCV(model_params['model'], model_params['parameters'], cv=5, return_train_score=False)
        gs.fit(X, y)
        scores.append({
            'model': model_name,
            'best_parameters': gs.best_params_,
            'score': gs.best_score_
        })

    return pd.DataFrame(scores, columns=['model', 'best_parameters', 'score'])

print("\nFinding best model using GridSearchCV...")
best_models = find_best_model(X, y)
print(best_models)

# Since Linear Regression has the highest accuracy, use it with cross_val_score
print("\nEvaluating Linear Regression with cross_val_score...")
lr_pipeline = make_pipeline(StandardScaler(), LinearRegression(fit_intercept=True))
scores = cross_val_score(lr_pipeline, X, y, cv=5)
print('Highest Accuracy : {:.2f}%'.format(np.mean(scores)*100))

# Splitting the dataset into train and test samples
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=5)
print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")

# Creating Linear Regression Model and training
model = make_pipeline(StandardScaler(), LinearRegression(fit_intercept=True))
model.fit(X_train, y_train)

# Model evaluation on test set
test_score = model.score(X_test, y_test)
print(f"Test set score: {test_score:.4f}")

# Predicting the values using our trained model
# Prediction 1
input1 = [[337, 118, 4, 4.5, 4.5, 9.65, 0]]
pred1 = model.predict(input1)[0]
print('Chance of getting into UCLA is {:.3f}%'.format(pred1 * 100))

# Prediction 2
input2 = [[320, 113, 2, 2.0, 2.5, 8.64, 1]]
pred2 = model.predict(input2)[0]
print('Chance of getting into UCLA is {:.3f}%'.format(pred2 * 100))
