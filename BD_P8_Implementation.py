
## BD 2023 Project - Code
### Rain Prediction 
# Atudore Darius, Bouruc Liviu, Francu Richard, Tender Laura (Data Science - 411)

### Dataset Analysis and Preprocessing


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import Isomap, LocallyLinearEmbedding

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("/content/Weather Training Data.csv", delimiter = ',')
data.head()

print(f'Our proccesed data have {len(data.columns)} columns.')
print(f'Columns: {data.columns}')

data.RainTomorrow.value_counts()
# it rains tomorrow in <25% cases

print(data.RainToday.unique())

"""Empty cells need to be replaced with 0 and RainToday should have binary values."""

CAT_COLUMNS = ['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm']
le = LabelEncoder()
data[CAT_COLUMNS] = data[CAT_COLUMNS].apply(le.fit_transform)

data.fillna(0, inplace=True)
data.RainToday = data.RainToday.map({'Yes': 1, 'No': 0, 0: 0})

data.head()

data.to_csv('processed_data.csv', index=False)

y = data['RainTomorrow']
X = data.drop(columns=['row ID', 'RainTomorrow'])
alpha = 0.15 # percentage of dataset used
beta = 0.8 # percentage of training data

all_data, _, all_labels, _ = train_test_split(X, y, train_size = alpha, random_state = 28) 
X_train, X_test, y_train, y_test = train_test_split(all_data, all_labels, train_size = beta, random_state = 42)

all_labels.value_counts()

all_data.to_csv('reduced_dataset.csv', index=False)

"""### Random Forest

"""

# Baseline

Bag = RandomForestClassifier(n_estimators = 250, max_features = 21)
Bag = Bag.fit(X_train, y_train)

importances = Bag.feature_importances_
forest_importances = pd.Series(importances, index = X.columns)

# We'll use standard deviation of the values obtained above
std = np.std([tree.feature_importances_ for tree in Bag.estimators_], axis = 0)

# Plotting the graph
fig, ax = plt.subplots()
forest_importances.plot.bar(yerr = std, ax = ax)
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()

sorted_importances = pd.Series(forest_importances, index = X_train.columns).sort_values(ascending=False)
print(sorted_importances)

rf_test_pred = Bag.predict(X_test) 
cm = confusion_matrix(y_test, rf_test_pred, labels = Bag.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels = Bag.classes_)
disp.plot()

def compute_scores(model_name, predicted_labels, real_labels):
  conclusion = pd.DataFrame({'Model': [model_name]})

  m_accuracy = accuracy_score(real_labels, predicted_labels)
  m_precision = precision_score(real_labels, predicted_labels)
  m_recall = recall_score(real_labels, predicted_labels)
  m_f1_score = f1_score(real_labels, predicted_labels)

  conclusion['accuracy'] = [m_accuracy]
  conclusion['precision'] = [m_precision]
  conclusion['recall'] = [m_recall]
  conclusion['f1'] = [m_f1_score]

  return conclusion

compute_scores("Random Forest baseline", rf_test_pred, y_test)

# PCA
n_components_tries = [3, 5, 10, 15]
for n_components in n_components_tries:
  pca = PCA(n_components = n_components, random_state = 42)
  pca.fit(X_train)
  X_train_pca = pca.transform(X_train)
  X_test_pca = pca.transform(X_test)
  Bag = RandomForestClassifier(n_estimators = 250, max_features = n_components)
  Bag = Bag.fit(X_train_pca, y_train)
  y_test_pca = Bag.predict(X_test_pca) 
  m_f1_score = f1_score(y_test, y_test_pca)
  print(f'PCA with n_components {n_components} has f1 score {m_f1_score}')

n_components_tries = [3, 5, 10, 15]
for n_components in n_components_tries:
  pca = PCA(n_components = n_components, random_state = 42)
  pca.fit(X_train)
  X_train_pca = pca.transform(X_train)
  X_test_pca = pca.transform(X_test)
  Bag = RandomForestClassifier(n_estimators = 250, max_features = n_components)
  Bag = Bag.fit(X_train_pca, y_train)
  y_test_pca = Bag.predict(X_test_pca) 
  m_f1_score = f1_score(y_test, y_test_pca)
  print(f'PCA with n_components {n_components} has f1 score {m_f1_score}')

pca = PCA(n_components = 15, svd_solver = "full", random_state = 42)
pca.fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
Bag = RandomForestClassifier(n_estimators = 250, max_features = n_components)
Bag = Bag.fit(X_train_pca, y_train)
y_test_pca = Bag.predict(X_test_pca)

compute_scores("PCA", y_test_pca, y_test)

cm = confusion_matrix(y_test, rf_test_pred, labels = Bag.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels = Bag.classes_)
disp.plot()

# Isomap
n_components_tries = [3, 5, 10, 15]
for n_components in n_components_tries:
  isomap = Isomap(n_components = n_components)
  isomap.fit(X_train)
  X_train_isomap = isomap.transform(X_train)
  X_test_isomap = isomap.transform(X_test)
  Bag = RandomForestClassifier(n_estimators = 250, max_features = n_components)
  Bag = Bag.fit(X_train_isomap, y_train)
  y_test_isomap = Bag.predict(X_test_isomap) 
  m_f1_score = f1_score(y_test, y_test_isomap)
  print(f'Isomap with n_components {n_components} has f1 score {m_f1_score}')

isomap = Isomap(n_components = 15)
isomap.fit(X_train)
X_train_isomap = isomap.transform(X_train)
X_test_isomap = isomap.transform(X_test)
Bag = RandomForestClassifier(n_estimators = 250, max_features = n_components)
Bag = Bag.fit(X_train_isomap, y_train)
y_test_isomap = Bag.predict(X_test_isomap)

compute_scores("Isomap", y_test_isomap, y_test)

cm = confusion_matrix(y_test, y_test_isomap, labels = Bag.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels = Bag.classes_)
disp.plot()

# LLE
n_components_tries = [3, 5, 10, 15]

for n_components in n_components_tries:
    lle = LocallyLinearEmbedding(n_components = n_components)
    lle.fit(X_train)

    X_train_lle = lle.transform(X_train)
    X_test_lle = lle.transform(X_test)

    Bag = RandomForestClassifier(n_estimators = 100, max_features = n_components)
    Bag = Bag.fit(X_train_lle, y_train)

    y_test_lle = Bag.predict(X_test_lle) 
    m_f1_score = f1_score(y_test, y_test_lle)

    print(f'Locally Linear Embedding with n_components {n_components} has f1 score {m_f1_score}')

lle = LocallyLinearEmbedding(n_components = 15)
lle.fit(X_train)

X_train_lle = lle.transform(X_train)
X_test_lle = lle.transform(X_test)

Bag = RandomForestClassifier(n_estimators = 100, max_features = 15)
Bag = Bag.fit(X_train_lle, y_train)

y_test_lle = Bag.predict(X_test_lle) 
compute_scores("Locally Linear Embedding", y_test_lle, y_test)

cm = confusion_matrix(y_test, y_test_lle, labels = Bag.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels = Bag.classes_)
disp.plot()

# Kernel PCA
n_components_tries = [3, 5, 10, 15]
kernels = ['linear', 'poly', 'rbf', 'sigmoid', 'cosine']

for n_components in n_components_tries:
    for kernel in kernels:
      kernel_PCA = KernelPCA(n_components = n_components, kernel = kernel)
      kernel_PCA.fit(X_train)

      X_train_pca = kernel_PCA.transform(X_train)
      X_test_pca = kernel_PCA.transform(X_test)

      Bag = RandomForestClassifier(n_estimators = 100, max_features = n_components)
      Bag = Bag.fit(X_train_pca, y_train)

      y_test_pca = Bag.predict(X_test_pca) 
      m_f1_score = f1_score(y_test, y_test_pca)

      print(f'PCA with kernel {kernel} and n_components {n_components} has f1 score {m_f1_score}')

kernel_PCA = KernelPCA(n_components = 15, kernel = 'linear')
kernel_PCA.fit(X_train)

X_train_pca = kernel_PCA.transform(X_train)
X_test_pca = kernel_PCA.transform(X_test)

Bag = RandomForestClassifier(n_estimators = 100, max_features = 15)
Bag = Bag.fit(X_train_pca, y_train)

y_test_pca = Bag.predict(X_test_pca) 
compute_scores("Kernel PCA", y_test_pca, y_test)

cm = confusion_matrix(y_test, y_test_pca, labels = Bag.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels = Bag.classes_)
disp.plot()

"""### Logistic Regression

"""

def evaluate(true, pred):
    accuracy = accuracy_score(true, pred)
    precision = precision_score(true, pred)
    recall = recall_score(true, pred)
    f1 = f1_score(true, pred)
    print('Accuracy: %s' % accuracy)
    print('Recall: %s' % recall)
    print('Precision: %s' % precision)
    print('F1: %s' % f1)
    cm = confusion_matrix(true, pred)
    sns.heatmap(cm, annot=True, cmap='viridis', xticklabels=['Predicted 0', 'Predicted 1'], yticklabels=['Actual 0', 'Actual 1'], fmt='5g')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

df = pd.read_csv('processed_data.csv')
y = df['RainTomorrow']
X = df.drop(columns=['row ID', 'RainTomorrow'])
alpha = 0.15 # percentage of dataset used
beta = 0.8 # percentage of training data

all_data, _, all_labels, _ = train_test_split(X, y, train_size=alpha, random_state=28)
X_train, X_test, y_train, y_test = train_test_split(all_data, all_labels, train_size=beta, random_state=42)
df_train = pd.concat([X_train, y_train], axis=1)
df_train.to_csv(r'C:\Users\rfrancu\Projects\eda\data\processed_train.csv', index=False)
df_test = pd.concat([X_test, y_test], axis=1)
df_test.to_csv(r'C:\Users\rfrancu\Projects\eda\data\processed_test.csv', index=False)

df_train = pd.read_csv(r'C:\Users\rfrancu\Projects\eda\data\processed_train.csv')
df_test = pd.read_csv(r'C:\Users\rfrancu\Projects\eda\data\processed_test.csv')
sns.countplot(x='RainTomorrow', data=df_train)
plt.show()

# make a selection on features and compute the correlation matrix
# V1
# FEATURE_COLUMN = ['Location', 'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation',
#                   'Sunshine', 'WindGustDir', 'WindGustSpeed', 'WindDir9am', 'WindDir3pm',
#                   'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am', 
#                   'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm',
#                   'RainToday']

# # V2, after feature selection based on model coeficients
FEATURE_COLUMN = ['MinTemp', 'MaxTemp', 'Rainfall','Sunshine',
                  'WindGustSpeed', 'Humidity9am', 'Humidity3pm', 
                  'Pressure9am', 'Pressure3pm', 'Cloud9am', 
                  'Cloud3pm', 'Temp9am', 'Temp3pm', 'RainToday']

TARGET_COLUMN = ['RainTomorrow']

df_train = df_train[FEATURE_COLUMN + TARGET_COLUMN]
df_test = df_test[FEATURE_COLUMN + TARGET_COLUMN]
corr_matrix = df_train.corr()
fig, axs = plt.subplots(figsize=(20, 20))
sns.heatmap(corr_matrix, annot=True, ax=axs)
plt.show()

X_train = df_train[FEATURE_COLUMN]
y_train = df_train[TARGET_COLUMN].to_numpy().ravel()
X_test = df_test[FEATURE_COLUMN]
y_test = df_test[TARGET_COLUMN].to_numpy().ravel()
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# define model
model = LogisticRegression(random_state=42, max_iter=1000)
# fit
model.fit(X_train_scaled, y_train)
y_train_pred = model.predict(X_train_scaled)
# test
y_test_pred = model.predict(X_test_scaled)
# performance
evaluate(y_test, y_test_pred)

from imblearn.over_sampling import SMOTE

# Apply SMOTE oversampling
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train_scaled, y_train)


# Train a logistic regression model on the oversampled data
model = LogisticRegression()
model.fit(X_train_sm, y_train_sm)

# Make predictions on the test data
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

evaluate(y_test, y_test_pred)

# Get feature importance based on model coeficients

coefficients = model.coef_

# Create a dataframe of the feature importance
feature_importance = pd.DataFrame(coefficients[0], X_train.columns, columns=['importance'])
feature_importance['importance'] = feature_importance['importance'].abs()
feature_importance.sort_values(by='importance', ascending=False, inplace=True)

# Print the top 10 most important features
print(feature_importance.head(12))

# HyperParameter tuning and cross-validation

# Define the hyperparameters and their possible values
parameters = {
    'C':[0.01,0.1,1,10], 
    'penalty':['l1','l2', 'elasticnet'], 
    'solver': ['liblinear', 'lbfgs']
    }

# Define the logistic regression model
log_reg = LogisticRegression(random_state=42, max_iter=1000)

# Define the cross-validation method
cv = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)

# Create a GridSearchCV object
clf = GridSearchCV(log_reg, parameters, cv=cv, scoring='f1')

# Fit the GridSearchCV object to the data
clf.fit(X_train_sm, y_train_sm)

# Print the best hyperparameters
print("Best Hyperparameters : " + str(clf.best_params_))

# define model
pca = PCA(n_components=10)
X_train_sm_pca = pca.fit_transform(X_train_sm)
X_train_pca = pca.transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)
model = LogisticRegression()
# fit it
model.fit(X_train_sm_pca, y_train_sm)
# test
y_train_pred = model.predict(X_train_pca)
y_test_pred = model.predict(X_test_pca)
# performance
evaluate(y_test, y_test_pred)

# define model
isomap = Isomap()
X_train_sm_iso = isomap.fit_transform(X_train_sm)
X_train_iso = isomap.transform(X_train_scaled)
X_test_iso = isomap.transform(X_test_scaled)
model = LogisticRegression()
# fit it
model.fit(X_train_sm_iso, y_train_sm)
# test
y_train_pred = model.predict(X_train_iso)
y_test_pred = model.predict(X_test_iso)
# performance
evaluate(y_train, y_train_pred)
evaluate(y_test, y_test_pred)

lle = LocallyLinearEmbedding()
X_train_sm_lle = lle.fit_transform(X_train_sm)
X_train_lle = lle.transform(X_train_scaled)
X_test_lle = lle.transform(X_test_scaled)
model = LogisticRegression()
# fit it
model.fit(X_train_sm_lle, y_train_sm)
# test
y_train_pred = model.predict(X_train_lle)
y_test_pred = model.predict(X_test_lle)
# performance
evaluate(y_train, y_train_pred)
evaluate(y_test, y_test_pred)

"""### SVM

"""

# find the best hyperparameters for baseline SVM
for ker in ['rbf', 'linear']:
    for c in [1, 1.5, 3]:
        svm_classifier = svm.SVC(kernel=ker, C=c)
        svm_classifier.fit(X_train, y_train)

        print("------------------------------------")
        print('kernel ' + ker + '; C: ' + str(c))
        predictions = svm_classifier.predict(X_test)
        print("Accuracy: " + str(accuracy_score(y_test, predictions)) + "; F1: " + str(f1_score(y_test, predictions)))

svm_classifier = svm.SVC(kernel='linear', C=1)
svm_classifier.fit(X_train, y_train)
y_test_base = svm_classifier.predict(X_test)
cm = confusion_matrix(y_test, y_test, labels = svm_classifier.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels = svm_classifier.classes_)
disp.plot()

# PCA
n_components_tries = [3, 5, 10, 15]
for n_components in n_components_tries:
  pca = PCA(n_components = n_components, random_state = 42)
  pca.fit(X_train)
  X_train_pca = pca.transform(X_train)
  X_test_pca = pca.transform(X_test)
  
  svm_classifier = svm.SVC(kernel='linear', C=1)
  svm_classifier.fit(X_train_pca, y_train)
  y_test_pca = svm_classifier.predict(X_test_pca) 
  m_f1_score = f1_score(y_test, y_test_pca)
  print(f'PCA with n_components {n_components} has f1 score {m_f1_score}')

pca = PCA(n_components = 15, random_state = 42)
pca.fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

svm_classifier = svm.SVC(kernel='linear', C=1)
svm_classifier.fit(X_train_pca, y_train)

y_test_pca = svm_classifier.predict(X_test_pca) 
compute_scores("PCA", y_test_pca, y_test)

cm = confusion_matrix(y_test, y_test_pca, labels = svm_classifier.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels = svm_classifier.classes_)
disp.plot()

# Isomap
n_components_tries = [3, 5, 10, 15]
for n_components in n_components_tries:
  isomap = Isomap(n_components = n_components)
  isomap.fit(X_train)
  X_train_isomap = isomap.transform(X_train)
  X_test_isomap = isomap.transform(X_test)
  
  svm_classifier = svm.SVC(kernel='linear', C=1)
  svm_classifier.fit(X_train_isomap, y_train)
  y_test_isomap = svm_classifier.predict(X_test_isomap) 
  m_f1_score = f1_score(y_test, y_test_isomap)
  print(f'Isomap with n_components {n_components} has f1 score {m_f1_score}')

isomap = Isomap(n_components = 15)
isomap.fit(X_train)
X_train_isomap = isomap.transform(X_train)
X_test_isomap = isomap.transform(X_test)
svm_classifier = svm.SVC(kernel='linear', C=1)
svm_classifier.fit(X_train_isomap, y_train)
y_test_isomap = svm_classifier.predict(X_test_isomap) 

cm = confusion_matrix(y_test, y_test_isomap, labels = svm_classifier.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels = svm_classifier.classes_)
disp.plot()

compute_scores("Isomap", y_test_isomap, y_test)

# Locally Linear Embedding
n_components_tries = [3, 5, 10, 15]

for n_components in n_components_tries:
    lle = LocallyLinearEmbedding(n_components = n_components)
    lle.fit(X_train)

    X_train_lle = lle.transform(X_train)
    X_test_lle = lle.transform(X_test)

    svm_classifier = svm.SVC(kernel='linear', C=1)
    svm_classifier.fit(X_train_lle, y_train)
    y_test_lle = svm_classifier.predict(X_test_lle) 
    m_f1_score = f1_score(y_test, y_test_lle)

    print(f'Locally Linear Embedding with n_components {n_components} has f1 score {m_f1_score}')

lle = LocallyLinearEmbedding(n_components = 15)
lle.fit(X_train)

X_train_lle = lle.transform(X_train)
X_test_lle = lle.transform(X_test)

svm_classifier = svm.SVC(kernel='linear', C=1)
svm_classifier.fit(X_train_lle, y_train)
y_test_lle = svm_classifier.predict(X_test_lle) 

cm = confusion_matrix(y_test, y_test_lle, labels = svm_classifier.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels = svm_classifier.classes_)
disp.plot()

compute_scores("Locally Linear Embedding", y_test_lle, y_test)

# Kernel PCA
n_components_tries = [3, 5, 10, 15]
kernels = ['linear', 'poly', 'rbf', 'sigmoid', 'cosine']

for n_components in n_components_tries:
    for kernel in kernels:
      kernel_PCA = KernelPCA(n_components = n_components, kernel = kernel)
      kernel_PCA.fit(X_train)

      X_train_pca = kernel_PCA.transform(X_train)
      X_test_pca = kernel_PCA.transform(X_test)

      svm_classifier = svm.SVC(kernel='linear', C=1)
      svm_classifier.fit(X_train_pca, y_train)
      
      y_test_pca = svm_classifier.predict(X_test_pca) 
      m_f1_score = f1_score(y_test, y_test_pca)

      print(f'PCA with kernel {kernel} and n_components {n_components} has f1 score {m_f1_score}')

"""### XGBoost

"""

from xgboost import XGBClassifier

def train_xgboost(X_train, y_train, X_val, y_val):
    model = XGBClassifier()
    model.fit(X_train, y_train)

    # show importances
    importances = model.feature_importances_
    importances = pd.Series(importances, index = X.columns)
    print("==== IMPORTANCES ====\n")
    print(importances)

    predictions = model.predict(X_val)
    
    # plot confusion matrix
    cm = confusion_matrix(y_val, predictions, labels=model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    print("\n==== CONFUSION MATRIX ====")
    disp.plot()
    plt.show()
    
    # show classification report
    print("\n==== CLASSIFICATION REPORT ====\n")
    print(classification_report(y_val, predictions))
    
    return model, predictions
    

def compute_result(y, predictions):   
    acc = accuracy_score(y, predictions)
    prec = precision_score(y, predictions)
    rec = recall_score(y, predictions)
    f1 = f1_score(y, predictions)

    result = pd.DataFrame({'Model': ['XGBoost']})

    result['accuracy'] = acc
    result['precision'] = prec
    result['recall'] = rec
    result['f1'] = f1

    return result

def plot_features(model, X_val):
    feature_importance = model.feature_importances_
    sorted_idx = np.argsort(feature_importance)
    fig = plt.figure(figsize=(12, 6))
    plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
    plt.yticks(range(len(sorted_idx)), np.array(X_val.columns)[sorted_idx])
    plt.title('Feature Importance')

# baseline

model, predictions = train_xgboost(X_train, y_train, X_test, y_test)
plot_features(model, X_train)
results = compute_result(y_test, predictions)
results

# PCA
n_components_tries = [3, 5, 10, 15]
for n_components in n_components_tries:
  pca = PCA(n_components = n_components, random_state = 42)
  pca.fit(X_train)
  X_train_pca = pca.transform(X_train)
  X_test_pca = pca.transform(X_test)
  Bag = XGBClassifier()
  Bag = Bag.fit(X_train_pca, y_train)
  y_test_pca = Bag.predict(X_test_pca) 
  m_f1_score = f1_score(y_test, y_test_pca)
  print(f'PCA with n_components {n_components} has f1 score {m_f1_score}')

svd_solvers = ['full', 'arpack', 'randomized']
for svd_solver in svd_solvers:
  pca = PCA(n_components = 10, random_state = 42, svd_solver = svd_solver)
  pca.fit(X_train)
  X_train_pca = pca.transform(X_train)
  X_test_pca = pca.transform(X_test)
  Bag = XGBClassifier()
  Bag = Bag.fit(X_train_pca, y_train)
  y_test_pca = Bag.predict(X_test_pca) 
  m_f1_score = f1_score(y_test, y_test_pca)
  print(f'PCA with n_components {n_components} and {svd_solver} svd has f1 score {m_f1_score}')

pca = PCA(n_components = 10, svd_solver = "arpack", random_state = 42)
pca.fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
Bag = XGBClassifier()
Bag = Bag.fit(X_train_pca, y_train)
y_test_pca = Bag.predict(X_test_pca) 


pca_results = compute_result(y_test, y_test_pca)
pca_results

# plot confusion matrix
cm = confusion_matrix(y_test, y_test_pca, labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
print("\n==== CONFUSION MATRIX ====")
disp.plot()
plt.show()

# ISOMAP
n_components_tries = [3, 5, 10, 15]
for n_components in n_components_tries:
  isomap = Isomap(n_components = n_components)
  isomap.fit(X_train)
  X_train_isomap = isomap.transform(X_train)
  X_test_isomap = isomap.transform(X_test)
  Bag = XGBClassifier()
  Bag = Bag.fit(X_train_isomap, y_train)
  y_test_isomap = Bag.predict(X_test_isomap) 
  m_f1_score = f1_score(y_test, y_test_isomap)
  print(f'Isomap with n_components {n_components} has f1 score {m_f1_score}')


isomap = Isomap(n_components = 10)
isomap.fit(X_train)
X_train_isomap = isomap.transform(X_train)
X_test_isomap = isomap.transform(X_test)
Bag = XGBClassifier()
Bag = Bag.fit(X_train_isomap, y_train)
y_test_isomap = Bag.predict(X_test_isomap)

isomap_results = compute_result(y_test, y_test_isomap)
isomap_results

# plot confusion matrix
cm = confusion_matrix(y_test, y_test_isomap, labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
print("\n==== CONFUSION MATRIX ====")
disp.plot()
plt.show()

# LLE

n_components_tries = [5, 10, 15]

for n_components in n_components_tries:
    lle = LocallyLinearEmbedding(n_components = n_components)
    lle.fit(X_train)

    X_train_lle = lle.transform(X_train)
    X_test_lle = lle.transform(X_test)

    Bag = XGBClassifier()
    Bag = Bag.fit(X_train_lle, y_train)

    y_test_lle = Bag.predict(X_test_lle) 
    m_f1_score = f1_score(y_test, y_test_lle)

    print(f'Locally Linear Embedding with n_components {n_components} has f1 score {m_f1_score}')

lle = LocallyLinearEmbedding(n_components = 18)
lle.fit(X_train)

X_train_lle = lle.transform(X_train)
X_test_lle = lle.transform(X_test)

Bag = XGBClassifier()
Bag = Bag.fit(X_train_lle, y_train)

y_test_lle = Bag.predict(X_test_lle) 

lle_results = compute_result(y_test, y_test_lle)
lle_results

# plot confusion matrix
cm = confusion_matrix(y_test, y_test_lle, labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
print("\n==== CONFUSION MATRIX ====")
disp.plot()
plt.show()

# Kernel PCA

n_components_tries = [5, 10, 15]
kernels = ['linear', 'poly', 'rbf', 'sigmoid', 'cosine']

for n_components in n_components_tries:
    for kernel in kernels:
      kernel_PCA = KernelPCA(n_components = n_components, kernel = kernel)
      kernel_PCA.fit(X_train)

      X_train_pca = kernel_PCA.transform(X_train)
      X_test_pca = kernel_PCA.transform(X_test)

      Bag = XGBClassifier()
      Bag = Bag.fit(X_train_pca, y_train)

      y_test_pca = Bag.predict(X_test_pca) 
      m_f1_score = f1_score(y_test, y_test_pca)

      print(f'PCA with kernel {kernel} and n_components {n_components} has f1 score {m_f1_score}')

kernel_PCA = KernelPCA(n_components = 15, kernel = 'cosine')
kernel_PCA.fit(X_train)

X_train_pca = kernel_PCA.transform(X_train)
X_test_pca = kernel_PCA.transform(X_test)

Bag = XGBClassifier()
Bag = Bag.fit(X_train_pca, y_train)

y_test_pca = Bag.predict(X_test_pca) 

kpca_results = compute_result(y_test, y_test_pca)
kpca_results