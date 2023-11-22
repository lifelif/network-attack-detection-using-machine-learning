import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from matplotlib import pyplot

from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import TimeSeriesSplit



import warnings
warnings.filterwarnings("ignore")


df8=pd.read_csv("Wednesday-workingHours.pcap_ISCX.csv" )
df7=pd.read_csv("Tuesday-WorkingHours.pcap_ISCX.csv" )
df6=pd.read_csv("Monday-WorkingHours.pcap_ISCX.csv")
df1=pd.read_csv("Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")#,nrows = 50000
df2=pd.read_csv("Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv")
df3=pd.read_csv("Friday-WorkingHours-Morning.pcap_ISCX.csv")
df4=pd.read_csv("Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv")
df5=pd.read_csv("Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv")

df=pd.concat([df1, df2, df3, df4, df5, df6, df7, df8], ignore_index=True)
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)


df[' Label']=df[' Label'].apply(lambda x: 0 if x=='BENIGN' else 1)

safe_y = df[[' Label']]

col_to_exclude = [' Label']
df = df.drop(col_to_exclude, axis=1)

#Just select the categorical variables
cat_col = ['object']
cat_columns = list(df.select_dtypes(include=cat_col).columns)
cat_data = df[cat_columns]
cat_vars = cat_data.columns


def cap_data(df):
    for col in df.columns:
        print("Capping the ",col)
        if (((df[col].dtype)=='float64') | ((df[col].dtype)=='int64')):
            percentiles=df[col].quantile([0.1,0.99]).values
            df[col][df[col] <=percentiles[0]]= percentiles[0]
            df[col][df[col] >= percentiles[1]] = percentiles[1]
        else:
            df[col]=df[col]
    return df

final_df=cap_data(df)
df= final_df


#Create dummy variables for each cat. variable
for var in cat_vars:
    cat_list = pd.get_dummies(df[var], prefix=var)
    bank=df.join(cat_list)


data_vars=df.columns.values.tolist()
to_keep=[i for i in data_vars if i not in cat_vars]

#Create final dataframe
df_final=df[to_keep]
df = pd.concat([df_final, safe_y], axis=1)
df.head()

x = df.drop(' Label', axis=1)
y = df[' Label']

trainX, testX, trainY, testY = train_test_split(x, y, test_size = 0.2)
clf=DecisionTreeClassifier()
clf.fit(trainX, trainY)


preds_train = clf.predict(trainX)
preds_test = clf.predict(testX)


from sklearn.metrics import confusion_matrix

def confMatrix(y_test_scaled, y_predicted):
    cm = confusion_matrix(testY, y_predicted)
    plt.figure(figsize=(15, 10))
    plt.clf()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)
    classNames = ['Negative', 'Positive']
    plt.title('Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    tick_marks = np.arange(len(classNames))
    plt.xticks(tick_marks, classNames, rotation=45)
    plt.yticks(tick_marks, classNames)
    s = [['TN', 'FP'], ['FN', 'TP']]

    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(s[i][j]) + " = " + str(cm[i][j]))
    plt.show()


confMatrix(testY,preds_test)

print('Decision Tree:'
      '\n> ....Accuracy on training data = {:.4f}'
      '\n> ....Accuracy on validation data = {:.4f}'
      '\n> ....Recall on training data = {:.4f}'
      '\n> ....Recall on validation data = {:.4f}'
      '\n> ....Precision on validation data = {:.4f}'
      '\n> ....Precision on validation data = {:.4f}'.format(
    accuracy_score(y_true=trainY, y_pred=preds_train),
    accuracy_score(y_true=testY, y_pred=preds_test),
    recall_score(y_true=trainY, y_pred=preds_train),
    recall_score(y_true=testY, y_pred=preds_test),
    precision_score(y_true=trainY, y_pred=preds_train),
    precision_score(y_true=testY, y_pred=preds_test)
))


progress = dict()
eval_set = [(testX, testY)]
clf.fit(trainX, trainY)

clf_es = DecisionTreeClassifier()

eval_set = [(testX, testY)]
clf_es.fit(trainX, trainY, early_stopping_rounds=7, eval_metric="error", eval_set=eval_set, verbose=False)

preds_train = clf_es.predict(trainX)
preds_test = clf_es.predict(testX)

print('DecisionTree:'
      '\n> Accuracy on training data = {:.4f}'
      '\n> Accuracy on validation data = {:.4f}'
      '\n> Recall on training data = {:.4f}'
      '\n> Recall on validation data = {:.4f}'
      '\n> Precision on validation data = {:.4f}'
      '\n> Precision on validation data = {:.4f}'.format(
    accuracy_score(y_true=trainY, y_pred=preds_train),
    accuracy_score(y_true=testY, y_pred=preds_test),
    recall_score(y_true=trainY, y_pred=preds_train),
    recall_score(y_true=testY, y_pred=preds_test),
    precision_score(y_true=trainY, y_pred=preds_train),
    precision_score(y_true=testY, y_pred=preds_test)
))

print(clf_es.feature_importances_)
print()
print('Length of feature_importances_ list: ' + str(len(clf_es.feature_importances_)))
print()
print('Number of predictors in trainX: ' + str(trainX.shape[1]))

# plot feature importance
plot_importance(clf_es)
pyplot.show()

feature_names = trainX.columns

feature_importance_df = pd.DataFrame(clf_es.feature_importances_, feature_names)
feature_importance_df = feature_importance_df.reset_index()
feature_importance_df.columns = ['Feature', 'Importance']
print("Feature Importance -->",feature_importance_df)

feature_importance_df_top_10 = feature_importance_df.sort_values(by='Importance', ascending=False).head(10)
print("Top 10 Feature ---> ",feature_importance_df_top_10)

plt.barh(feature_importance_df_top_10.Feature, feature_importance_df_top_10.Importance)

features_selected = clf_es.get_booster().get_score(importance_type='gain')
keys = list(features_selected.keys())
values = list(features_selected.values())

features_selected = pd.DataFrame(data=values,
                                              index=keys,
                                              columns=["Importance"]).sort_values(by = "Importance",
                                                                             ascending=False)
features_selected.plot(kind='barh')

print()
print('Length of remaining predictors after DT: ' + str(len(features_selected)))

print(feature_importance_df[(feature_importance_df["Importance"] == 0)])
print()
print('Length of features with Importance = zero:  ' + str(feature_importance_df[(feature_importance_df["Importance"] == 0)].shape[0] ))

top_10_of_retained_features_from_model = features_selected.sort_values(by='Importance', ascending=False).head(10)
print("Top_10_of_retained_features_from_model ---> ", top_10_of_retained_features_from_model)

plt.barh(top_10_of_retained_features_from_model.index, top_10_of_retained_features_from_model.Importance)

dt_grid = DecisionTreeClassifier(objective= 'binary:logistic')

parameters = {
    'max_depth': range (6, 7, 1),
    'colsample_bytree': [0.6],
    'gamma': [1],
    'n_estimators': range(100, 140, 40),
    'learning_rate': [0.1]}
eval_metric = ["rmse","mae"]
fit_params={"early_stopping_rounds":10,
            "eval_metric" : "rmse",
            "eval_set" : [[testX, testY]]}
history=dt_grid.fit(trainX, trainY, eval_metric=eval_metric, eval_set=eval_set)
cv = 5

grid_search = GridSearchCV(
    estimator=dt_grid,
    param_grid=parameters,
    scoring = 'neg_log_loss',
    n_jobs = -1,
    cv = TimeSeriesSplit(n_splits=cv).get_n_splits([trainX, trainY]),
    verbose=0)

dt_grid_model = grid_search.fit(trainX, trainY, **fit_params)

print('Best Parameter:')
print(dt_grid_model.best_params_)
print()
print('------------------------------------------------------------------')
print()
print(dt_grid_model.best_estimator_)

preds_train = dt_grid_model.predict(trainX)
preds_test = dt_grid_model.predict(testX)


print('Decision Tree:'
      '\n> Accuracy on training data = {:.4f}'
      '\n> Accuracy on validation data = {:.4f}'
      '\n> Recall on training data = {:.4f}'
      '\n> Recall on validation data = {:.4f}'
      '\n> Precision on validation data = {:.4f}'
      '\n> Precision on validation data = {:.4f}'.format(
    accuracy_score(y_true=trainY, y_pred=preds_train),
    accuracy_score(y_true=testY, y_pred=preds_test),
    recall_score(y_true=trainY, y_pred=preds_train),
    recall_score(y_true=testY, y_pred=preds_test),
    precision_score(y_true=trainY, y_pred=preds_train),
    precision_score(y_true=testY, y_pred=preds_test)
))


