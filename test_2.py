
from automl.models import getregression,getclassification
from joblib import load,dump
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,precision_score,recall_score,accuracy_score
from sklearn.metrics import mean_absolute_error,mean_squared_error
##-------------------------Classification Problem---------------------------#
df = pd.read_csv('diabetes_1.csv',nrows=1000)
# print(df.head())
target = 'Outcome'
x_train,x_test,y_train,y_test = train_test_split(df.drop(target,axis=1),df[target],test_size=0.2,stratify=df[target],random_state=45)
train_df = pd.concat([x_train,y_train],axis=1)
test_df = pd.concat([x_test,y_test],axis=1)

##---------------------------Training-------------------------------#
basemodel = getclassification.BestClassificationModel(train_df,target)
basemodel.fit() #fits to best model
dump(basemodel.preprocess_pipe,'classif_preprocess_pipe.joblib')
metrics = basemodel.scores_grid
print(metrics)
dump(basemodel,'best-classif-model.joblib')
dump(metrics,'classif-metrics.joblib')

# #---------------------------Testing---------------------------------#
pipe = load('classif_preprocess_pipe.joblib')
model = load('best-classif-model.joblib')
print(model.max_model.modelname)
print(model.max_model.get_params())
predictions = model.predict(x_test)
predictions = model.get_inverse_label(predictions)
# #-------------------------Score-------------------------------------#
n_df = pd.DataFrame(list(zip(predictions,y_test)),columns=['predicts','actual'])
n_df['predicts'] = n_df['predicts'].apply(lambda x: 1 if x=='Yes' else 0)
n_df['actual'] = n_df['actual'].apply(lambda x: 1 if x=='Yes' else 0)
print(n_df.head())
# print('CORRECT:',model.score(n_df['actual'],n_df['predicts'],normalize=False),"TOTAL:",len(n_df['actual']))
print('SCORE:',accuracy_score(n_df['actual'],n_df['predicts']))


##------------------------------Regression Problem--------------------#

df = pd.read_csv('house_price.csv',nrows=50)
print(df.shape[0])
target = 'TARGET(PRICE_IN_LACS)'
x_train,x_test,y_train,y_test = train_test_split(df.drop(target,axis=1),df[target],test_size=0.2,random_state=45)
train_df = pd.concat([x_train,y_train],axis=1)
test_df = pd.concat([x_test,y_test],axis=1)

# #---------------------------Training-------------------------------#
basemodel = getregression.BestRegessionModel(train_df,target)
basemodel.fit() #fits to best model
dump(basemodel.preprocess_pipe,'regress_preprocess_pipe.joblib')
metrics = basemodel.scores_grid
dump(basemodel,'best-regress-model.joblib')
dump(metrics,'regress-metrics.joblib')
print(metrics)
##---------------------------Testing---------------------------------#
# pipey = load('preprocess_pipe.joblib')
model = load('best-regress-model.joblib')
predictions = basemodel.predict(x_train)
##-------------------------Score-------------------------------------#
n_df = pd.DataFrame(list(zip(predictions,y_train)),columns=['predicts','actual'])
print(n_df.head())
print('SCORE:',mean_squared_error(n_df['actual'],n_df['predicts'],squared=False))
print(basemodel.scores_grid)
