from automl.models.basemodels.classifiers import *
from automl.models.basemodels.regressors import *
from automl.models.basemodels.ensemblers import *
from skopt import BayesSearchCV
import matplotlib.pyplot as plt
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

plt.switch_backend('agg')
import base64
from io import BytesIO
import logging
# logging.basicConfig(level=logging.INFO)
classify_models={
    'lr':BaseLogisticRegression(),
         'dt':BaseDecisionTreeClassifier(),
         'svc':BaseSVC(),
         'rf':BaseRandomForestClassifier(),
         'knn':BaseKNeighborsClassifier(),
     'sgd':BaseSGDClassifier(),
     'nb':BaseGaussianNB(),
}

regress_models={
    'lr':BaseLinearRegressor(),
     'lasso':BaseLassoRegressor(),
     'dt':BaseDecisionRegressor(),
     'elnet':BaseElasticNetRegressor(),
         'knn':BaseKNeighborsRegressor(),
         'svr':BaseSVRRegressor(),
         'rf':BaseRandomForRegressor(),
         'knn':BaseKNeighborsRegressor(),
}

classif_bag_model = {
    'vc':BaseVotingClassifier(),
}
classif_boost_model ={
    'xgboost':BaseXGBClassifier()
}
regress_bag_model = {
    'vc':BaseVotingRegressor(),
}
regress_boost_model={
    'xgboost':BaseXGBRegressor()
}
def get_best_param(estimator,tune_grid,X,y):
    logging.info(f"Tune grid: {tune_grid}")
    logging.info(f"Model name: {estimator}")
    grid = BayesSearchCV(estimator=estimator,search_spaces=tune_grid,cv=10,n_jobs=-1) #cv=kfold
    
    logging.info(f'Total iterations: {grid.total_iterations}')
    grid.fit(X,y,callback=on_step)
    logging.info(f"Best score:  {grid.best_score_}")
    logging.info(f"Best Parameters: {grid.best_params_}")
    return (grid.best_params_,grid.best_score_)
    
def on_step(optim_result):
    score = -optim_result['fun']
    logging.info("best score: %s" % score)
    if score >= 0.90:
        logging.info('Interrupting!')
        return True

def plot_config(**kwargs):
        # plt.figure(figsize=kwargs.get('figsize',(8,8)))
        plt.title(kwargs.get('title',''))
        plt.xlabel(kwargs.get("xlabel",'feature'))
        plt.ylabel(kwargs.get('ylabel','stat'))
        tmpfile = BytesIO()
        plt.savefig(tmpfile, format='png')
        plt.close()
        encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')    
        
        return encoded

from skopt.utils import use_named_args

# @use_named_args(space)
# def objective(**params):
#     reg.set_params(**params)

#     return -np.mean(cross_val_score(reg, X, y, cv=5, n_jobs=-1,
#                                     scoring="neg_mean_absolute_error"))

# def best_space(estimator,tune_grid,X,y):
    