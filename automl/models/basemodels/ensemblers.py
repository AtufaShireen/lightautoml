import numpy as np
class BaseEstimator():
    """
    For Ensuring format of Ensemble models
    id: for index,
    name: display name,
    class_def: model name,
    tune_grid: hyperparameters to tune,
    is_gpu_enabled: default: None
    preprocess_steps(in future): preprocess steps as per model
    """
    def __init__(
        self,id,name,class_def,tune_grid,args={},
        tune_args={},
        is_gpu_enabled=False,preprocess_steps={},random_state=45):
        
        if not args:
            args = {}
        if not tune_grid:
            tune_grid = {}
        if not tune_args:
            tune_args = {}
        if not preprocess_steps:
            preprocess_steps= {}
        self.args = args
        self.is_gpu_enabled = is_gpu_enabled
        self.tune_grid = tune_grid
        self.tune_args = tune_args
        self.preprocess_steps=preprocess_steps
        self.id= id
        self.name = name
        self.class_def = class_def


    def get_dict(self):
        """
        TO get model properties
        """
        d = [
            ("ID", self.id),
            ("Name", self.name),
                ("Class", self.class_def),
                ("Args", self.args),
                ("Tune Grid", self.tune_grid),
                ("Tune Args", self.tune_args),
        ]

        return dict(d)

# classification
class BaseAdaBoostClassifier(BaseEstimator):
    def __init__(self, ) :
        #logger = get_logger\(\)
        preprocess_steps = {'scale_data':False,'dummify_categoricals':False}
        from sklearn.ensemble import AdaBoostClassifier

        args = {"random_state": 45}
        tune_args = {}
        tune_grid = {
            "n_estimators": np.arange(10, 300, 10, ),
            "learning_rate": [
                0.0000001,
                0.000001,
                0.0001,
                0.001,
                0.01,
                0.0005,
                0.005,
                0.05,
                0.1,
                0.15,
                0.2,
                0.3,
                0.4,
                0.5,
            ],
            "algorithm": ["SAMME", "SAMME.R"],
        }
        super().__init__(
            id="ada",
            name="Ada Boost Classifier",
            class_def=AdaBoostClassifier(),
            args=args,
            tune_grid=tune_grid,
            tune_args=tune_args,
            preprocess_steps=preprocess_steps,
        )


class BaseGradientBoostingClassifier(BaseEstimator):
    def __init__(self, ) :
        #logger = get_logger\(\)
        preprocess_steps = {'scale_data':False,'dummify_categoricals':False}
        from sklearn.ensemble import GradientBoostingClassifier

        args = {"random_state": 45}
        tune_args = {}
        tune_grid = {
            "n_estimators": np.arange(10, 300, 10, ),
            "learning_rate": [
                0.0000001,
                0.000001,
                0.0001,
                0.001,
                0.01,
                0.0005,
                0.005,
                0.05,
                0.1,
                0.15,
                0.2,
                0.3,
                0.4,
                0.5,
            ],
            "subsample": np.arange(0.2, 1, 0.05, ),
            "min_samples_split": [2, 4, 5, 7, 9, 10],
            "min_samples_leaf": [1, 2, 3, 4, 5],
            "max_depth": np.arange(1, 11, 1, ),
            "min_impurity_decrease": [
                0,
                0.0001,
                0.001,
                0.01,
                0.0002,
                0.002,
                0.02,
                0.0005,
                0.005,
                0.05,
                0.1,
                0.2,
                0.3,
                0.4,
                0.5,
            ],
            "max_features": [ "sqrt", "log2"],
        }
        super().__init__(
            id="gbc",
            name="Gradient Boosting Classifier",
            class_def=GradientBoostingClassifier(),
            args=args,
            tune_grid=tune_grid,
            
            tune_args=tune_args,
            preprocess_steps=preprocess_steps,
        )

class BaseExtraTreesClassifier(BaseEstimator):
    pass

class BaseXGBClassifier(BaseEstimator):
    def __init__(self, ) :
        # logger = get_logger()
        preprocess_steps = {'scale_data':False,'dummify_categoricals':False}
        from xgboost import XGBClassifier

        args = {
        }
        tune_args = {}
        tune_grid = {
            "learning_rate": [
                0.0000001,
                0.000001,
                0.0001,
                0.001,
                0.01,
                0.0005,
                0.005,
                0.05,
                0.1,
                0.15,
                0.2,
                0.3,
                0.4,
                0.5,
            ],
            "n_estimators": np.arange(10, 300, 10, ),
            "subsample": [0.2, 0.3, 0.5, 0.7, 0.9, 1],
            "max_depth": np.arange(1, 11, 1, ),
            "colsample_bytree": [0.5, 0.7, 0.9, 1],
            "min_child_weight": [1, 2, 3, 4],
            "reg_alpha": [
                0.0000001,
                0.000001,
                0.0001,
                0.001,
                0.01,
                0.0005,
                0.005,
                0.05,
                0.1,
                0.15,
                0.2,
                0.3,
                0.4,
                0.5,
                0.7,
                1,
                2,
                3,
                4,
                5,
                10,
            ],
            "reg_lambda": [
                0.0000001,
                0.000001,
                0.0001,
                0.001,
                0.01,
                0.0005,
                0.005,
                0.05,
                0.1,
                0.15,
                0.2,
                0.3,
                0.4,
                0.5,
                0.7,
                1,
                2,
                3,
                4,
                5,
                10,
            ],
            "scale_pos_weight": np.arange(0.5, 50, 0.1, ),
            "use_label_encoder":[False],
            "eval_metric":['merror'],
        }
        super().__init__(
            id="xgboost",
            name="Extreme Gradient Boosting",
            class_def=XGBClassifier(),
            args=args,
            tune_grid=tune_grid,
            tune_args=tune_args,
        )

class BaseCatBoostClassifier(BaseEstimator):
    pass

class BaseBaggingClassifier(BaseEstimator):
    def __init__(self, ) :
        #logger = get_logger\(\)
        preprocess_steps = {'scale_data':False,'dummify_categoricals':False}
        from sklearn.ensemble import BaggingClassifier

        args = {
            "random_state": 45,
            "n_jobs": 1,
        }
        tune_args = {}
        tune_grid = {
            "bootstrap": [True, False],
            "bootstrap_features": [True, False],
            "max_samples": np.arange(0.5, 1, 0.2),
        }

        super().__init__(
            id="Bagging",
            name="Bagging Classifier",
            class_def=BaggingClassifier(),
            args=args,
            tune_grid=tune_grid,
            tune_args=tune_args,
            preprocess_steps=preprocess_steps,
            
        )


class BaseStackingClassifier(BaseEstimator):
    def __init__(self, ) :
        #logger = get_logger\(\)
        
        from sklearn.ensemble import StackingClassifier
        preprocess_steps = {'scale_data':False,'dummify_categoricals':False}
        args = {}
        tune_args = {}
        tune_grid = {}
        

        

        super().__init__(
            id="Stacking",
            name="Stacking Classifier",
            class_def=StackingClassifier(),
            args=args,
            tune_grid=tune_grid,
            
            tune_args=tune_args,
            preprocess_steps=preprocess_steps,
            
        )


class BaseVotingClassifier(BaseEstimator):
    def __init__(self, ) :
        from sklearn.ensemble import VotingClassifier

        args = {}
        tune_args = {}
        tune_grid = {}

        super().__init__(
            id="Voting",
            name="Voting Classifier",
            class_def=VotingClassifier(estimators=[]),
            args=args,
            tune_grid=tune_grid,
            
            tune_args=tune_args,
            
        )


# regression
class BaseExtraTreesRegressor(BaseEstimator):
    pass

class BaseAdaBoostRegressor(BaseEstimator):
    def __init__(self, ) -> None:
        
        from sklearn.ensemble import AdaBoostRegressor

        args = {"random_state": 45}
        tune_args = {}
        tune_grid = {
            "n_estimators": np.arange(10, 300, 10),
            "learning_rate": [
                0.0000001,
                0.000001,
                0.0001,
                0.001,
                0.01,
                0.0005,
                0.005,
                0.05,
                0.1,
                0.15,
                0.2,
                0.3,
                0.4,
                0.5,
            ],
            "loss": ["linear", "square", "exponential"],
        }
        preprocess_steps={'scale_data':True,'dummify_categoricals':True}
        super().__init__(
            id="ada",
            name="AdaBoost Regressor",
            class_def=AdaBoostRegressor(),
            args=args,
            tune_grid=tune_grid,
            tune_args=tune_args,
            preprocess_steps = preprocess_steps,
        )


class BaseGradientBoostingRegressor(BaseEstimator):
    def __init__(self, ) -> None:
        

        from sklearn.ensemble import GradientBoostingRegressor

        args = {"random_state": 45}
        tune_args = {}
        tune_grid = {
            "n_estimators": np.arange(10, 300, 10),
            "learning_rate": [
                0.0000001,
                0.000001,
                0.0001,
                0.001,
                0.01,
                0.0005,
                0.005,
                0.05,
                0.1,
                0.15,
                0.2,
                0.3,
                0.4,
                0.5,
            ],
            "subsample": np.arange(0.2, 1, 0.05),
            "min_samples_split": [2, 4, 5, 7, 9, 10],
            "min_samples_leaf": [1, 2, 3, 4, 5],
            "max_depth": np.arange(1, 11, 1),
            "min_impurity_decrease": [
                0,
                0.0001,
                0.001,
                0.01,
                0.0002,
                0.002,
                0.02,
                0.0005,
                0.005,
                0.05,
                0.1,
                0.2,
                0.3,
                0.4,
                0.5,
            ],
            "max_features": ["sqrt", "log2"],
        }
        preprocess_steps={'scale_data':True,'dummify_categoricals':True}
        super().__init__(
            id="gbr",
            name="Gradient Boosting Regressor",
            class_def=GradientBoostingRegressor(),
            args=args,
            tune_grid=tune_grid,
            tune_args=tune_args,
            preprocess_steps = preprocess_steps,
        )


class BaseXGBRegressor(BaseEstimator):
    def __init__(self, ) -> None:
        
        try:
            import xgboost
        except ImportError:
            print("Couldn't import xgboost.XGBRegressor")
            
            return
        from xgboost import XGBRegressor

        args = {
        }
        tune_args = {}
        tune_grid = {
            "booster": ["gbtree"],
            "tree_method":  ["auto"],
            "learning_rate": [
                0.0000001,
                0.000001,
                0.0001,
                0.001,
                0.01,
                0.0005,
                0.005,
                0.05,
                0.1,
                0.15,
                0.2,
                0.3,
                0.4,
                0.5,
            ],
            "n_estimators": np.arange(10, 300, 10),
            "subsample": [0.2,  1, 0.3, 0.5,0.7, 0.9],
            "max_depth": np.arange(1, 11, 1),
            "colsample_bytree": [0.5, 0.7, 1, 0.9],
            "min_child_weight": [1,  4,2, 3],
            "reg_alpha": [
                0.0000001,
                0.000001,
                0.0001,
                0.001,
                0.01,
                0.0005,
                0.005,
                0.05,
                0.1,
                0.15,
                0.2,
                0.3,
                0.4,
                0.5,
                0.7,
                1,
                2,
                3,
                4,
                5,
                10,
            ],
            "reg_lambda": [
                0.0000001,
                0.000001,
                0.0001,
                0.001,
                0.01,
                0.0005,
                0.005,
                0.05,
                0.1,
                0.15,
                0.2,
                0.3,
                0.4,
                0.5,
                0.7,
                1,
                2,
                3,
                4,
                5,
                10,
            ],
            "scale_pos_weight": np.arange(1, 50, 0.1),
        }
        preprocess_steps={'scale_data':False,'dummify_categoricals':True}
        super().__init__(
            id="xgboost",
            name="Extreme Gradient Boosting",
            class_def=XGBRegressor(),
            args=args,
            tune_grid=tune_grid,
            tune_args=tune_args,
            preprocess_steps=preprocess_steps,
        )
  
class BaggingRegressor(BaseEstimator):
    def __init__(self, ) -> None:
        
        from sklearn.ensemble import BaggingRegressor

        args = {
            "random_state": 45,
            "n_jobs": -1,
        }
        tune_args = {}
        tune_grid = {
            "bootstrap": [True, False],
            "bootstrap_features": [True, False],
            "max_samples": np.arange(0.4, 1, 0.1),
        }
        super().__init__(
            id="Bagging",
            name="Bagging Regressor",
            class_def=BaggingRegressor(),
            args=args,
            tune_grid=tune_grid,
            
            tune_args=tune_args,
        )


class BaseStackingRegressor(BaseEstimator):
    def __init__(self, ) -> None:
        
        from sklearn.ensemble import StackingRegressor

        args = {}
        tune_args = {}
        tune_grid = {}       

        super().__init__(
            id="Stacking",
            name="Stacking Regressor",
            class_def=StackingRegressor(),
            args=args,
            tune_grid=tune_grid,
            
            tune_args=tune_args,
        )

class BaseVotingRegressor(BaseEstimator):
    def __init__(self, ) :
        from sklearn.ensemble import VotingRegressor

        args = {}
        tune_args = {}
        tune_grid = {}
        estimators = []

        super().__init__(
            id="Voting",
            name="Voting Regressor",
            class_def=VotingRegressor(estimators=estimators),
            args=args,
            tune_grid=tune_grid,
            tune_args=tune_args,
            
        )
