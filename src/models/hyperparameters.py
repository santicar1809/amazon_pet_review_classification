from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier

## Logistic Regression Model
def all_models():
    seed=42
    '''This function will host all the model parameters, can be used to iterate the
    grid search '''

    lr_pipeline = Pipeline([
        ('Logreg', LogisticRegression(max_iter=10000,multi_class='multinomial',solver='lbfgs',class_weight="balanced"))
    ])

    lr_param_grid = {
            'Logreg__C': [0.1, 1, 10]
        }

    lr = ['Logreg',lr_pipeline,lr_param_grid]

    xg_pipeline = Pipeline([
        ('xgboost', XGBClassifier(objective='multi:softmax',num_class=3, eval_metric='mlogloss', random_state=seed))
    ])

    xg_param_grid = {
        'xgboost__max_depth': [3, 5, 7],  # Profundidad máxima del árbol
        'xgboost__learning_rate': [0.1, 0.01, 0.001],  # Tasa de aprendizaje
        'xgboost__n_estimators': [100, 200, 500],  # Número de árboles en el bosque
    }

    xg = ['XGboost',xg_pipeline,xg_param_grid]
    
    lgbm_pipeline = Pipeline([
        ('lightgbm', LGBMClassifier(objective='multiclass',num_class=3, random_state=seed,is_unbalance=True))
    ])

    lgbm_param_grid = {
        'lightgbm__max_depth': [3, 5, 7],  # Profundidad máxima del árbol
        'lightgbm__learning_rate': [0.1, 0.01, 0.001],  # Tasa de aprendizaje
        'lightgbm__n_estimators': [100, 200, 500],  # Número de árboles en el bosque
        
    }
    lgbm = ['LGBM',lgbm_pipeline,lgbm_param_grid]
        
    
    rf_pipeline = Pipeline([
    ('random_forest', RandomForestClassifier(random_state=seed,class_weight="balanced"))])

    # Crear el grid de parámetros para Random Forest
    rf_param_grid = {
        'random_forest__n_estimators': [100, 500, 1000],  # Número de árboles en el bosque
        'random_forest__max_depth': [10, 20, 30],  # Profundidad máxima del árbol
        'random_forest__min_samples_leaf': [1, 2, 4],  # Número mínimo de muestras requeridas para estar en un nodo hoja
    }

    # Evaluar el modelo con la función model_evaluation
    rf = ['Random_Forest',rf_pipeline,rf_param_grid]
    
    cat_param_grid = {
        'cat__iterations': range(50, 201, 50),
        'cat__depth': range(1, 11)
    }
    
    cat_pipeline = Pipeline([
    ('cat',CatBoostClassifier(random_state=seed, loss_function='MultiClass',auto_class_weights='Balanced'))])
    
    cat = ['cat',cat_pipeline,cat_param_grid]
    
    models = [lr,xg,lgbm,rf,cat] #Activate to run all the models
    #models = [lr]
    return models