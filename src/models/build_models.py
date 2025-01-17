from src.models.hyperparameters import all_models
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score
import joblib
import os
import pandas as pd


def build_models(features_train,features_valid,target_train,target_valid):
    models = all_models()
    if not os.path.exists('./models/'):
        os.makedirs('./models/')
    models_path='./models/'
    results=[]
    for model in models:
        best,accuracy,f1,precision,recall=model_structure(features_train,features_valid,target_train,target_valid,model[1],model[2])
        joblib.dump(best,models_path+f'{model[0]}.joblib')
        results.append([model[0],best,accuracy, f1,precision,recall])
    results_df=pd.DataFrame(results,columns=['model','best','accuracy_score','f1_score','precision','recall'])
    results_df.to_csv('./files/output/results_train.csv',index=False)
    
    return results_df
    
def eval_model(best,features_valid,target_valid):
    preds=best.predict(features_valid)
    acc=accuracy_score(target_valid,preds)
    f1=f1_score(target_valid,preds,average='macro')
    prec=precision_score(target_valid,preds,average='macro')
    recall=recall_score(target_valid,preds,average='macro')
    return acc,f1,prec,recall
    
def model_structure(features_train,features_valid,target_train,target_valid,model,parameters):
    
    seed=42
    gs=GridSearchCV(model,parameters,cv=2,scoring='accuracy',n_jobs=-1,verbose=2)
    gs.fit(features_train,target_train)
    best_estimator=gs.best_estimator_
    accuracy,f1,precision,recall =eval_model(best_estimator,features_valid,target_valid)
    
    return best_estimator,accuracy,f1,precision,recall
