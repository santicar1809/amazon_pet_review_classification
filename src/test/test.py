import pandas as pd
import joblib
from src.models.build_models import eval_model
import scipy.sparse
from src.models.hyperparameters import all_models
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score

def eval_nn(model,features_valid,target_valid):
    preds = model.predict(features_valid)
    preds_classes = preds.argmax(axis=1)
    acc=accuracy_score(target_valid,preds_classes)
    f1=f1_score(target_valid,preds_classes,average='macro')
    prec=precision_score(target_valid,preds_classes,average='macro')
    recall=recall_score(target_valid,preds_classes,average='macro')
    return acc,f1,prec,recall

def test():
    features_test = scipy.sparse.load_npz('./files/output/features_test.npz')
    target=pd.read_csv('./files/output/target_test.csv')
    target_test=target['sentiment']
    results=[]
    models=all_models()
    for model in models:
        test_model=joblib.load(f'./models/{model[0]}.joblib')
        acc,f1,precision,recall=eval_model(test_model,features_test,target_test)
        results.append([model[0],acc,f1,precision,recall])
    results_df=pd.DataFrame(results,columns=['model','accuracy_score','f1_score','precision','recall'])
    
    results_nn=[]
    model_nn = load_model('./models/nn.h5')
    acc,f1,precision,recall=eval_nn(model_nn,features_test,target_test)
    results_nn.append(['nn',acc,f1,precision,recall])
    results_df_nn=pd.DataFrame(results_nn,columns=['model','accuracy_score','f1_score','precision','recall'])
    results_data=pd.concat([results_df,results_df_nn])
    results_data.to_csv('./files/output/results_test.csv',index=False)
    
    return results_data
test()