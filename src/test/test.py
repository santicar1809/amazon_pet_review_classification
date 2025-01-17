import pandas as pd
import joblib
from src.models.build_models import eval_model
import scipy.sparse
from src.models.hyperparameters import all_models

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
    results_df.to_csv('./files/output/results_test.csv',index=False)
    return results_df
test()