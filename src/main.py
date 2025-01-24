from src.preprocessing.load_dataset import load_dataset
from src.preprocessing.preprocessing import preprocessing
from src.eda.eda import eda
from src.feature_engineer_s.text_processing import feature_engineer
from src.models.build_models import build_models
from src.feature_engineer_s.fasttext import fast_text

def main():
    data=load_dataset()
    preprocessed_data=preprocessing(data)
    eda(preprocessed_data)
    fast_text(preprocessed_data)
    X_train,X_valid,y_train,y_valid=feature_engineer(preprocessed_data)
    results=build_models(X_train,X_valid,y_train,y_valid)
    
    return results

main()
