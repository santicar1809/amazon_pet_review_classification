from src.models.hyperparameters import all_models
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score
import joblib
import os
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

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
    results_nn=tens_flow(features_train,features_valid,target_train,target_valid)
    
    all_results=pd.concat([results_df,results_nn],ignore_index=True)
    return all_results
    
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

def build_model(data,num_clases):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(data,), kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        Dropout(0.3),  # Dropout for regularization
        Dense(64, activation='relu', input_shape=(data,), kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        #Dropout(0.3),  # Dropout for regularization
        Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        #Dropout(0.3),  # More dropout for regularization        
        Dense(16, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        #Dropout(0.3),  # More dropout for regularization        
        Dense(num_clases, activation='softmax')
    ])
    return model

def tens_flow(features_train,features_valid,target_train,target_valid):
    num_classes = len(set(target_train))
    seed=12345
    # Compiling the model
    model = build_model(features_train.shape[1],num_classes)
    optimizer = Adam(learning_rate=0.0005)
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.summary()
    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

    # Training the model using GPU if available
    with tf.device('/GPU:0'):  
        history = model.fit(features_train, target_train, epochs=200, batch_size=32, 
                            validation_data=(features_valid, target_valid), callbacks=[early_stopping])
    results=[]
    # Evaluating the model
    preds = model.predict(features_valid)
    preds_classes = preds.argmax(axis=1)
    acc=accuracy_score(target_valid,preds_classes)
    f1=f1_score(target_valid,preds_classes,average='macro')
    prec=precision_score(target_valid,preds_classes,average='macro')
    recall=recall_score(target_valid,preds_classes,average='macro')
    
    results.append(['Neural_network','',acc,f1,prec,recall])
    results_df=pd.DataFrame(results,columns=['model','best','accuracy_score','f1_score','precision','recall'])
    model.save('./models/nn.h5')
    return results_df