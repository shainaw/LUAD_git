import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score

import joblib

import random

def calculate_feature_importance(X:pd.DataFrame,y:pd.DataFrame,dir:str):


    
    y_pred_base = joblib.load(f'{dir}/baseline.pkl').predict_proba(X)[:, 1]

    base_auc = roc_auc_score(y,y_pred_base)
    
    df_raw_roc_score = pd.DataFrame(index=X.columns, columns=['ROC_AUC'])
    df_z_score = pd.DataFrame(index=X.columns, columns=['z-score'])
    
    for feature in X.columns:
        
        x_without_feat = X.drop(columns=[feature])
        
        model_without_feat = joblib.load(f'{dir}/without_{feature}.pkl')
        
        y_pred_without_feat = model_without_feat.predict_proba(x_without_feat)[:, 1]
        
        auc_without_feat = roc_auc_score(y, y_pred_without_feat)
        
        df_raw_roc_score.loc[feature, 'ROC_AUC'] = auc_without_feat
        
        
        
    sd_auc_change = (base_auc-df_raw_roc_score['ROC_AUC']).std()
    
    for feature in df_raw_roc_score.index:
        auc_score = df_raw_roc_score.loc[feature,'ROC_AUC']
        z_score = (base_auc-auc_score)/(sd_auc_change)

        df_z_score.loc[feature,"z-score"] = z_score
        
        
    df_z_score['z-score'] = pd.to_numeric(df_z_score['z-score'], errors='coerce')
    
    return df_z_score






if __name__ == '__main__':
    
    random_data = pd.DataFrame({'x1':[random.random() for _ in range(20)],
                                'x2':[random.random() for _ in range(20)],
                                'x3':[random.random() for _ in range(20)],
                                'x4':[random.random() for _ in range(20)],
                                'x5':[random.random() for _ in range(20)],
                                'y':[0 if x < 0.5 else 1 for x in [random.random() for _ in range(20)]]})
    
    x_data = random_data.drop(['y'],axis = 1)
    y_data= random_data['y']

    assert(isinstance(calculate_feature_importance(x_data,y_data,'./models_made/test/gb'), pd.DataFrame))
    
    