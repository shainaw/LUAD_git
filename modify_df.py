import pandas as pd
import numpy as np

def ic50_binary(df):
    df['IC50_binary'] = df['IC50'].apply(lambda x: 1 if x >0 else 0)
    df = df.drop(['Tissue Sub-type','IC50','AUC'],axis = 1)
    
    return df


def split_df_by_cell_line(df,k):
    '''
    Takes in df, splits data based on cell line
    
    '''
    cell_line_names = df['Cell Line Name'].unique()
    
    np.random.shuffle(cell_line_names)
    
    train, test = [], []

    train_cellline = set(cell_line_names[:int(k*len(cell_line_names))])
    test_cellline = set(cell_line_names[int(k*len(cell_line_names)):])
    
 
    for _, row in df.iterrows():
        if row['Cell Line Name'] in train_cellline:
            train.append(row)
        else:
            test.append(row)
    
    train_df = pd.DataFrame(train)
    test_df = pd.DataFrame(test)
    
    return train_df, test_df


def split_df_by_drugname(df,k):
    '''
    Takes in df, splits data based on drug name
    
    '''
    cell_line_names = df['Drug Name'].unique()
    
    np.random.shuffle(cell_line_names)
    
    train, test = [], []

    train_cellline = set(cell_line_names[:int(k*len(cell_line_names))])
    test_cellline = set(cell_line_names[int(k*len(cell_line_names)):])
    
 
    for _, row in df.iterrows():
        if row['Drug Name'] in train_cellline:
            train.append(row)
        else:
            test.append(row)
    
    train_df = pd.DataFrame(train)
    test_df = pd.DataFrame(test)
    
    return train_df, test_df