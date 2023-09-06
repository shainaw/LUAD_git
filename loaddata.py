import csv
import pandas as pd
import os

ic50_df = pd.read_csv('raw_data/LUAD_IC_Sun Aug 20 23_42_44 2023.csv')
mutation_df = pd.read_csv('raw_data/LUAD_Genetic_features_Sun Aug 20 23_31_09 2023.csv')

cell_line_names_mutation = mutation_df['Cell Line Name'].unique()
cell_line_names_ic50 = ic50_df['Cell Line Name'].unique()

shared_names = [cell_name for cell_name in cell_line_names_mutation if cell_name in cell_line_names_ic50]

mutation_lists = mutation_df['Genetic Feature'].unique()

column_names = ['Cell Line Name','Drug Name','Tissue Sub-type','IC50','AUC']+mutation_lists.tolist()

data = pd.DataFrame(columns=column_names)

for cellname in shared_names:
    rows_mutation = mutation_df [ mutation_df['Cell Line Name'] == cellname ]
    rows_ic50 = ic50_df [ ic50_df['Cell Line Name'] == cellname ]
    
    new_row_cellinfo = {}
        
    new_row_cellinfo['Cell Line Name'] = cellname
    new_row_cellinfo['Tissue Sub-type']= rows_ic50.iloc[0]['Tissue Sub-type']
    

    
    for row in rows_mutation.itertuples(index=False):
        new_row_cellinfo[row[5]] = row[6]
        
        
    for drug_row in rows_ic50.itertuples(index=False):
        new_row = dict(new_row_cellinfo)
        
        new_row['Drug Name'] = drug_row[0]
        new_row['IC50'] = drug_row[7]
        new_row['AUC'] = drug_row[8]
        
        data = data.append(new_row, ignore_index=True)
        
os.makedirs("data",exist_ok=True)

data.to_csv('data/datafile.csv', index=False)