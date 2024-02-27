import pandas as pd
 
num_files = 100
base_filename = 'comp_id_{}.csv'
dataframes = []
for i in range(num_files):
    filename = base_filename.format(i)
    df=pd.read_csv(filename,header=0)
    dataframes.append(df)
concatenated_df = pd.concat(dataframes, ignore_index=True)
concatenated_df.to_csv('concatenated_comp_ids.csv',index=False)