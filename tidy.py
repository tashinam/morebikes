import glob
import os
import pandas as pd

all_files = all_files = glob.glob(os.path.join('Train/Train', '*.csv'))
for f in all_files:
    print(f"Tidying #{f}...")
    df = pd.read_csv(f)
    # remove rows without target value. Should just be 1 timestamp when clocks changed.
    missing_bikes_row_count = df['bikes'].isna().sum()
    if missing_bikes_row_count > 1:
        raise RuntimeError("There should be at most 1 row with no bikes value per station")
    print(f"There are {missing_bikes_row_count} rows missing value for bikes. Removing these rows...")
    df = df[df['bikes'].isna() != True]
    # Drop precipidation column as all 0s    
    print(f"Precipitation column is all 0s. Removing...")
    df.drop("precipitation.l.m2", axis=1)
    
    # Save tided file to processed directory
    new_file = os.path.join('Processed', f'{os.path.basename(f)}')
    print(f"Saving tidied data to {new_file}...")
    df.to_csv(new_file)
