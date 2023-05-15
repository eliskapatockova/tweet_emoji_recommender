import os
import emoji
import pandas as pd
from helper_functions import fetch_dataset

def join_dataframes():
    dirname = os.path.join(os.getcwd(), 'filtered_data')
    ext = ('.csv')

    files = []
    for file in os.listdir(dirname):
        if file.endswith(ext):
            files.append(file)
        else:
            continue
    frames = []
    for f in files:
        frames.append(pd.read_csv(os.path.join(dirname, f)))
    # frames = [ process_your_file(f) for f in files ]
    # result = pd.concat(frames)
    df = pd.concat(frames)
    df = df.iloc[2:, :]

    # Save joined dataframes as CSV
    output_path = os.path.join(os.getcwd(), 'joined_dataframes.csv')
    df.to_csv(output_path, index=False)
    print(f"Joined dataframes saved as: {output_path}")

    return df
