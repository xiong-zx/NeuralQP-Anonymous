# %%
import pandas as pd
from pathlib import Path

root = Path.cwd().parent
files = [d for d in root.glob('runs/testNP/*/*/*/results.csv')]
# %%
all_df = pd.DataFrame()
for file in files:
    train_info, test_info = file.parts[-4:-2]
    df = pd.read_csv(file, index_col=0)
    df['train_problem'] = train_info.split('-')[0]
    df['train_difficulty'] = train_info.split('-')[1]
    df['test_problem'] = test_info.split('-')[0]
    df['test_difficulty'] = test_info.split('-')[1]
    all_df = pd.concat([all_df, df])
all_df = all_df[['train_problem','train_difficulty','test_problem','test_difficulty','name','init_obj','init_runtime','gurobi_obj','gurobi_runtime']]
all_df.to_csv(root / 'results' / 'NP_results_raw.csv', index=False)
# %%
