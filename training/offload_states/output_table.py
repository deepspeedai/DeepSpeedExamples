import pandas as pd
from pytablewriter import MarkdownTableWriter


def read_csv(file_path):
    return pd.read_csv(file_path)

df = read_csv('offload_states.log')
df.columns = ['pin_memory', 'non_blocking', 'offload_time', 'load_time']

df['ratio_string'] = df['offload_time'].round(2).astype(str) + " / " + df['load_time'].round(2).astype(str)

result_df = pd.DataFrame({
    'pin_memory=0_non_blocking=0': df[(df['pin_memory'] == 0) & (df['non_blocking'] == 0)]['ratio_string'].reset_index(drop=True),
    'pin_memory=0_non_blocking=1': df[(df['pin_memory'] == 0) & (df['non_blocking'] == 1)]['ratio_string'].reset_index(drop=True),
    'pin_memory=1_non_blocking=0': df[(df['pin_memory'] == 1) & (df['non_blocking'] == 0)]['ratio_string'].reset_index(drop=True),
    'pin_memory=1_non_blocking=1': df[(df['pin_memory'] == 1) & (df['non_blocking'] == 1)]['ratio_string'].reset_index(drop=True)
})
result_df = result_df.dropna()
result_df.index = range(1, len(result_df) + 1)
result_df.index.name = 'trial'
# print(result_df)

writer = MarkdownTableWriter()
writer.from_dataframe(result_df,
    add_index_column=True,
)
writer.write_table()