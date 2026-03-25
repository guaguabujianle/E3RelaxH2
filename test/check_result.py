# %%
import pandas as pd

df = pd.read_csv("./results/xmno_E3RelaxH2.csv")
for col in df.columns[1:]:
    print(f"{col}: ", df[col].mean().round(6))

# %%
