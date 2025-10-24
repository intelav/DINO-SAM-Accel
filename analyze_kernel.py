import pandas as pd

# Load kernel stats, skipping the header junk
df = pd.read_csv("kernel_stats.csv", skiprows=6)

# Clean up column names (strip spaces)
df.columns = df.columns.str.strip()

# Ensure numeric conversion
df["Total Time (ns)"] = pd.to_numeric(df["Total Time (ns)"], errors="coerce")
df["Time (%)"] = pd.to_numeric(df["Time (%)"], errors="coerce")

# Rank by contribution to total time
df_sorted = df.sort_values(by="Time (%)", ascending=False)

print("Top 10 kernels by total time share:")
print(df_sorted[["Time (%)", "Total Time (ns)", "Instances", "Name"]].head(10))
