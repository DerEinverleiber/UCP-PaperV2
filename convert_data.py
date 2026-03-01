import pandas as pd
import io

"""
!Note that labels like "Bus x" will be separated as [..., 'Bus', 'x', ...], so this implementation does not respect whitespace 
spearated names and thus changes the shape. Therefore, it currently only works for branch data!

This script takes 'data/ieee57cdf.txt' and converts it into 2 individual csv files
for bus and branch data, respectively. 
The generated csv files are stored as 'data/ieee57_bus.csv' and 'data/ieee57_branch.csv'.
For later evaluation, only a subset of columns will be considered.
"""

def parse_ieee_57(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Identify indices for data sections [cite: 101, 143]
    bus_start = next(i for i, line in enumerate(lines) if "BUS DATA FOLLOWS" in line) + 1
    branch_start = next(i for i, line in enumerate(lines) if "BRANCH DATA FOLLOWS" in line) + 1
    
    # Extract raw lines until the -999 terminator 
    bus_lines = []
    for line in lines[bus_start:]:
        if line.strip().startswith("-999"): break
        bus_lines.append(line)
        
    branch_lines = []
    for line in lines[branch_start:]:
        if line.strip().startswith("-999"): break
        branch_lines.append(line)

    # Define Columns based on IEEE CDF standard
    bus_cols = ['Bus_Num', 'Name', 'Type', 'V_Mag', 'V_Angle', 'Load_MW', 'Load_MVAR', 
                'Gen_MW', 'Gen_MVAR', 'Base_KV', 'Desired_V', 'Max_MVAR', 'Min_MVAR']
    
    branch_cols = ['Tap_Bus', 'Z_Bus', 'Area', 'Loss_Zone', 'Circuit', 'Type', 
                   'R', 'X', 'B', 'Rate_A', 'Rate_B', 'Rate_C', 'Ratio']

    # Load into DataFrames using whitespace as separator
    # We limit columns to the core data fields for clarity
    df_bus = pd.read_csv(io.StringIO("".join(bus_lines)), sep=r'\s+', names=bus_cols, usecols=range(13))
    df_branch = pd.read_csv(io.StringIO("".join(branch_lines)), sep=r'\s+', names=branch_cols, usecols=range(13))

    return df_bus, df_branch

# Execution
bus_df, branch_df = parse_ieee_57('data/ieee57cdf.txt')

# Save to CSV for convenience
bus_df.to_csv('data/ieee57_bus.csv', index=False)
branch_df.to_csv('data/ieee57_branch.csv', index=False)

print("Shape of ieee57_bus.csv is irregular!")
print("Conversion complete. Preview of Bus Data:")
print(bus_df.head())