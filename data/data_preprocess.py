import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import os
from matplotlib import font_manager, rcParams
from statsmodels.nonparametric.smoothers_lowess import lowess

print(os.listdir())
# Define file paths
Processfile = 'FC2'

if Processfile == 'FC1':
    input_file_path = './data/FC1_Ageing.csv'
    output_file_path = './data/FC1_Ageing_processed.csv'
    smoothed_folder_path = './data/smoothedFC1/'  # Define folder path to save smoothed result images
else:
    input_file_path = './data/FC2_Ageing.csv'
    output_file_path = './data/FC2_Ageing_processed.csv'
    smoothed_folder_path = './data/smoothedFC2/'  # Define folder path to save smoothed result images

# Check if input file exists
if not os.path.exists(input_file_path):
    raise FileNotFoundError(f"Input file not found: {input_file_path}")

# Load CSV file
df = pd.read_csv(input_file_path)

# Check if it contains 'Time (h)' and at least one other column
required_columns = ['Time (h)']
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Input data is missing required column: {col}")

# Convert 'Time (h)' to float, ensuring there are no non-numeric values
df['Time (h)'] = pd.to_numeric(df['Time (h)'], errors='coerce')

# Drop rows containing NaN
df.dropna(subset=['Time (h)'], inplace=True)

# Sort by 'Time (h)'
df.sort_values('Time (h)', inplace=True)

# Drop duplicate time points, keep the last value
df = df.drop_duplicates(subset='Time (h)', keep='last')

# Ensure 'Time (h)' is unique
if df['Time (h)'].duplicated().any():
    raise ValueError("There are still duplicate 'Time (h)' values in the processed data.")

# Create new time points array from 0 to 1154, one point per hour
if input_file_path == './data/FC2_Ageing.csv':
    time_new = np.arange(0, 1021, 1)
else:
    time_new = np.arange(0, 1155, 1)
# Initialize dictionary to store interpolated data
data_interp = {'Time': time_new}

# Initialize column name mapping dictionary
col_mapping = {}  # new_col_name -> original_col_name

# Interpolate each column (skip 'Time (h)')
for col in df.columns:
    if col != 'Time (h)':
        # Check if column is numeric
        if not np.issubdtype(df[col].dtype, np.number):
            print(f"Skipping non-numeric column: {col}")
            continue
        try:
            # Create interpolation function
            f = interp1d(df['Time (h)'], df[col], kind='linear', fill_value="extrapolate")
            # Compute interpolated values
            interpolated_values = f(time_new)
            # Add to dictionary, remove units
            new_col_name = col.split(' (')[0]
            data_interp[new_col_name] = interpolated_values
            # Record column name mapping
            col_mapping[new_col_name] = col
        except Exception as e:
            print(f"Error interpolating column: {col}, Error: {e}")

# Create interpolated DataFrame
df_interp = pd.DataFrame(data_interp)

# Round data to five decimal places
df_interp = df_interp.round(5)
possible_fonts = ['Hiragino Sans GB', 'PingFang SC', 'Heiti SC', 'STHeiti', 'Hiragino Sans']
available_fonts = set(f.name for f in font_manager.fontManager.ttflist)
selected_font = None
for font in possible_fonts:
    if font in available_fonts:
        selected_font = font
        break

if selected_font is None:
    print("Specified Chinese font not found")
else:
    plt.rcParams['font.sans-serif'] = [selected_font]
    plt.rcParams['font.family'] = 'sans-serif'
    print(f"Chinese font set to: {selected_font}")

# Fix the issue of minus sign '-' displaying as a square
plt.rcParams['axes.unicode_minus'] = False

# Apply LOESS smoothing
# Window width is 20 hours, frac is window width divided by total data points
if Processfile == 'FC1':
    frac = 20 / len(time_new)
else:
    frac = 10 / len(time_new)

# Create folder to save smoothed results (if it does not exist)
if not os.path.exists(smoothed_folder_path):
    os.makedirs(smoothed_folder_path)
    print(f"Folder created: {smoothed_folder_path}")

# Get all columns to be smoothed (all columns except 'Time')
columns_to_smooth = [col for col in df_interp.columns if col != 'Time']

# Initialize a new DataFrame to store all smoothed data
df_smoothed = pd.DataFrame({'Time': df_interp['Time']})

# Iterate through each column to be smoothed, apply LOESS, and save comparison images
for new_col in columns_to_smooth:
    original_col = col_mapping.get(new_col, None)
    if original_col is None:
        print(f"Original column name for '{new_col}' not found, skipping plot.")
        continue

    print(f"Smoothing column: {new_col}")

    try:
        # Apply LOESS smoothing
        smoothed = lowess(endog=df_interp[new_col], exog=df_interp['Time'], frac=frac, return_sorted=False)
        smoothed_rounded = np.round(smoothed, 5)
        # Add to smoothed DataFrame
        df_smoothed[new_col] = smoothed_rounded

        # Generate and save comparison image
        plt.figure(figsize=(12, 6))

        # Plot original data (using original DataFrame df)
        plt.scatter(df['Time (h)'].values, df[original_col].values, label='Original Data', color='blue', s=10, alpha=0.5)

        # Plot interpolated data
        plt.plot(df_interp['Time'].values, df_interp[new_col].values, label='Interpolated Data', color='red', linestyle='--')

        # Plot LOESS smoothed data
        plt.plot(df_interp['Time'].values, smoothed_rounded, label='LOESS Smoothed Data', color='green', linewidth=2)

        # Set labels and title
        plt.xlabel('Time (h)')
        plt.ylabel(original_col)
        plt.title(f'Comparison of Original, Interpolated, and LOESS Smoothed Data for {original_col}')

        # Add legend and grid
        plt.legend()
        plt.grid(True)

        # Adjust layout
        plt.tight_layout()

        # Save image to smoothed folder
        plot_file_path = os.path.join(smoothed_folder_path, f"{new_col}_smoothed_plot.png")
        plt.savefig(plot_file_path, dpi=300, bbox_inches='tight')
        print(f"Image for {new_col} saved to: {plot_file_path}")

        # Close figure to free memory
        plt.close()

    except Exception as e:
        print(f"Error processing column '{new_col}': {e}")

# Save all smoothed data to a new CSV file
df_smoothed.to_csv(output_file_path, index=False)
print(f"All smoothed data saved to: {output_file_path}")