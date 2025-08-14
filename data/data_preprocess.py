import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import os
from matplotlib import font_manager, rcParams
from statsmodels.nonparametric.smoothers_lowess import lowess

print(os.listdir())
Processfile = 'FC2'

if Processfile == 'FC1':
    input_file_path = './data/FC1_Ageing.csv'
    output_file_path = './data/FC1_Ageing_processed.csv'
    smoothed_folder_path = './data/smoothedFC1/'
else:
    input_file_path = './data/FC2_Ageing.csv'
    output_file_path = './data/FC2_Ageing_processed.csv'
    smoothed_folder_path = './data/smoothedFC2/'

if not os.path.exists(input_file_path):
    raise FileNotFoundError(f"Input file not found: {input_file_path}")

df = pd.read_csv(input_file_path)

required_columns = ['Time (h)']
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Input data is missing required column: {col}")

df['Time (h)'] = pd.to_numeric(df['Time (h)'], errors='coerce')

df.dropna(subset=['Time (h)'], inplace=True)

df.sort_values('Time (h)', inplace=True)

df = df.drop_duplicates(subset='Time (h)', keep='last')

if df['Time (h)'].duplicated().any():
    raise ValueError("There are still duplicate 'Time (h)' values in the processed data.")

if input_file_path == './data/FC2_Ageing.csv':
    time_new = np.arange(0, 1021, 1)
else:
    time_new = np.arange(0, 1155, 1)

data_interp = {'Time': time_new}

col_mapping = {}

for col in df.columns:
    if col != 'Time (h)':
        if not np.issubdtype(df[col].dtype, np.number):
            print(f"Skipping non-numeric column: {col}")
            continue
        try:
            f = interp1d(df['Time (h)'], df[col], kind='linear', fill_value="extrapolate")
            interpolated_values = f(time_new)
            new_col_name = col.split(' (')[0]
            data_interp[new_col_name] = interpolated_values
            col_mapping[new_col_name] = col
        except Exception as e:
            print(f"Error interpolating column: {col}, Error: {e}")

df_interp = pd.DataFrame(data_interp)

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

plt.rcParams['axes.unicode_minus'] = False

if Processfile == 'FC1':
    frac = 20 / len(time_new)
else:
    frac = 10 / len(time_new)

if not os.path.exists(smoothed_folder_path):
    os.makedirs(smoothed_folder_path)
    print(f"Folder created: {smoothed_folder_path}")

columns_to_smooth = [col for col in df_interp.columns if col != 'Time']

df_smoothed = pd.DataFrame({'Time': df_interp['Time']})

for new_col in columns_to_smooth:
    original_col = col_mapping.get(new_col, None)
    if original_col is None:
        print(f"Original column name for '{new_col}' not found, skipping plot.")
        continue

    print(f"Smoothing column: {new_col}")

    try:
        smoothed = lowess(endog=df_interp[new_col], exog=df_interp['Time'], frac=frac, return_sorted=False)
        smoothed_rounded = np.round(smoothed, 5)
        df_smoothed[new_col] = smoothed_rounded

        plt.figure(figsize=(12, 6))

        plt.scatter(df['Time (h)'].values, df[original_col].values, label='Original Data', color='blue', s=10, alpha=0.5)

        plt.plot(df_interp['Time'].values, df_interp[new_col].values, label='Interpolated Data', color='red', linestyle='--')

        plt.plot(df_interp['Time'].values, smoothed_rounded, label='LOESS Smoothed Data', color='green', linewidth=2)

        plt.xlabel('Time (h)')
        plt.ylabel(original_col)
        plt.title(f'Comparison of Original, Interpolated, and LOESS Smoothed Data for {original_col}')

        plt.legend()
        plt.grid(True)

        plt.tight_layout()

        plot_file_path = os.path.join(smoothed_folder_path, f"{new_col}_smoothed_plot.png")
        plt.savefig(plot_file_path, dpi=300, bbox_inches='tight')
        print(f"Image for {new_col} saved to: {plot_file_path}")

        plt.close()

    except Exception as e:
        print(f"Error processing column '{new_col}': {e}")

df_smoothed.to_csv(output_file_path, index=False)
print(f"All smoothed data saved to: {output_file_path}")
