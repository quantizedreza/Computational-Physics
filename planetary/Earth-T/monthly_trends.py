import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import logging

logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

FILENAME = 'data.csv'
TARGET_MONTH = 2                   # ← change here (1 = Jan, 2 = Feb, ..., 9 = Sep, etc.)
XMIN, XMAX = 1990, 2026
XTICK_STEP = 5
FIGURE_SIZE = (10, 5.5)              # width, height in inches
DPI = 150

df = pd.read_csv(FILENAME)
df.columns = df.columns.str.strip()

# Find month information and filter desired month
month_col = None
df_month = None

if 'Month' in df.columns:
    month_col = 'Month'
    df_month = df[df[month_col].astype(str).str.strip() == str(TARGET_MONTH)]

elif 'month' in df.columns.str.lower():
    month_col = df.columns[df.columns.str.lower().str.contains('month')][0]
    df_month = df[df[month_col].astype(str).str.strip() == str(TARGET_MONTH)]

elif any('date' in c.lower() for c in df.columns):
    date_col = next(c for c in df.columns if 'date' in c.lower())
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df_month = df[df[date_col].dt.month == TARGET_MONTH]

else:
    print("Could not find a usable month/date column. Please check CSV structure.")
    df_month = df  # fallback — will most likely fail later

if 'Year' not in df_month.columns:
    raise ValueError("Column 'Year' not found in the dataset.")

annual_means = (
    df_month.groupby('Year', as_index=False)['Temperature']
            .mean()
            .rename(columns={'Temperature': 'Mean_Temperature'})
)

annual_means['Mean_Temperature'] = annual_means['Mean_Temperature'].round(2)

years = annual_means['Year']
temps = annual_means['Mean_Temperature']

slope, intercept = np.polyfit(years, temps, 1)
trend_line = slope * years + intercept

trend_per_year = slope
trend_text = f"Linear trend: {trend_per_year:.2f} °F/year"

plt.style.use('default')   

fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=DPI)

ax.plot(years, temps,
        marker='o', markersize=8, color='#1f77b4',
        linestyle='none', label=f'{TARGET_MONTH}- Average Temperature')

ax.plot(years, trend_line,
        color='#d62728', linewidth=2.2, linestyle='--',
        label=trend_text)

ax.set_title(f"U.S. Average Temperature – Month {TARGET_MONTH:02d} \n Data: NOAA", 
             fontsize=15, pad=14, weight='medium')
ax.set_xlabel("Year", fontsize=12)
ax.set_ylabel("Mean Temperature (°F)", fontsize=12)

ax.set_xlim(XMIN - 1, XMAX + 1)
ax.set_xticks(np.arange(XMIN, XMAX + 1, XTICK_STEP))

ax.grid(True, linestyle=':', alpha=0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.legend(frameon=True, fontsize=11, loc='upper left')

plt.savefig(f"us_temp_month_{TARGET_MONTH:02d}.png", dpi=300)

plt.tight_layout()
plt.show()

