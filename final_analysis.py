# DSA210 - Screen Time vs Spending Analysis

import os
from pathlib import Path
from datetime import datetime
from mpl_toolkits.mplot3d import Axes3D   

import numpy as np
import pandas as pd
import seaborn as sns              
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression

# Output Settings
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams.update({'axes.labelsize': 12,
                     'axes.titlesize': 14,
                     'xtick.labelsize': 10,
                     'ytick.labelsize': 10})

OUTPUT_DIR = Path('analysis_output')
OUTPUT_DIR.mkdir(exist_ok=True)
print(f"Results will be saved in the {OUTPUT_DIR.resolve()} folder.\n")

# Uploading the Data
def load_data():
    """Read, clean, and merge the CSV files."""
    print("Loading data...")

    screen_fp = Path('data/screentime.csv')
    spend_fp  = Path('data/spending.csv')

    if not screen_fp.exists() or not spend_fp.exists():
        raise FileNotFoundError("screentime.csv and/or spending.csv not found in the data folder.")

    screen_df = pd.read_csv(screen_fp, sep=';')
    spending_df = pd.read_csv(spend_fp, sep=';')
    spending_df.columns = spending_df.columns.str.replace('_', ' ')

    # Converting the dates
    screen_df['Date']   = pd.to_datetime(screen_df['Date'],   format='%d.%m.%Y')
    spending_df['Date'] = pd.to_datetime(spending_df['Date'], format='%d.%m.%Y')

    # Minute data to number
    screen_df['Time'] = pd.to_numeric(screen_df['Time'], errors='coerce')

    # Daily total screen time
    daily_screen = screen_df.groupby('Date', as_index=False)['Time'].sum().rename(columns={'Time': 'TotalScreenTime'})

    # Daily total spending
    spending_df['Total Spending'] = pd.to_numeric(spending_df['Total Spending'], errors='coerce')
    daily_spend = spending_df.groupby('Date', as_index=False)['Total Spending'].sum().rename(columns={'Total Spending': 'TotalSpending'})

    # Merge
    merged = pd.merge(daily_screen, daily_spend, on='Date', how='inner')

    # Weekdays & Weekends
    merged['DayOfWeek'] = merged['Date'].dt.day_name()
    merged['IsWeekend'] = merged['Date'].dt.dayofweek >= 5

    print(f"Merged dataset: {merged.shape[0]} days, {merged.shape[1]} columns")
    return merged

# Statistics & Graphs                  

def descriptive_stats(df: pd.DataFrame):
    """Generate basic distribution plots and summary statistics."""
    print("\n▶ Descriptive Statistics")

    desc = df[['TotalScreenTime', 'TotalSpending']].describe().T.round(2)
    print(desc, end="\n\n")

    # Histogram – Spending
    plt.figure(figsize=(8, 4))
    sns.histplot(df['TotalSpending'], bins=15, kde=False)
    plt.title('Daily Spending Distribution')
    plt.xlabel('Spending (TL)')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'hist_daily_spending.png')
    plt.close()

    # Histogram – Screen Time
    plt.figure(figsize=(8, 4))
    sns.histplot(df['TotalScreenTime'], bins=15, kde=False, color='orange')
    plt.title('Daily Total Screen Time Distribution')
    plt.xlabel('Screen Time (min)')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'hist_screen_time.png')
    plt.close()

    # Scatter – Screen Time vs Spending
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x='TotalScreenTime', y='TotalSpending', data=df, hue='IsWeekend', palette='viridis')
    sns.regplot(x='TotalScreenTime', y='TotalSpending', data=df, scatter=False, ci=None, color='red', line_kws={'linestyle': '--'})    
    plt.xlabel('Total Screen Time (min)')
    plt.ylabel('Spending (TL)')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'scatter_screen_vs_spend.png')
    plt.close()

# Correlation & Hypotheses Tests
def correlation_and_tests(df: pd.DataFrame):
    print("\n▶ Correlation and Hypothesis Tests")

    # Pearson & Spearman
    pearson_r, pearson_p = stats.pearsonr(df['TotalScreenTime'], df['TotalSpending'])
    spearman_r, spearman_p = stats.spearmanr(df['TotalScreenTime'], df['TotalSpending'])

    print(f"Pearson r = {pearson_r:.3f} (p={pearson_p:.4f})")
    print(f"Spearman ρ = {spearman_r:.3f} (p={spearman_p:.4f})")

    # Median-based two-group test
    median_sc = df['TotalScreenTime'].median()
    high = df[df['TotalScreenTime'] >  median_sc]['TotalSpending']
    low  = df[df['TotalScreenTime'] <= median_sc]['TotalSpending']

    print(f"\nMedian screen time = {median_sc:.0f} min")
    print(f"High screen time days: {len(high)}, Low: {len(low)}")

    t_stat, t_p = stats.ttest_ind(high, low, equal_var=False)
    mw_stat, mw_p = stats.mannwhitneyu(high, low, alternative='two-sided')

    print(f"t‑test:  t={t_stat:.3f}, p={t_p:.4f}")
    print(f"Mann‑Whitney U: U={mw_stat:.1f}, p={mw_p:.4f}")

# Weekdays / Weekends
def weekend_analysis(df: pd.DataFrame):
    print("\n▶ Weekday vs Weekend Spending Analysis")

    weekday_spend = df[df['IsWeekend'] == False]['TotalSpending']
    weekend_spend = df[df['IsWeekend'] == True ]['TotalSpending']

    mean_weekday = weekday_spend.mean()
    mean_weekend = weekend_spend.mean()

    print(f"- Weekday avg. spending : {mean_weekday:.2f} TL")
    print(f"- Weekend avg. spending: {mean_weekend:.2f} TL")

    plt.figure(figsize=(6, 4))
    sns.boxplot(x='IsWeekend', y='TotalSpending', data=df)
    plt.xticks([0, 1], ['Weekday', 'Weekend'])
    plt.title('Weekday vs Weekend Spending Comparison')
    plt.xlabel('')
    plt.ylabel('Spending (TL)')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'box_weekday_weekend_spending.png')
    plt.close()

    u_stat, u_p = stats.mannwhitneyu(weekday_spend, weekend_spend, alternative='two-sided')
    print(f"Mann‑Whitney U testi: U={u_stat:.1f}, p={u_p:.4f}")

# Extra Visuals      
def extra_visuals(df: pd.DataFrame):
    """Generate additional visualizations to enrich the project output."""
    print("\n▶ Generating additional visualizations...")

    plt.figure(figsize=(12, 6))
    ax1 = plt.gca()
    ax2 = ax1.twinx()

    ax1.plot(df['Date'], df['TotalScreenTime'], label='Ekran Süresi (dk)', color='darkorange', linewidth=2)
    ax2.plot(df['Date'], df['TotalSpending'], label='Harcama (TL)', color='steelblue', linewidth=2, alpha=0.7)

    ax1.set_ylabel('Screen Time (min)')
    ax2.set_ylabel('Spending (TL)')
    ax1.set_title('Time Series: Screen Time & Spending')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'timeseries_screen_spend.png')
    plt.close()

    order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    plt.figure(figsize=(10, 5))
    sns.boxplot(x='DayOfWeek', y='TotalSpending', data=df, order=order, hue='DayOfWeek', palette='Pastel1', legend=False)
    plt.title('Spending Distribution – Days of the Week')
    plt.xlabel('Day')
    plt.ylabel('Spending (TL)')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'box_spend_by_dayofweek.png')
    plt.close()

    plt.figure(figsize=(10, 5))
    sns.violinplot(x='DayOfWeek', y='TotalScreenTime', data=df, order=order, hue='DayOfWeek', palette='Pastel2', legend=False)
    plt.title('Screen Time Distribution – Days of the Week')
    plt.xlabel('Day')
    plt.ylabel('Screen Time (min)')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'violin_screen_by_dayofweek.png')
    plt.close()

    df_sorted = df.sort_values('Date').set_index('Date')
    roll_sc = df_sorted['TotalScreenTime'].rolling(window=7, min_periods=3).mean()
    roll_sp = df_sorted['TotalSpending'].rolling(window=7, min_periods=3).mean()

    plt.figure(figsize=(12, 6))
    plt.plot(roll_sc, label='7-day Avg. Screen Time', color='darkorange')
    plt.plot(roll_sp, label='7-day Avg. Spending', color='steelblue')
    plt.title('7-Day Moving Averages')
    plt.xlabel('Date')
    plt.ylabel('Value (scales are independent)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'rolling_means.png')
    plt.close()


# App-based visuals
def app_usage_visuals():
    """
    Analyzes and visualizes application/category-based usage durations
    from screentime.csv.
    """
    print("\n▶ Generating app-based visualizations...")

    screen_fp = Path('data/screentime.csv')
    if not screen_fp.exists():
        print("screentime.csv not found, skipping app-based visualizations.")
        return

    sc_df = pd.read_csv(screen_fp, sep=';')

    sc_df['Date'] = pd.to_datetime(sc_df['Date'], format='%d.%m.%Y')
    sc_df['Time'] = pd.to_numeric(sc_df['Time'], errors='coerce')

    app_total = (sc_df.groupby('Application_Name')['Time']
                 .sum()
                 .sort_values(ascending=False)
                 .head(10))

    plt.figure(figsize=(10, 5))
    sns.barplot(x=app_total.values, y=app_total.index, color='skyblue')
    plt.ylabel('')
    plt.title('Top 10 Most Used Applications')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'top10_apps.png')
    plt.close()

    if 'Category' in sc_df.columns:
        cat_total = (sc_df.groupby('Category')['Time']
                     .sum()
                     .sort_values(ascending=False))

        plt.figure(figsize=(6, 6))
        plt.pie(cat_total.values,
                labels=cat_total.index,
                autopct='%1.1f%%',
                startangle=140)
        plt.title('Total Usage Share by Category')
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'category_pie.png')
        plt.close()

    top3_apps = app_total.index[:3]
    top3_df   = (sc_df[sc_df['Application_Name'].isin(top3_apps)]
                 .groupby(['Date', 'Application_Name'])['Time']
                 .sum()
                 .reset_index())

    plt.figure(figsize=(12, 6))
    for app in top3_apps:
        subset = top3_df[top3_df['Application_Name'] == app]
        plt.plot(subset['Date'], subset['Time'], marker='o', label=app)

    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('Daily Usage Time (min)')
    plt.title('Daily Usage Trends of Top 3 Apps')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'top3_apps_timeseries.png')
    plt.close()

    print("App-based visualizations have been saved.")

# Correlation Heatmap & 3D Influential Factors Plot
def correlation_heatmap(df: pd.DataFrame):
    """Draws a correlation heatmap among numerical variables."""
    print("\n▶ Generating correlation heatmap...")

    tmp = df.copy()

    tmp['DayIndex']     = tmp['Date'].dt.dayofweek          
    tmp['IsWeekendInt'] = tmp['IsWeekend'].astype(int)      

    numeric_cols = ['TotalScreenTime', 'TotalSpending', 'DayIndex', 'IsWeekendInt']
    corr_mat = tmp[numeric_cols].corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_mat,
                annot=True, fmt=".2f",
                cmap="coolwarm", vmin=-1, vmax=1,
                linewidths=0.5)
    plt.title('Correlation Matrix Between Variables')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'correlation_heatmap.png')
    plt.close()


def influential_3d_plot(df: pd.DataFrame):
    """Displays the two most influential features in a linear model using a 3D scatter plot."""
    print("\n▶ Preparing 3D plot for the most influential factors...")

    tmp = df.copy()
    tmp['DayIndex']     = tmp['Date'].dt.dayofweek
    tmp['IsWeekendInt'] = tmp['IsWeekend'].astype(int)

    feature_cols = ['TotalScreenTime', 'DayIndex', 'IsWeekendInt']
    X = tmp[feature_cols].values
    y = tmp['TotalSpending'].values

    model = LinearRegression().fit(X, y)
    abs_coefs = np.abs(model.coef_)
    top2_idx  = np.argsort(abs_coefs)[-2:][::-1]        
    f1, f2    = top2_idx
    fname1, fname2 = feature_cols[f1], feature_cols[f2]

    fig = plt.figure(figsize=(9, 7))
    ax  = fig.add_subplot(111, projection='3d')
    sc  = ax.scatter(tmp[fname1], tmp[fname2], y,
                     c=y, cmap='viridis', s=60, alpha=0.8)

    ax.set_xlabel(fname1)
    ax.set_ylabel(fname2)
    ax.set_zlabel('Spending (TL)')
    ax.set_title(f'Most Influential Factors: {fname1} & {fname2}')

    cbar = fig.colorbar(sc, ax=ax, shrink=0.6, label='Spending (TL)')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'influential_factors_3d.png')
    plt.close()

# Simple Regression Model                             

def simple_regression(df: pd.DataFrame):
    print("\n▶ Simple Linear Regression (Screen Time → Spending)")
    
    X = df[['TotalScreenTime']].values
    y = df['TotalSpending'].values

    model = LinearRegression()
    model.fit(X, y)

    print(f"Intercept: {model.intercept_:.2f}")
    print(f"Eğim     : {model.coef_[0]:.4f}")
    print(f"R²       : {model.score(X, y):.4f}")

# Main
def main():
    data = load_data()
    descriptive_stats(data)
    correlation_and_tests(data)
    weekend_analysis(data)
    extra_visuals(data)         
    correlation_heatmap(data)    
    influential_3d_plot(data) 
    app_usage_visuals()         
    simple_regression(data)
    print("\nAnalysis completed. Please review the charts and outputs!")

if __name__ == "__main__":
    main()
