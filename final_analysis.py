# DSA210 - Screen Time vs Spending Analysis

from pathlib import Path
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

    # Converting the dates
    screen_df['Date']   = pd.to_datetime(screen_df['Date'],   format='%d.%m.%Y')
    spending_df['Date'] = pd.to_datetime(spending_df['Date'], format='%d.%m.%Y')

    # Minute data to number
    screen_df['Time'] = pd.to_numeric(screen_df['Time'], errors='coerce')

    # Daily total screen time
    daily_screen = screen_df.groupby('Date', as_index=False)['Time'].sum().rename(columns={'Time': 'TotalScreenTime'})

    # Daily total spending
    spending_df['Total_Spending'] = pd.to_numeric(spending_df['Total_Spending'], errors='coerce')
    daily_spend = spending_df.groupby('Date', as_index=False)['Total_Spending'].sum().rename(columns={'Total_Spending': 'TotalSpending'})

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

    fig, axes = plt.subplots(2, 1, figsize=(8, 10))

    # Histogram – Screen Time
    sns.histplot(df['TotalScreenTime'], bins=15, kde=False,
                 color='orange', ax=axes[0])
    axes[0].set_title('Daily Total Screen Time Distribution')
    axes[0].set_xlabel('Screen Time (min)')
    axes[0].set_ylabel('Frequency')

    # Violin – Screen Time by Day of Week
    order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday',
             'Friday', 'Saturday', 'Sunday']
    sns.violinplot(x='DayOfWeek', y='TotalScreenTime', data=df,
                   order=order, palette='Pastel2', ax=axes[1], legend=False)
    axes[1].set_title('Screen Time Distribution – Days of the Week')
    axes[1].set_xlabel('Day')
    axes[1].set_ylabel('Screen Time (min)')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'screen_time_distribution.png')
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

# Extra Visuals      
def extra_visuals(df: pd.DataFrame):
    """Generate additional visualizations to enrich the project output."""

    plt.figure(figsize=(12, 6))
    ax1 = plt.gca()
    ax2 = ax1.twinx()

    ax1.plot(df['Date'], df['TotalScreenTime'], label='Screen Time (min)', color='darkorange', linewidth=2)
    ax2.plot(df['Date'], df['TotalSpending'], label='Spending (TL)', color='steelblue', linewidth=2, alpha=0.7)

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

# App-based visuals
def simple_regression(df: pd.DataFrame):
    print("\n▶ Simple Linear Regression (Screen Time → Spending)")
    
    X = df[['TotalScreenTime']].values
    y = df['TotalSpending'].values

    model = LinearRegression()
    model.fit(X, y)

    print(f"Intercept: {model.intercept_:.2f}")
    print(f"Slope    : {model.coef_[0]:.4f}")
    print(f"R²       : {model.score(X, y):.4f}")

def _read_raw_screen(fp='data/screentime.csv'):
    """Read the raw screen‑time CSV (application‑level) and perform basic cleaning."""
    sc = pd.read_csv(fp, sep=';')
    sc['Date'] = pd.to_datetime(sc['Date'], format='%d.%m.%Y')
    sc['Time'] = pd.to_numeric(sc['Time'], errors='coerce')
    sc.rename(columns={'Application_Name': 'App'}, inplace=True)
    return sc


def top3_apps_vs_spending():
    """
    Show the daily usage time of the top‑3 most‑used applications together with total spending on the same graph.
    """
    print("\n▶ Top-3 application ↔ spending time series")
    sc = _read_raw_screen()
    spend = pd.read_csv('data/spending.csv', sep=';')
    spend['Date'] = pd.to_datetime(spend['Date'], format='%d.%m.%Y')
    spend['Total_Spending'] = pd.to_numeric(spend['Total_Spending'], errors='coerce')

    top3 = (sc.groupby('App')['Time'].sum()
              .sort_values(ascending=False).head(3).index)
    sc_top3 = (sc[sc['App'].isin(top3)]
               .groupby(['Date', 'App'])['Time']
               .sum()
               .reset_index())

    daily_spend = spend.groupby('Date', as_index=False)['Total_Spending'].sum()

    plt.figure(figsize=(12, 6))
    ax1 = plt.gca()
    for app in top3:
        subset = sc_top3[sc_top3['App'] == app]
        plt.plot(subset['Date'], subset['Time'], marker='o', label=f"{app} (min)")
    ax2 = ax1.twinx()
    ax2.plot(daily_spend['Date'], daily_spend['Total_Spending'],
             color='black', linewidth=2, label='Harcama (TL)')
    ax1.set_title('Top-3 Application Usage and Daily Spending')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Usage Time (min)')
    ax2.set_ylabel('Spending (TL)')
    lines1, lbl1 = ax1.get_legend_handles_labels()
    lines2, lbl2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, lbl1 + lbl2, loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'top3_apps_vs_spending.png')
    plt.close()


def app_spending_correlation(top_n=10):
    """
    Produce a correlation heatmap between screen time and spending for the top‑N applications (ranked by cumulative usage duration).
    """
    print(f"\n▶ Top-{top_n} application correlation heatmap")
    sc = _read_raw_screen()
    spend = pd.read_csv('data/spending.csv', sep=';')
    spend['Date'] = pd.to_datetime(spend['Date'], format='%d.%m.%Y')
    spend['Total_Spending'] = pd.to_numeric(spend['Total_Spending'], errors='coerce')
    daily_spend = spend.groupby('Date', as_index=False)['Total_Spending'].sum()

    top_apps = (sc.groupby('App')['Time']
                  .sum()
                  .sort_values(ascending=False)
                  .head(top_n)
                  .index)

    pivot = (sc[sc['App'].isin(top_apps)]
             .pivot_table(index='Date', columns='App', values='Time',
                          aggfunc='sum')
             .fillna(0))

    merged = pivot.merge(daily_spend, on='Date', how='inner')
    corr = merged.corr()

    plt.figure(figsize=(1+0.6*len(corr), 0.6*len(corr)))
    sns.heatmap(corr, annot=True, cmap='coolwarm',
                vmin=-1, vmax=1, fmt='.2f', linewidths=0.5)
    plt.title(f'Top-{top_n} Applications & Spending Correlation')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'app_spend_corr_top{top_n}.png')
    plt.close()

def transactions_vs_usage():
    """
    Draw the relationship between screen time and number of transactions (scatter + regression).
    """
    print("\n▶ Number of transactions ↔ screen time scatter")
    sc = _read_raw_screen()
    spend = pd.read_csv('data/spending.csv', sep=';')
    spend['Date'] = pd.to_datetime(spend['Date'], format='%d.%m.%Y')
    spend['Number_of_Transactions'] = pd.to_numeric(
        spend['Number_of_Transactions'], errors='coerce')

    total_sc = sc.groupby('Date')['Time'].sum().rename('TotalScreenTime')
    merged = spend.merge(total_sc, on='Date', how='inner')

    plt.figure(figsize=(6, 6))
    sns.scatterplot(x='TotalScreenTime',
                    y='Number_of_Transactions', data=merged)
    sns.regplot(x='TotalScreenTime', y='Number_of_Transactions',
                data=merged, scatter=False, color='red',
                line_kws={'linestyle': '--'})
    plt.xlabel('Total Screen Time (min)')
    plt.ylabel('Number of Transactions')
    plt.title('Screen Time vs Number of Transactions')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'screen_vs_transactions.png')
    plt.close()


def additional_tables():

    sc = _read_raw_screen() 
    spend = pd.read_csv('data/spending.csv', sep=';')
    spend['Date'] = pd.to_datetime(spend['Date'], format='%d.%m.%Y')
    spend['Total_Spending'] = pd.to_numeric(spend['Total_Spending'], errors='coerce')

    daily_spend = spend.groupby('Date', as_index=False)['Total_Spending'].sum()
    sc_daily    = sc.groupby('Date', as_index=False)['Time'].sum() \
                    .rename(columns={'Time':'TotalScreenTime'})
    merged_day  = pd.merge(daily_spend, sc_daily, on='Date', how='inner')

    total_time  = sc.groupby('App')['Time'].sum().sort_values(ascending=False)
    total_spend = spend['Total_Spending'].sum()

    tbl_spm = (total_spend / total_time).round(2).rename('TL_per_min').reset_index()

    alloc_spend = (total_time / total_time.sum() * total_spend).round(2)
    percent_tot = (alloc_spend / total_spend * 100).round(2)
    tbl_share = pd.DataFrame({
        'App': alloc_spend.index,
        'Allocated_Spend_TL': alloc_spend.values,
        'Percent_of_Total_%': percent_tot.values
    })

    fig_a, ax_a = plt.subplots(figsize=(len(tbl_spm.columns)*2.5,
                                        max(1, len(tbl_spm))*0.35 + 1))
    ax_a.axis('off')
    t_a = ax_a.table(cellText=tbl_spm.values,
                     colLabels=tbl_spm.columns,
                     cellLoc='center', loc='center')
    t_a.auto_set_font_size(False)
    t_a.set_fontsize(8)
    fig_a.tight_layout()
    fig_a.savefig(OUTPUT_DIR / 'spend_per_minute.png', dpi=300)
    plt.close(fig_a)

    fig_b, ax_b = plt.subplots(figsize=(len(tbl_share.columns)*2.5,
                                        max(1, len(tbl_share))*0.35 + 1))
    ax_b.axis('off')
    t_b = ax_b.table(cellText=tbl_share.values,
                     colLabels=tbl_share.columns,
                     cellLoc='center', loc='center')
    t_b.auto_set_font_size(False)
    t_b.set_fontsize(8)
    fig_b.tight_layout()
    fig_b.savefig(OUTPUT_DIR / 'spend_share_by_app.png', dpi=300)
    plt.close(fig_b)

    mean_sp, std_sp = merged_day['Total_Spending'].mean(), merged_day['Total_Spending'].std()
    threshold = mean_sp + 2*std_sp
    anomalies = merged_day[merged_day['Total_Spending'] > threshold] \
                   .rename(columns={'Date':'Anomaly_Date',
                                    'Total_Spending':'Spending_TL',
                                    'TotalScreenTime':'ScreenTime_min'})

    if not anomalies.empty:
        fig9, ax9 = plt.subplots(figsize=(len(anomalies.columns)*2.5,
                                          max(1, len(anomalies))*0.5 + 1))
        ax9.axis('off')
        tbl9 = ax9.table(cellText=anomalies.values,
                         colLabels=anomalies.columns,
                         cellLoc='center', loc='center')
        tbl9.auto_set_font_size(False)
        tbl9.set_fontsize(8)
        fig9.tight_layout()
        fig9.savefig(OUTPUT_DIR / 'spending_anomalies.png', dpi=300)
        plt.close(fig9)

    pivot_apps = sc.pivot_table(index='Date', columns='App', values='Time',
                                aggfunc='sum').fillna(0).sort_index()

    spend_series = daily_spend.set_index('Date')['Total_Spending'].reindex(pivot_apps.index).fillna(0)

    delta_apps   = pivot_apps.diff().fillna(0)
    delta_spend  = spend_series.diff().fillna(0)
    delta_apps['Δ_Spending'] = delta_spend

    corr_series = delta_apps.corr()['Δ_Spending'].drop('Δ_Spending').round(3)
    corr_table  = corr_series.reset_index().rename(columns={'index':'App',
                                                            'Δ_Spending':'Correlation'})
    fig10, ax10 = plt.subplots(figsize=(len(corr_table.columns)*2.5,
                                        max(1, len(corr_table))*0.5 + 1))
    ax10.axis('off')
    tbl10 = ax10.table(cellText=corr_table.values,
                       colLabels=corr_table.columns,
                       cellLoc='center', loc='center')
    tbl10.auto_set_font_size(False)
    tbl10.set_fontsize(8)
    fig10.tight_layout()
    fig10.savefig(OUTPUT_DIR / 'delta_corr_apps.png', dpi=300)
    plt.close(fig10)


# Main
def main():
    data = load_data()
    descriptive_stats(data)
    correlation_and_tests(data)
    extra_visuals(data)         
    simple_regression(data)
    top3_apps_vs_spending()
    app_spending_correlation(top_n=10)     
    transactions_vs_usage()
    additional_tables()
    print("\nAnalysis completed. Please review the charts and outputs!")

if __name__ == "__main__":
    main()
