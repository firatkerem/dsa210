# The Impact of Screen Time on Financial Decisions - DSA210 Term Project

## Overview

In this project, I will analyze how my screen time and app usage influence my spending behavior. With the increasing role of digital platforms in everyday life, I have become more aware of how my phone usage might be shaping my financial habits. I often spend a significant amount of time on various apps, and I suspect that this engagement—especially with e-commerce and food delivery platforms—may contribute to increased spending.

Through data tracking and analysis, I will explore trends in my app interactions, comparing them with my transaction history to identify potential spending triggers. By examining patterns in my daily phone activity, I aim to uncover whether higher screen time or specific app categories influence my purchasing decisions. Ultimately, this study will help me gain deeper insights into my financial habits and assess whether modifying my digital behavior could lead to more mindful spending choices.

### **Hypotheses**
- **Null Hypothesis (H₀):**
  There is no statistically significant relationship between screen time or app usage duration and spending behavior.


- **Alternative Hypothesis (H₁):**
  Increased screen time or frequent use of certain applications (e.g., e-commerce and food delivery apps) leads to higher spending habits.
  
---

## Research Objectives  

1. **Identify Patterns**  
   - Determine whether there is a significant correlation between daily screen time and spending patterns.

2. **App-Specific Impact**  
   - Understand which categories of apps (e-commerce, food delivery, social media) have the strongest influence on spending behavior.

3. **Behavioral Triggers**  
   - Pinpoint the moments or triggers—such as weekend browsing—that may lead to impulsive or higher spending.

4. **Practical Recommendations**  
   - Propose actionable steps to improve financial habits if a substantial link between screen time and spending is confirmed.

---

## Methodology  

### **1. Raw Data Collection**  

- **Screen Time Data**  
  - Tracked using built-in phone settings or a digital well-being app.  
  - Exported to an Excel file and categorized by individual apps (e.g., shopping, social media, entertainment).  
  - Converted into a CSV format for analysis.  

- **Transaction Records**  
  - Official bank statements will be used to track categorized expenses.
  - Transactions will be classified into categories such as food, shopping, entertainment, and subscriptions.  
  - Data will be transferred to an Excel file and later converted into CSV format for analysis.  

- **Day of the Week**  
  - Recorded to analyze differences in spending behavior between weekdays and weekends.  

- **Data Collection Period**  
  - The data collection will span several weeks or months to capture both regular spending patterns and anomalies.

---

### **2. Data Cleaning and Integration**  

- **Standardization**  
  - Timestamps and formats will be unified for seamless data merging.  

- **Error Handling**  
  - Duplicate entries and missing values will be identified and appropriately handled.
    
- **Categorization**  
  - Expenses and app usage will be grouped into relevant categories for structured analysis.  

- **Data Merging**  
  - Processed screen time and spending data will be combined based on date to create a unified dataset.  

---

### **3. Exploratory Data Analysis (EDA)**  

- **Descriptive Statistics**  
  - Identify trends in screen time and spending.  

- **Visualizations**  
  - **Histograms** for spending distribution.  
  - **Scatter plots to examine the relationship between screen time and spending patterns.
  - **Line charts will be used to analyze weekly and monthly spending trends.

---

### **4. Analysis and Modeling**  

- **Correlation Analysis**  
  - Measure relationships between screen time and spending habits.  

- **Regression Models**  
  - **Linear Regression** to assess predictive power.  
  - **Decision Trees** if non-linear trends emerge.  

---

### **5. Storage and Version Control**  

- **Final Dataset**  
  - Cleaned CSV files will be stored systematically for analysis.  

- **Analysis Scripts**  
  - Python scripts will be used for data processing, visualization, and modeling.

---

# Results and Data Analysis

## Findings

Multiple analyses were performed to evaluate the project’s main hypotheses. This section outlines the findings and results related to these hypotheses.

### App Usage–Spending Correlation Matrix

![App Spending Correlation](./analysis_output/app_spend_corr_top10.png)

The correlation matrix illustrates the relationship between screen time on top-10 apps and total spending. Most apps show weak correlations with spending, with TikTok (0.21) and LinkedIn (0.04) being slightly more positively related. Overall, no strong relationship is observed, suggesting app usage alone may not be a reliable predictor of spending.

### App-wise Correlation Table with Spending

![App-wise Correlation Table with Spending](./analysis_output/delta_corr_apps.png)

This table displays the correlation coefficients between screen time for each app and total daily spending. Most apps show very weak or negligible correlations, with the highest positive relationship observed for Trendyol (0.35) and the most negative for Binance (-0.34). These findings suggest that while a few apps may influence spending behavior, most app usage has minimal predictive value.

### Total Screen Time vs Daily Spending (Weekday vs Weekend)

![Total Screen Time vs Daily Spending (Weekday vs Weekend)](./analysis_output/scatter_screen_vs_spend.png)

This scatter plot compares total screen time with daily spending, distinguishing between weekdays and weekends. Each point represents a single day, with the red dashed line showing a weak negative trend. The trend suggests that as screen time increases, spending slightly decreases, though this relationship is not strong and likely not statistically significant.

### Total Screen Time vs Number of Transactions

![Total Screen Time vs Number of Transactions](./analysis_output/screen_vs_transactions.png)

This scatter plot explores the relationship between daily total screen time and the number of financial transactions. The red regression line shows a slight positive trend, but the wide confidence interval suggests a weak or negligible correlation. In general, screen time does not appear to strongly influence transaction frequency.

### Spending per Minute by App Usage

![Spending per Minute by App Usage](./analysis_output/spend_per_minute.png)

This table ranks applications based on the average amount of money spent per minute of screen time. Apps like TikTok, Instagram, and WhatsApp show relatively low TL-per-minute values, while platforms such as Trendyol, ChatGPT, and McKinsey Insights are linked to significantly higher spending efficiency. The extreme values may be influenced by low usage time paired with high spending, highlighting potential outliers.

### Spending Allocation by Application

![Spending Allocation by Application](./analysis_output/spend_share_by_app.png)

This table presents the total spending attributed to each app along with its percentage share of total expenditures. TikTok (27.41%) and Snapchat (24.87%) account for more than half of the overall spending, suggesting these platforms may have the strongest influence on financial decisions. In contrast, the majority of apps contribute less than 1%, indicating a highly skewed spending distribution.

### Spending Anomaly Detection Table

![Spending Anomaly Detection Table](./analysis_output/spending_anomalies.png)

This table highlights two specific dates with exceptionally high spending values compared to the rest of the dataset. On March 1 and March 4, 2025, spending exceeded 40,000 TL, despite moderate screen time levels. These dates may represent outliers caused by one-time purchases or unusual financial behavior.

### Time Series of Screen Time and Spending

![Time Series of Screen Time and Spending](./analysis_output/timeseries_screen_spend.png)

This time series plot illustrates the daily variation in total screen time (orange line) and spending (blue line) over the analysis period. Notable spending spikes around March 1st and 4th contrast with relatively stable screen time patterns, suggesting spending anomalies not directly linked to usage time. Overall, screen time shows more regular fluctuation, while spending remains erratic and infrequent.

### Daily Usage of Top-3 Apps vs Total Spending

![Daily Usage of Top-3 Apps vs Total Spending](./analysis_output/top3_apps_vs_spending.png)

This time series graph compares daily usage duration of TikTok, Snapchat, and Instagram with overall daily spending. While TikTok and Snapchat exhibit consistently high usage, there is no strong alignment with spending spikes. Notably, even on days with peak spending (early March), app usage does not drastically change, suggesting spending may not be directly triggered by time spent on these platforms.

### Top 10 Most Used Applications by Screen Time

![Top 10 Most Used Applications by Screen Time](./analysis_output/top10_apps.png)

This horizontal bar chart ranks the ten most-used applications based on total screen time. TikTok and Snapchat dominate usage, each exceeding 4,000 minutes, followed by Instagram and WhatsApp. The remaining apps show considerably lower usage, indicating that a small number of platforms account for the majority of daily screen time.

---

# Hypothesis Test Results

### Null Hypothesis (H₀)
---
There is no statistically significant relationship between screen-time (or app-usage duration) and spending behaviour. Daily spending is completely random and unaffected by how long, or on which apps, I use my phone.

### Alternative Hypothesis (H₁)
---

There is a statistically significant relationship between screen-time and spending behaviour. In particular, more total screen-time or frequent use of certain apps (e-commerce / food-delivery) leads to higher daily spending.

---
# Conclusion

### 1. Screen-Time vs. Spending

- Result
— Null hypothesis retained.
-	Correlation and regression tests all yield non-significant p-values ( >* 0.05 ) and near-zero effect sizes.
-	The regression slope ( -8 TL per extra minute ) is negative but trivial and statistically unreliable.
-	Therefore, total daily screen-time does not appear to drive overall spending.

 ---
 

### 2. App-Specific Effects

-	Result: Alternative hypothesis partially supported.
-	Exploratory correlations show a modest positive link between spending and a few apps (e.g., Trendyol r ≈ 0.35, Snapchat r ≈ 0.25).
-	Most apps exhibit weak or negative associations, and formal app-level significance tests have not yet been performed.
-	Consequently, some individual apps may influence spending, but current evidence is tentative.
 
---

# Overall Assessment

The analyses indicate that general screen-time is largely independent of daily expenditures, upholding the null hypothesis for the primary question. In contrast, the alternative hypothesis receives limited support at the app level: certain high-impact platforms show preliminary signs of driving higher spending, whereas the majority do not. Robust app-specific tests and a larger dataset will be required to confirm these early signals.

---

