import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the dataset
try:
    df = pd.read_csv('StudentsPerformance.csv')
    df.columns = df.columns.str.strip()
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Error: 'StudentsPerformance.csv' not found.")
    exit()

# --- 1. Data Preparation ---
if 'Student_ID' not in df.columns:
    df['Student_ID'] = (df.index + 1001).astype(str)

score_cols = [col for col in df.columns if 'score' in col.lower()]

# --- 2. Calculations ---
df['Average_Score'] = df[score_cols].mean(axis=1)
df['Consistency_Score'] = 100 - df[score_cols].std(axis=1)

def identify_risk(row):
    reasons = []
    if row['Average_Score'] < 50: reasons.append("Low Average")
    if any(row[score_cols] < 40): reasons.append("Subject Failure")
    if len(reasons) > 0: return "High Risk"
    elif row['Average_Score'] < 60: return "Moderate Risk"
    else: return "Low Risk"

df['Risk_Level'] = df.apply(identify_risk, axis=1)

# --- 3. Visualizations ---

# NEW: Line Chart Analysis (Subject Progression)
# We "melt" the data to create a 'Subject' column and a 'Score' column
df_melted = df.melt(id_vars=['Student_ID', 'Risk_Level'], 
                    value_vars=score_cols, 
                    var_name='Subject', 
                    value_name='Score')

plt.figure(figsize=(12, 6))

# Plotting a sample of 15 students to keep the chart readable
sample_ids = df['Student_ID'].head(15)
df_sample = df_melted[df_melted['Student_ID'].isin(sample_ids)]

sns.lineplot(data=df_sample, x='Subject', y='Score', hue='Student_ID', marker='o', linewidth=2.5)

plt.title('Individual Student Performance Progression Across Subjects', fontsize=14)
plt.ylabel('Score (0-100)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Student ID')
plt.tight_layout()
plt.show()

# A. Subject Difficulty Comparison (Bar Chart)
subject_means = df[score_cols].mean()
difficulty_df = ((100 - subject_means) / 100).reset_index()
difficulty_df.columns = ['Subject', 'Difficulty_Index']

plt.figure(figsize=(10, 5))
sns.barplot(x='Subject', y='Difficulty_Index', data=difficulty_df, palette='magma')
plt.title('Subject Difficulty Index (Higher = Harder)')
plt.show()

# B. Risk Distribution (Pie Chart)
plt.figure(figsize=(7, 7))
df['Risk_Level'].value_counts().plot.pie(autopct='%1.1f%%', colors=['#66b3ff','#ffcc99','#ff9999'], startangle=140)
plt.title('Student Risk Distribution')
plt.ylabel('')
plt.show()

# C. ADDED: Correlation Heatmap Analysis
# This shows how performance in one subject relates to others
plt.figure(figsize=(8, 6))
correlation_matrix = df[score_cols].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='RdYlGn', center=0.8)
plt.title('Correlation Heatmap: Subject Score Relationships')
plt.show()

# --- 4. Final Report ---
print("\n--- AT-RISK STUDENTS SUMMARY ---")
at_risk = df[df['Risk_Level'] == 'High Risk'].sort_values(by='Average_Score')
print(at_risk[['Student_ID'] + score_cols + ['Average_Score']].head())
