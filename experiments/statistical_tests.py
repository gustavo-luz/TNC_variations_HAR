import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import scipy.stats as stats

data = pd.read_csv('experiments/out/experiment_results.csv')
data


data_plot = pd.read_csv('experiments/out/experiment_results.csv')

# Extract encoder type (RNN or TS2Vec) from the variant column
data_plot['encoder'] = data_plot['variant'].str.extract(r'(RNN|TS2Vec)')

def create_boxplots(df, metrics):
    for metric in metrics:
        plt.figure(figsize=(8, 6))
        sns.boxplot(x='encoder', y=metric, data=df)
        plt.title(f'Boxplot of {metric} by Encoder')
        plt.xlabel('Encoder')
        plt.ylabel(metric)
        plt.show()

metrics = ['acc', 'auprc', 'auroc', 'balanced_acc', 'f1', 'training_time']

create_boxplots(data_plot, metrics)



data_plot['w'] = data_plot['variant'].str.extract(r'(?:adf|sim)-(\d+\.\d+)').astype(float)


def create_boxplots(df, metrics):
    for metric in metrics:
        plt.figure(figsize=(12, 8))
        sns.boxplot(x='w', y=metric, hue='encoder', data=df)
        plt.title(f'Boxplot of {metric} by Encoder and w')
        plt.xlabel('w')
        plt.ylabel(metric)
        plt.legend(title='Encoder')
        plt.show()

metrics = ['acc', 'auprc', 'auroc', 'balanced_acc', 'f1', 'training_time']

create_boxplots(data_plot, metrics)


data_plot['prefix'] = data_plot['variant'].str.extract(r'(adf|sim)')

def create_training_time_boxplot(df):
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='prefix', y='training_time', hue='encoder', data=df)
    plt.title('Boxplot of Training Time by Encoder and Prefix')
    plt.xlabel('Prefix')
    plt.ylabel('Training Time')
    plt.legend(title='Encoder')
    plt.show()

# Create the boxplot
create_training_time_boxplot(data_plot)

df = pd.DataFrame(data)
numeric_columns = ['acc', 'auprc', 'auroc', 'balanced_acc', 'f1', 'training_time']
df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric)


grouped = df.groupby('variant').agg(['mean', 'std'])


# Group by 'variant' and calculate mean and standard deviation
grouped = df.groupby('variant').agg(['mean', 'std'])

# Convert mean and std to percentages (except training_time)
for col in grouped.columns.levels[0]:
    if col != 'training_time':
        grouped[(col, 'mean')] = grouped[(col, 'mean')] * 100
        grouped[(col, 'std')] = grouped[(col, 'std')] * 100

grouped = grouped.round(2)

# Flatten MultiIndex columns
grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]

# Reset index to have 'variant' as a column
grouped = grouped.reset_index()



# Shapiro-Wilk test - normality
def shapiro_test(df, metric):
    stat, p_value = stats.shapiro(df[metric])
    return pd.DataFrame({
        'Shapiro-Wilk Statistic': [stat],
        'p-value': [p_value]
    }, index=[metric])

def interpret_shapiro(p_value):
    if p_value < 0.05:
        return "p-value < 0.05. reject H0 \n data is not normally distributed."
    else:
        return "p-value >= 0.05. fail to reject H0 \n data is normally distributed."

# create a Q-Q plot
def qq_plot(df, metric):
    plt.figure(figsize=(6, 4))
    stats.probplot(df[metric], dist="norm", plot=plt)
    plt.title(f'Q-Q Plot for {metric}')
    plt.show()

for encoder in ['RNN', 'TS2Vec']:
    subset = data[data['variant'].str.contains(encoder)]
    print(f"\nNormality test results for {encoder}:")
    for metric in metrics:
        print(f"\nMetric: {metric}")
        # Shapiro-Wilk Test
        shapiro_results = shapiro_test(subset, metric)
        print(shapiro_results)

        # Interpret Shapiro-Wilk Test Results
        p_value = shapiro_results['p-value'].iloc[0]
        interpretation = interpret_shapiro(p_value)
        print(f"Interpretation: {interpretation}")

        # Q-Q Plot
        qq_plot(subset, metric)



metrics = ['acc', 'auprc', 'auroc', 'balanced_acc', 'f1']

# Wilcoxon signed-rank test
def wilcoxon_test(df, metric):
    rnn_values = df[df['variant'].str.contains('RNN')][metric].dropna()
    ts2vec_values = df[df['variant'].str.contains('TS2Vec')][metric].dropna()
    min_len = min(len(rnn_values), len(ts2vec_values))
    if min_len > 0 and len(rnn_values) == len(ts2vec_values):  # Check for pairing
        rnn_values = rnn_values.sample(min_len).reset_index(drop=True)
        ts2vec_values = ts2vec_values.sample(min_len).reset_index(drop=True)
        stat, p_value = stats.wilcoxon(rnn_values, ts2vec_values)
        return pd.DataFrame({'Wilcoxon Statistic': [stat], 'p-value': [p_value]}, index=[metric])
    else:
        return pd.DataFrame({'Wilcoxon Statistic': [None], 'p-value': [None]}, index=[metric])

def interpret_test(p_value):
    if p_value < 0.05:
        return "Yes, there is a significant difference."
    else:
        return "No, there is no significant difference."

results_list = []

# Perform  test for each metric
for metric in metrics:

    wilcoxon_results = wilcoxon_test(data, metric)
    wilcoxon_stat = wilcoxon_results['Wilcoxon Statistic'].iloc[0]
    wilcoxon_p_value = wilcoxon_results['p-value'].iloc[0]
    interpretation_wilcoxon = interpret_test(wilcoxon_p_value)
    
    result = ({
        'Metric': metric,
        'Wilcoxon Statistic': wilcoxon_stat,
        'Wilcoxon p-value': wilcoxon_p_value,
        'Difference (Wilcoxon)': interpretation_wilcoxon
    })

    results_list.append(result)

results_df = pd.DataFrame(results_list)

print("Analysis complete.")

results_df.to_csv('experiments/out/wilcoxon_results.csv', index=False)

print(results_df)
