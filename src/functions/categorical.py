'''Helper functions for EDA/cleaning of categorical variables.'''

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.stats import skew, kurtosis

pd.set_option('display.max_columns', None)

'''Load Dataset'''
def load_data(file_path):

    '''Load dataset from a CSV File'''   
    return pd.read_csv(file_path)

'''Summary Statistics'''
def summarize_data(df):
    """Prints key statistics, detects missing values, and provides distribution insights."""
    
    # General summary statistics (same as before)
    print("\nBasic Summary Statistics:\n", df.describe(percentiles=[0.25, 0.50, 0.75]))

    # Check missing values
    print("\nMissing values:\n", df.isnull().sum())

    # Compute skewness & kurtosis for numeric columns
    skew_kurt_data = {
        feature: {"Skewness": skew(df[feature]), "Kurtosis": kurtosis(df[feature])}
        for feature in df.select_dtypes(include=['number']).columns
    }
    
    skew_kurt_df = pd.DataFrame(skew_kurt_data).T  # Convert dictionary to DataFrame
    
    print("\nDistribution Insights (Skewness & Kurtosis):\n", skew_kurt_df)
    
    return skew_kurt_df  # Return the distribution insights if needed elsewhere


'''Distribution Analysis'''
def plot_distribution(df, numerical_features, bins=30, add_kde=True):
    """Plots separate histograms with optional KDE for better readability."""
    num_features = len(numerical_features)
    fig, axes = plt.subplots(nrows=num_features, ncols=1, figsize=(8, num_features * 4))

    for i, feature in enumerate(numerical_features):
        sns.histplot(df[feature], bins=bins, kde=add_kde, ax=axes[i])
        axes[i].set_title(f"Distribution of {feature}")

    plt.tight_layout()
    plt.show()


'''Apply One-Hot Encoding'''
def encode_categorical(df):
    """Performs One-Hot Encoding on categorical features."""
    categorical_features = ['Gender', 'Workout Type', 'Workout Intensity', 'Mood Before Workout', 'Mood After Workout']
    
    df = pd.get_dummies(df, columns=categorical_features, drop_first=False)
    
    return df


'''Correlation Analysis'''
def correlation_analysis(df):

    """Displays correlation heatmap of dataset."""
    correlation_matrix = df.corr()
    plt.figure(figsize=(10,6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Feature Correlation Matrix')
    plt.show()


def compare_intensity_levels(df, intensity_features, numerical_feature):
    
    """Plots boxplots for multiple workout intensity categories."""
    df_filtered = df[intensity_features + [numerical_feature]]

    # Melt the DataFrame for better plotting
    df_melted = df_filtered.melt(id_vars=[numerical_feature], var_name="Workout Intensity", value_name="Active")
    
    # Filter out non-active (zero values)
    df_melted = df_melted[df_melted["Active"] == 1]

    plt.figure(figsize=(10,6))
    sns.boxplot(x=df_melted["Workout Intensity"], y=df_melted[numerical_feature])
    plt.title(f'{numerical_feature} Across Workout Intensity Levels')
    plt.xlabel("Workout Intensity")
    plt.ylabel(numerical_feature)
    plt.show()


'''K-Means Clustering'''
def cluster_users(df, num_clusters=3):

    """Clusters users based on workout efficiency metrics."""
    X = df[['Age', 'Weight (kg)', 'Workout Duration (mins)', 'Calories Burned', 'Heart Rate (bpm)']]
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X)

    """Plot Clusters"""
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=df['Calories Burned'], y=df['Workout Duration (mins)'], hue=df['Cluster'], palette='Set1')
    plt.title('User Clustering by Workout Efficiency')
    plt.show()

    return df


'''Efficiency Score Calculation'''
def calculate_efficiency(df):

    """Computes workout efficiency based on calorie burn rate and heart rate."""
    df['Efficiency_score'] = df['Calories Burned'] / df['Workout Duration (mins)'] * (df['Heart Rate (bpm)'] / 100)
    return df


def plot_boxplots(df, numerical_features):
    """Plots boxplots for multiple numerical features."""
    plt.figure(figsize=(12, 6))
    
    for feature in numerical_features:
        sns.boxplot(x=df[feature])
        plt.title(f"Boxplot of {feature}")
        plt.show()


def plot_kde(df, numerical_features):
    """Plots KDE (density curves) for numerical features."""
    plt.figure(figsize=(12, 6))

    for feature in numerical_features:
        sns.kdeplot(df[feature], label=feature, shade=True)

    plt.title("Feature Density Comparisons")
    plt.legend()
    plt.show()


def plot_relationship(df, feature_x, feature_y):
    """Plots scatter plot to visualize relationships."""
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=df[feature_x], y=df[feature_y])
    plt.title(f'{feature_x} vs {feature_y}')
    plt.show()


def plot_correlation_heatmap(df):
    """Plots correlation heatmap."""
    plt.figure(figsize=(10,8))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Feature Correlation Heatmap")
    plt.show()

def plot_efficiency_vs_intensity(df, intensity_features, efficiency_feature):
    """Plots boxplots for efficiency across workout intensity levels."""
    df_filtered = df[intensity_features + [efficiency_feature]]

    # Melt for better visualization
    df_melted = df_filtered.melt(id_vars=[efficiency_feature], var_name="Workout Intensity", value_name="Active")
    df_melted = df_melted[df_melted["Active"] == 1]  # Remove inactive cases

    # Plot efficiency distribution across intensity levels
    plt.figure(figsize=(10,6))
    sns.boxplot(x=df_melted["Workout Intensity"], y=df_melted[efficiency_feature])
    plt.title(f"Efficiency Score Across Workout Intensity Levels")
    plt.xlabel("Workout Intensity")
    plt.ylabel("Efficiency Score")
    plt.show()


def intensity_efficiency_correlation(df, efficiency_feature):
    """Finds correlation between workout intensity levels and efficiency."""
    correlation_values = df[["Workout Intensity_Low", "Workout Intensity_Medium", "Workout Intensity_High", efficiency_feature]].corr()
    return correlation_values[efficiency_feature].drop(efficiency_feature)


def rank_feature_importance(df, target_feature):
    """Ranks features by their correlation strength with the target metric."""
    correlations = df.corr()[target_feature].drop(target_feature).abs().sort_values(ascending=False)
    return correlations.head(10)  # Show top 10 impactful features

def select_high_correlation_features(df, target_feature, threshold=0.5):
    """Filters dataset to only keep high-correlation features."""
    correlation_matrix = df.corr()
    high_corr_features = correlation_matrix[target_feature][correlation_matrix[target_feature].abs() > threshold].index.tolist()
    return df[high_corr_features]


# '''Main Execution'''
# if __name__ == "__main__":
#     file_path = "workout_fitness_tracker_data.csv"
#     df = load_data(file_path)
    
#     summarize_data(df)

#     numerical_features = ['Age', 'Height (cm)', 'Weight (kg)', 'Workout Duration (mins)', 'Calories Burned', 'Heart Rate (bpm)', 'Steps Taken']
#     plot_distributions(df, numerical_features)

#     correlation_analysis(df)

#     df = cluster_users(df)
#     df = calculate_efficiency(df)

#     print("Final dataset with efficiency scores:\n", df.head())