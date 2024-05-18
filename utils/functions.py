"""
    Utility functions
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric

"""
Data Preprocessing
"""

def transform_cand_id(df):
    df["cand_id"] = df["cand_id"].str.replace(",", "").astype(int)
    return df

def transform_cand_gender(df):
    df["cand_gender"] = (df["cand_gender"] == "Male").astype(int)
    return df

def transform_cand_age_bucket(df):
    unique_values = sorted(df["cand_age_bucket"].unique().tolist())
    cand_age_bucket_encoding = dict(zip(unique_values, range(len(unique_values))))
    df["cand_age_bucket"] = df["cand_age_bucket"].replace(cand_age_bucket_encoding)
    return df

def transform_categorical_column(df, column_name):
    unique_values = sorted(df[column_name].unique().tolist(), key=lambda x: str(x))
    encoding = dict(zip(unique_values, range(len(unique_values))))
    df[column_name] = df[column_name].replace(encoding)
    return df, encoding

def transform_cand_education(df):
    ranked_education_levels = {
        "Dottorato": 8, "Master": 7, "Laurea": 6, "Attestato": 5, "Diploma": 4,
        "ITS": 3, "Media": 2, "Elementare": 1, "Null": 0,
    }
    df["cand_education"] = df["cand_education"].fillna("Null")
    for level in ranked_education_levels.keys():
        df.loc[df["cand_education"].str.contains(level, case=False), "cand_education"] = level
    df["cand_education"] = df["cand_education"].replace(ranked_education_levels)
    return df

def transform_cand_languages_spoken(df):
    df['cand_languages_spoken'] = df['cand_languages_spoken'].fillna('').apply(lambda x: x if x == '' else x.split(';'))
    languages = set()
    for item in df['cand_languages_spoken']:
        languages.update(item)
    languages.discard('')  # Remove '[]' if it's considered a language
    for lang in languages:
        df[lang] = df['cand_languages_spoken'].apply(lambda x: 1 if lang in x else 0)
    df = df.drop(columns=['cand_languages_spoken'])
    return df

def process_full_dataset(df):
    df_processed = df.copy()
    df_processed = transform_cand_id(df_processed)
    df_processed = transform_cand_gender(df_processed)
    df_processed = transform_cand_age_bucket(df_processed)
    for column in ["cand_domicile_province", "cand_domicile_region", "job_contract_type", "job_sector", "job_work_province"]:
        df_processed, _ = transform_categorical_column(df_processed, column)
    df_processed = transform_cand_education(df_processed)
    df_processed = transform_cand_languages_spoken(df_processed)
    return df_processed

"""
Data Analysis
"""

def plot_gender_distribution(
    df: pd.DataFrame, sector_col: str, gender_col: str
) -> None:
    # Calculate the total number of each job sector
    total = df[sector_col].value_counts()

    # Calculate the number of each gender in each job sector
    counts = df.groupby([sector_col, gender_col]).size()

    # Calculate the percentage
    percentages = counts.div(total, level=sector_col) * 100

    # Create a new figure with a specified size
    fig, ax = plt.subplots(figsize=(10, 8))

    # Get the data
    data = percentages.unstack()

    # Normalize the data
    data_normalized = data.div(data.sum(axis=1), axis=0)

    # Plot the data
    y = range(len(data_normalized.index))
    cumulative_size = np.zeros(len(data_normalized.index))
    for i, (colname, values) in enumerate(data_normalized.items()):
        ax.barh(
            y, values, left=cumulative_size, height=0.5
        )  # adjust height parameter as needed
        cumulative_size += values

    ax.set_yticks(y)
    ax.set_yticklabels(data.index)
    ax.set_xlabel("Percentage")
    ax.set_title("Gender Distribution among Job Sectors")
    ax.legend(["Female", "Male"], title="Gender")
    plt.tight_layout()
    plt.show()

def plot_correlation_matrix(df, selected_columns, cmap='bwr', figsize=(10, 10)):
    """
    Plot the correlation matrix for selected columns of a DataFrame.

    Parameters:
    - df: DataFrame to analyze.
    - selected_columns: List of column names to include in the correlation matrix.
    - cmap: Color map for the matrix plot.
    - figsize: Tuple indicating figure size.
    """
    # Calculate correlation matrix
    correlation = df[selected_columns].corr().round(2)

    # Plot the correlation matrix
    fig, ax = plt.subplots(figsize=figsize)
    cax = ax.matshow(correlation, vmin=-1, vmax=1, cmap=cmap)
    fig.colorbar(cax)

    # Add text annotations
    for (i, j), val in np.ndenumerate(correlation.values):
        ax.text(j, i, f"{val:.2f}", ha="center", va="center")

    # Set ticks and labels
    plt.xticks(range(len(correlation.columns)), correlation.columns, rotation=90)
    plt.yticks(range(len(correlation.columns)), correlation.columns)

    plt.tight_layout()
    plt.show()

"""
Bias Analysis
"""

def get_sector_metric(sector, df):
    """
    Calculate disparity metrics for candidates in a given job sector.

    Parameters:
    - sector: The job sector to analyze.
    - df: A DataFrame containing candidate and job information.

    Returns:
    - A DataFrame with columns ['Sector', 'Job', 'Disparate_Impact', 'Statistical_Parity_Difference']
      containing the disparity metrics for each job in the sector.
    """
    # Ensure the sector exists in the DataFrame to avoid empty results
    if sector not in df['job_sector'].unique():
        raise ValueError(f"Sector '{sector}' not found in the DataFrame.")
    
    # Retrieve unique job IDs for the specified sector
    job_list = df[df["job_sector"] == sector]['job_id'].unique()
    job_info_list = []

    for job in job_list:
        # Split DataFrame into idoneous and other candidates based on job_id
        idoneus_candidates_df = df[df["job_id"] == job].copy()
        other_candidates_df = df[df["job_id"] != job].copy()

        # Mark candidates as idoneous or not
        idoneus_candidates_df['idoneous'] = 1
        other_candidates_df['idoneous'] = 0

        # Combine and clean up the DataFrame
        job_df = pd.concat([idoneus_candidates_df, other_candidates_df], axis=0) \
                   .drop_duplicates(subset=['cand_id'], keep='first') \
                   .drop(columns=['job_id', 'distance_km', 'match_score', 'match_rank', 
                                  'job_contract_type', 'job_professional_category', 
                                  'job_sector', 'job_work_province', 'cand_id', 
                                  'cand_domicile_province', 'cand_domicile_region', 
                                  'cand_education', 'cand_languages_spoken'])

        # Prepare dataset for AIF360 analysis
        binaryLabelDataset = BinaryLabelDataset(
            favorable_label=1,
            unfavorable_label=0,
            df=job_df,
            label_names=['idoneous'],
            protected_attribute_names=['cand_gender']
        )

        # Calculate metrics
        metric_orig = BinaryLabelDatasetMetric(
            binaryLabelDataset,
            unprivileged_groups=[{"cand_gender": 0}],
            privileged_groups=[{"cand_gender": 1}]
        )

        job_info = [sector, job, metric_orig.disparate_impact(), metric_orig.statistical_parity_difference()]
        job_info_list.append(job_info)

    columns = ['Sector', 'Job', 'Disparate_Impact', 'Statistical_Parity_Difference']
    return pd.DataFrame(job_info_list, columns=columns)