"""
    Utility functions
"""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric

DEFAULT_COLS = [
    "Sector",
    "Job",
    "Disparate_Impact",
    "Statistical_Parity_Difference",
    "DIDI",
]

"""
Data Preprocessing
"""


def transform_cand_id(df):
    df["cand_id"] = df["cand_id"].str.replace(",", "").astype(int)
    return df


def transform_cand_gender(df):
    df["cand_gender"] = (df["cand_gender"] == "Male").astype(int)
    return df


def transform_cand_age_bucket(df, dataset_name, save_encoding=True):
    unique_values = sorted(df["cand_age_bucket"].unique().tolist())
    cand_age_bucket_encoding = dict(zip(unique_values, range(len(unique_values))))
    df["cand_age_bucket"] = df["cand_age_bucket"].replace(cand_age_bucket_encoding)

    if save_encoding:
        with open(f"utils/encodings/{dataset_name}_age_bucket_encoding.json", "w") as f:
            json.dump(cand_age_bucket_encoding, f)
    return df


def transform_categorical_column(df, column_name, dataset_name, save_encoding=True):
    unique_values = sorted(df[column_name].unique().tolist(), key=lambda x: str(x))
    encoding = dict(zip(unique_values, range(len(unique_values))))
    df[column_name] = df[column_name].replace(encoding)

    if save_encoding:
        with open(
            f"utils/encodings/{dataset_name}_{column_name}_encoding.json", "w"
        ) as f:
            json.dump(encoding, f)

    return df, encoding


def transform_cand_education(df, dataset_name, save_encoding=True):
    ranked_education_levels = {
        "Dottorato": 8,
        "Master": 7,
        "Laurea": 6,
        "Attestato": 5,
        "Diploma": 4,
        "ITS": 3,
        "Media": 2,
        "Elementare": 1,
        "Null": 0,
    }
    df["cand_education"] = df["cand_education"].fillna("Null")
    for level in ranked_education_levels.keys():
        df.loc[
            df["cand_education"].str.contains(level, case=False), "cand_education"
        ] = level
    df["cand_education"] = df["cand_education"].replace(ranked_education_levels)

    if save_encoding:
        with open(
            f"utils/encodings/{dataset_name}_education_levels_encoding.json", "w"
        ) as f:
            json.dump(ranked_education_levels, f)
    return df


def transform_cand_languages_spoken(df):
    df["cand_languages_spoken"] = (
        df["cand_languages_spoken"]
        .fillna("")
        .apply(lambda x: x if x == "" else x.split(";"))
    )
    languages = set()
    for item in df["cand_languages_spoken"]:
        languages.update(item)
    languages.discard("")  # Remove '[]' if it's considered a language
    for lang in languages:
        df[lang] = df["cand_languages_spoken"].apply(lambda x: 1 if lang in x else 0)
    df = df.drop(columns=["cand_languages_spoken"])
    return df


def transform_provinces(df, dataset_name, save_encoding=True):
    column_name1, column_name2 = "cand_domicile_province", "job_work_province"
    df[column_name1] = df[column_name1].str.strip()
    df[column_name2] = df[column_name2].str.strip()
    unique_values = sorted(
        list(set(df[column_name1].unique()) | set(df[column_name2].unique())),
        key=lambda x: str(x),
    )
    provinces_encoding = dict(zip(unique_values, range(len(unique_values))))
    df[column_name1] = df[column_name1].replace(provinces_encoding)
    df[column_name2] = df[column_name2].replace(provinces_encoding)
    if save_encoding:
        with open(f"utils/encodings/{dataset_name}_provinces_encoding.json", "w") as f:
            json.dump(provinces_encoding, f)
    return df


def transform_to_macrosectors(df, dataset_name, save_encoding=True):
    if "direct" in dataset_name:
        macro_sectors_dir = {
            "Ambiente / Energie Rinnovabili": "Ambiente, Ingegneria e Ricerca",
            "Ingegneria / Ricerca e Sviluppo / Laboratorio": "Ambiente, Ingegneria e Ricerca",
            "Ingegneria Impiantistica": "Ambiente, Ingegneria e Ricerca",
            "Installazione / Impiantistica / Cantieristica": "Ambiente, Ingegneria e Ricerca",
            "Progettisti / Design / Grafici": "Ambiente, Ingegneria e Ricerca",
            "Assistenziale / Paramedico / Tecnico": "Assistenziale e Paramedico",
            "Scientifico / Farmaceutico": "Assistenziale e Paramedico",
            "Banche / Assicurazioni / Istituti di Credito": "Banche e Finanza",
            "Finanza / Contabilità": "Banche e Finanza",
            "Bar / Catering / Personale Di Sala / Cuochi / Chef": "Ristorazione, Alberghiero e Spettacolo",
            "Personale Per Hotel / Piani / Reception / Back Office": "Ristorazione, Alberghiero e Spettacolo",
            "Estetisti / Cure del Corpo": "Ristorazione, Alberghiero e Spettacolo",
            "Moda / Spettacolo / Eventi": "Ristorazione, Alberghiero e Spettacolo",
            "Call Center / Customer Care": "Call Center e Vendita",
            "Commerciale / Vendita": "Call Center e Vendita",
            "Management / Responsabili / Supervisori": "Management e Supervisori",
            "Risorse Umane / Recruitment": "Management e Supervisori",
            "Segreteria / Servizi Generali": "Management e Supervisori",
            "Help Desk / Assistenza Informatica": "Management e Supervisori",
            "Manutenzione / Riparazione": "Logistica e Manutenzione",
            "Magazzino / Logistica / Trasporti": "Logistica e Manutenzione",
            "Personale Aeroportuale": "Logistica e Manutenzione",
            "GDO / Retail / Commessi / Scaffalisti": "Impiegati",
            "Operai Generici": "Operai Generici",
            "Operai Specializzati": "Operai Specializzati",
        }
    elif "reverse" in dataset_name:
        macro_sectors_dir = {
            "Ambiente / Energie Rinnovabili": "Ambiente, Ingegneria e Ricerca",
            "Ingegneria / Ricerca e Sviluppo / Laboratorio": "Ambiente, Ingegneria e Ricerca",
            "Ingegneria Impiantistica": "Ambiente, Ingegneria e Ricerca",
            "Installazione / Impiantistica / Cantieristica": "Ambiente, Ingegneria e Ricerca",
            "Progettisti / Design / Grafici": "Ambiente, Ingegneria e Ricerca",
            "Assistenziale / Paramedico / Tecnico": "Assistenziale e Paramedico",
            "Scientifico / Farmaceutico": "Assistenziale e Paramedico",
            "Medico": "Assistenziale e Paramedico",
            "Banche / Assicurazioni / Istituti di Credito": "Banche e Finanza",
            "Finanza / Contabilità": "Banche e Finanza",
            "Bar / Catering / Personale Di Sala / Cuochi / Chef": "Ristorazione, Alberghiero e Spettacolo",
            "Personale Per Hotel / Piani / Reception / Back Office": "Ristorazione, Alberghiero e Spettacolo",
            "Estetisti / Cure del Corpo": "Ristorazione, Alberghiero e Spettacolo",
            "Moda / Spettacolo / Eventi": "Ristorazione, Alberghiero e Spettacolo",
            "Ristorazione E Hotel Management": "Ristorazione, Alberghiero e Spettacolo",
            "Turismo / Tour Operator / Agenzie Di Viaggio": "Ristorazione, Alberghiero e Spettacolo",
            "Call Center / Customer Care": "Call Center e Vendita",
            "Commerciale / Vendita": "Call Center e Vendita",
            "Management / Responsabili / Supervisori": "Management e Supervisori",
            "Risorse Umane / Recruitment": "Management e Supervisori",
            "Segreteria / Servizi Generali": "Management e Supervisori",
            "Help Desk / Assistenza Informatica": "Management e Supervisori",
            "Manutenzione / Riparazione": "Logistica e Manutenzione",
            "Magazzino / Logistica / Trasporti": "Logistica e Manutenzione",
            "Personale Aeroportuale": "Logistica e Manutenzione",
            "Operai Generici": "Operai Generici",
            "Operai Specializzati": "Operai Specializzati",
            "Analisi / Sviluppo Software / Web": "Sviluppo Software e IT",
            "IT Management / Pre-Sales / Post-Sales": "Sviluppo Software e IT",
            "Infrastruttura IT / DBA": "Sviluppo Software e IT",
            "Polizia / Vigili Urbani / Pubblica Sicurezza": "Vigilanza e Sicurezza",
            "Vigilanza / Sicurezza / Guardie Giurate": "Vigilanza e Sicurezza",
            "GDO / Retail / Commessi / Scaffalisti": "Impiegati e professionisti",
            "Impiegati": "Impiegati e professionisti",
            "Professioni Agricoltura / Pesca": "Impiegati e professionisti",
            "Professioni Artigiane": "Impiegati e professionisti",
            "Servizi Professionali": "Formazione",
            "Formazione / Istruzione / Educatori Professionali": "Formazione",
            "Affari Legali / Avvocati": "Amministrazione",
            "Amministrazione Pubblica": "Amministrazione",
        }
    else:
        raise ValueError(
            "Invalid dataset name. Please provide a valid dataset name. (direct or reverse)"
        )

    df["job_sector"] = df["job_sector"].replace(macro_sectors_dir)

    if save_encoding:
        with open(
            f"utils/encodings/{dataset_name}_macrosectors_encoding.json", "w"
        ) as f:
            json.dump(macro_sectors_dir, f)
    return df


def process_full_dataset(df, dataset_name):
    df_processed = df.copy()
    df_processed = transform_to_macrosectors(df_processed, dataset_name)
    df_processed = transform_cand_id(df_processed)
    df_processed = transform_cand_gender(df_processed)
    df_processed = transform_cand_age_bucket(df_processed, dataset_name)
    df_processed = transform_provinces(df_processed, dataset_name)
    for column in ["cand_domicile_region", "job_contract_type", "job_sector"]:
        df_processed, _ = transform_categorical_column(
            df_processed, column, dataset_name
        )
    df_processed = transform_cand_education(df_processed, dataset_name)
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


def plot_correlation_matrix(df, selected_columns, cmap="bwr", figsize=(10, 10)):
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


def get_sector_metric(
    df: pd.DataFrame,
    sector: int,
    protected_attr_col: str = "cand_gender",
    attr_favorable_value: int = 1,
    sector_column="job_sector",
):
    """
    Calculate fairness metrics for each job in a sector.

    Parameters:
    - df: DataFrame to analyze.
    - sector: Sector to analyze.
    - protected_attr_col: Protected attribute column name.
    - attr_favorable_value: Protected attribute value.
    - sector_column: Sector column name.

    Returns:
    - DataFrame containing fairness metrics for each job in the sector.
    """
    # Ensure the column exists in the DataFrame
    if protected_attr_col not in df.columns:
        raise ValueError(
            f"protected_attr_col '{protected_attr_col}' not found in the DataFrame."
        )

    # Ensure the protected attribute value exists in the DataFrame
    if attr_favorable_value not in df[protected_attr_col].unique():
        raise ValueError(
            f"attr_favorable_value '{attr_favorable_value}' not found in the DataFrame."
        )

    # Ensure the sector exists in the DataFrame
    if sector not in df[sector_column].unique():
        raise ValueError(f"Sector '{sector}' not found in the DataFrame.")

    # Pre-filter DataFrame for the sector
    sector_df = df[df[sector_column] == sector]

    # Retrieve unique job IDs for the specified sector
    job_list = sector_df["job_id"].unique()
    job_info_list = []

    for job in job_list:
        # Identify idoneous candidates directly without splitting and concatenating DataFrames
        job_df = sector_df.copy()

        # Add idoneous column
        job_df["idoneous"] = (job_df["job_id"] == job).astype(int)

        # Drop duplicated candidates, keeping only the first one
        job_df = job_df.sort_values(by=["idoneous"], ascending=False).drop_duplicates(
            subset=["cand_id"], keep="first"
        )

        # Transform the protected attribute value to binary
        job_df[protected_attr_col] = (
            job_df[protected_attr_col] == attr_favorable_value
        ).astype(int)

        # If there are no idoneous candidates with the specified protected attribute value, replicate one
        if (
            job_df[
                (job_df["idoneous"] == 1)
                & (job_df[protected_attr_col] == attr_favorable_value)
            ].shape[0]
            == 0
        ):
            # Get the first idoneous candidate and replicate it with the specified protected attribute value
            candidate_to_replicate = job_df[job_df["idoneous"] == 1].iloc[0].copy()
            candidate_to_replicate[protected_attr_col] = attr_favorable_value

            # Append the replicated candidate to the DataFrame
            job_df.loc[-1] = candidate_to_replicate

        job_df = job_df.drop(
            columns=[
                "job_id",
                "distance_km",
                "match_score",
                "match_rank",
                "job_contract_type",
                "job_professional_category",
                "job_sector",
                "job_work_province",
                "cand_id",
                "cand_domicile_province",
                "cand_domicile_region",
                "cand_education",
            ]
        )

        # Prepare dataset for AIF360 analysis
        binaryLabelDataset = BinaryLabelDataset(
            favorable_label=1,
            unfavorable_label=0,
            df=job_df,
            label_names=["idoneous"],
            protected_attribute_names=[protected_attr_col],
        )

        # Calculate metrics
        metric_orig = BinaryLabelDatasetMetric(
            binaryLabelDataset,
            privileged_groups=[{protected_attr_col: 1}],
            unprivileged_groups=[{protected_attr_col: 0}],
        )

        # Compute DIDI
        DIDI_metric = DIDI_r(job_df, job_df["idoneous"], {protected_attr_col: [1]})

        # Store the result
        job_info = [
            sector,
            job,
            metric_orig.disparate_impact(),
            metric_orig.statistical_parity_difference(),
            DIDI_metric,
        ]
        job_info_list.append(job_info)

    return pd.DataFrame(job_info_list, columns=DEFAULT_COLS)


def test_bias(df, protected_attr_col, attr_favorable_value):
    sectors = df["job_sector"].unique()
    all_sector_metrics = pd.DataFrame(columns=DEFAULT_COLS)

    for sector in sectors:
        sector_metrics = get_sector_metric(
            df, sector, protected_attr_col, attr_favorable_value
        )
        all_sector_metrics = pd.concat([all_sector_metrics, sector_metrics], axis=0)
    return all_sector_metrics


def get_all_sectors_metrics(
    df, sector_column="job_sector", protected_attribute="cand_gender"
):
    sectors = df[sector_column].unique()
    all_sector_metrics = pd.DataFrame(columns=DEFAULT_COLS)

    for sector in sectors:
        sector_metrics = get_sector_metric(
            df,
            sector,
            protected_attr_col=protected_attribute,
            sector_column=sector_column,
        )
        all_sector_metrics = pd.concat([all_sector_metrics, sector_metrics], axis=0)

    return all_sector_metrics


###################################################################################################################


# Function to compute DIDI
def DIDI_r(data, pred, protected):
    res, avg = 0, np.mean(pred)
    for aname, dom in protected.items():
        for val in dom:
            mask = data[aname] == val
            res += abs(avg - np.mean(pred[mask]))
    return res


def show_bias(df, protected_attr_col, attr_favorable_value):

    # Add a column to check if the candidate and the job are in the same location
    if protected_attr_col == "same_location":
        df["same_location"] = (
            df["cand_domicile_province"] == df["job_work_province"]
        ).astype(int)

    all_sector_metrics = test_bias(df, protected_attr_col, attr_favorable_value)
    all_sector_metrics =  all_sector_metrics.groupby("Sector").describe()

    # Save the results
    all_sector_metrics.to_csv("Results/bias_analysis_" + protected_attr_col + ".csv")

    return all_sector_metrics