"""
    Utility functions
"""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.algorithms.preprocessing import DisparateImpactRemover

DEFAULT_COLS = [
    "Sector",
    "Job",
    "Disparate_Impact",
    "Statistical_Parity_Difference",
    "DIDI",
]


"""
Data Exploration
"""


def get_rank_n_candidates(dataset, match_rank):
    return dataset[dataset["match_rank"] == match_rank]


def discretize_feature(data):
    distances_km_discrete = np.zeros(10)
    for dist in data:
        distances_km_discrete[int(dist // 10)] += 1
    return distances_km_discrete


def create_dictionary_from_series(series):  # in percentage
    dict_series = {}
    total = np.sum(series.values)
    for idx, val in zip(series.index, series.values):
        dict_series[idx] = np.around((val / total), 4)
    return dict_series


def create_dicts_rank_n(dataset, cols):
    dict_list = []
    distances_km = discretize_feature(dataset.distance_km)
    total_distances = np.sum(distances_km)
    dict_distances = {}
    for i in range(10):
        dict_distances[i] = np.around(distances_km[i] / total_distances, 4)
    dict_list.append(dict_distances)

    for col in cols:
        dict_list.append(create_dictionary_from_series(dataset[col].value_counts()))

    return dict_list


def create_table_for_feature(list_dict, idx=0):
    selected_dicts = [sublist[idx] for sublist in list_dict]

    total_keys = selected_dicts[0].keys()
    for dictionary in selected_dicts[1:]:
        for key in total_keys:
            if key not in dictionary.keys():
                dictionary[key] = 0

    data = [list(d.values()) for d in selected_dicts]

    return pd.DataFrame(np.vstack(data), columns=list(selected_dicts[0].keys()))


def show_global_distribution(df, feature):
    value_counts = df[feature].value_counts()

    plt.bar(value_counts.index, value_counts.values, color="skyblue")
    plt.title(f"Distribution of {feature}")
    plt.xlabel(feature)
    plt.ylabel("Count")
    plt.xticks(rotation=90)
    plt.show()


def print_feature_distribution(dataframe, title):  # across rank
    dataframe.plot(kind="bar", stacked=True)
    plt.title(f"Distribution of {title} by Rank")
    plt.xlabel("Rank")
    plt.ylabel("Proportion")
    plt.xticks(rotation=0)
    plt.legend(title=title)
    plt.show()


def plot_2_features(
    df,
    feature1,
    feature2,
    num_ranks=[1, 2],
    num_cols=2,
    legend_outside=False,
    response=None,
    x_axis_rotation=90,
):
    data = []
    if num_ranks == None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8), constrained_layout=True)
        distribution = (
            df.groupby(feature1)[feature2].value_counts(normalize=True).unstack()
        )
        data.append(distribution)
        distribution.plot(kind="bar", stacked=True, ax=ax)
        ax.set_title("Full dataset")
        ax.set_xlabel("")
        ax.set_ylabel("Count")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        if legend_outside:
            ax.legend(
                title=feature2,
                bbox_to_anchor=(1.05, 1),
                loc="upper left",
                fontsize="small",
            )
        else:
            ax.legend(title=feature2)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=x_axis_rotation)
        plt.show()
    else:

        fig, axs = plt.subplots(1, num_cols, figsize=(10, 8), constrained_layout=True)
        fig.suptitle(
            f"{feature1} Distribution by {feature2} for Different Ranks", fontsize=16
        )
        for i, rank in enumerate(num_ranks):
            if len(num_ranks) == 1:  # Handle single rank differently
                ax = axs
            else:
                ax = axs[i]
            new_df = df[df.match_rank == rank]
            distribution = (
                new_df.groupby(feature1)[feature2]
                .value_counts(normalize=True)
                .unstack()
            )
            distribution.plot(kind="bar", stacked=True, ax=ax)
            data.append(distribution)
            ax.set_title(f"Rank {rank}")
            ax.set_xlabel("")
            ax.set_ylabel("Count")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
            if legend_outside:
                ax.legend(
                    title=feature2,
                    bbox_to_anchor=(1.05, 1),
                    loc="upper left",
                    fontsize="small",
                )
            else:
                ax.legend(title=feature2)
                ax.set_xticklabels(ax.get_xticklabels(), rotation=x_axis_rotation)


        plt.show()

    if response != None:
        return data


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
    Calculate fairness metrics for each job in the specified sector.

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
    no_idoneous_sectors = 0

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
            job_df[protected_attr_col]
            == attr_favorable_value  # 1 if the candidate has the favorable attribute value
        ).astype(int)

        # If there are no idoneous candidates with the specified protected attribute value, replicate one
        if job_df[(job_df["idoneous"] == 1) & (job_df[protected_attr_col] == 1)].empty:
            # Skip the job if there are no idoneous candidates
            no_idoneous_sectors += 1
            continue

            # Get the first idoneous candidate and replicate it with the specified protected attribute value
            candidate_to_replicate = job_df[job_df["idoneous"] == 1].iloc[0].copy()
            candidate_to_replicate[protected_attr_col] = 1

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

    if no_idoneous_sectors > 0:
        print(
            f"Skipped {no_idoneous_sectors} jobs with no idoneous candidates in sector {sector}."
        )

    return pd.DataFrame(job_info_list, columns=DEFAULT_COLS)


def test_bias(df, protected_attr_col, attr_favorable_value):
    """
    Test bias for each sector in the DataFrame.

    Parameters:
    - df: DataFrame to analyze.
    - protected_attr_col: Protected attribute column name.
    - attr_favorable_value: Protected attribute value.

    Returns:
    - DataFrame containing fairness metrics for each sector.
    """
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
    """
    Calculate fairness metrics for each sector in the DataFrame.

    Parameters:
    - df: DataFrame to analyze.
    - sector_column: Sector column name. Default is 'job_sector'.
    - protected_attribute: Protected attribute column name to analyze for bias. Default is "cand_gender".

    Returns:
    - DataFrame containing fairness metrics for each sector.
    """
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


def DIDI_r(data, pred, protected):
    res, avg = 0, np.mean(pred)
    for aname, dom in protected.items():
        for val in dom:
            mask = data[aname] == val
            res += abs(avg - np.mean(pred[mask]))
    return res


def show_bias(df, protected_attr_col, attr_favorable_value, plot_histogram=False):
    """
    Show bias analysis for the specified protected attribute column.

    Parameters:
    - df: DataFrame to analyze.
    - protected_attr_col: Protected attribute column name.
    - attr_favorable_value: Protected attribute value.
    - plot_histogram: Boolean indicating whether to plot histograms for each sector.

    Returns:
    - DataFrame containing fairness metrics for each sector.
    """

    # Add a column to check if the candidate and the job are in the same location
    if protected_attr_col == "same_location":
        df["same_location"] = (
            df["cand_domicile_province"] == df["job_work_province"]
        ).astype(int)

    all_sector_metrics = test_bias(df, protected_attr_col, attr_favorable_value)

    # Save the results
    all_sector_metrics.to_csv("Results/bias_analysis_" + protected_attr_col + ".csv")

    if plot_histogram:
        for sector in df["job_sector"].unique():
            plot_histogram_metric(
                all_sector_metrics,
                "Disparate_Impact",
                sector,
                protected_attr_col,
                save=True,
            )
            plot_histogram_metric(
                all_sector_metrics,
                "Statistical_Parity_Difference",
                sector,
                protected_attr_col,
                save=True,
            )
            plot_histogram_metric(
                all_sector_metrics, "DIDI", sector, protected_attr_col, save=True
            )

    return all_sector_metrics


def plot_histogram_metric(df, metric, sector, protected_attr_col, save=True):
    """
    Plot a histogram for the specified metric in the specified sector.

    Parameters:
    - df: DataFrame containing the metrics.
    - metric: Metric to plot.
    - sector: Sector to analyze.
    - protected_attr_col: Protected attribute column name.
    - save: Boolean indicating whether to save the plot. Default is True. If False, the plot is displayed.
    """
    df_sector = df[df["Sector"] == sector]
    plt.figure(figsize=(8, 6))
    plt.hist(df_sector[metric], color="skyblue", bins=20, edgecolor="black")
    plt.xlabel(metric)
    plt.ylabel("Frequency")
    plt.title(f"{metric} Distribution for {protected_attr_col} in {sector}")
    plt.tight_layout()

    if save:
        plt.savefig(
            f"Results/Plots/Histogram_{protected_attr_col}_{metric}_{sector}.png"
        )
        plt.clf()
        plt.close()
    else:
        plt.show()


def compute_repaired_df(df, sector, protected_attribute):
    """
    Compute the repaired DataFrame for the specified sector and protected attribute.

    Parameters:
    - df: DataFrame to analyze.
    - sector: Sector to analyze.
    - protected_attribute: Protected attribute column name.

    Returns:
    - Original DataFrame for the specified sector and protected attribute.
    - Repaired DataFrame for the specified sector and protected attribute.
    """
    sector_df = df[df["job_sector"] == sector]

    job_list = sector_df["job_id"].unique()

    job = job_list[0]
    job_df = sector_df.copy()
    job_df["idoneous"] = (job_df["job_id"] == job).astype(int)

    job_df = job_df.drop(columns=["job_id", "job_sector"])

    binaryLabelDataset = BinaryLabelDataset(
        favorable_label=1,
        unfavorable_label=0,
        df=job_df,
        label_names=["idoneous"],
        protected_attribute_names=[protected_attribute],
    )

    level = 0.8
    di = DisparateImpactRemover(repair_level=level)

    binaryLabelDataset_repaired = di.fit_transform(binaryLabelDataset)

    job_df_orig = binaryLabelDataset.convert_to_dataframe()[0]
    job_df_repaired = binaryLabelDataset_repaired.convert_to_dataframe()[0]

    return job_df_orig, job_df_repaired


def compute_bias_differences_percentage(df, sectors, protected_attribute, columns, mode='percentage'):
    """
    Compute the differences between the original and repaired DataFrames for each sector.

    Parameters:
    - df: DataFrame to analyze.
    - sectors: List of sectors to analyze.
    - protected_attribute: Protected attribute column name.
    - columns: List of columns to analyze.
    - mode: percentage or total.

    Returns:
    - DataFrame containing the differences between the original and repaired DataFrames for each sector.
    """
    if protected_attribute == "same_location":
        df["same_location"] = (
            df["cand_domicile_province"] == df["job_work_province"]
        ).astype(int)
        columns = df.columns.drop(["job_id", "job_sector"])

    results_df = pd.DataFrame(columns=columns)

    for sector in sectors:
        job_df_orig, job_df_repaired = compute_repaired_df(
            df, sector, protected_attribute
        )
        differences_list = []
        for column in job_df_orig.columns[:-1]:  # do not compute for idoneous
            differences = job_df_orig[column] != job_df_repaired[column]
            num_differences = differences.sum()
            if mode == 'percentage':
                total_count = job_df_orig.shape[0]
                percentage = round((num_differences / total_count) * 100, 2)
                differences_list.append(percentage)
            else:
                differences_list.append(num_differences)

        differences_df = pd.DataFrame([differences_list], columns=columns)
        results_df = pd.concat([results_df, differences_df], ignore_index=True)
    return results_df


def plot_series(series, title, xlabel, ylabel="Count"):
    plt.bar(
        series.index,
        series.values,
    )
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=90)
    plt.show()


def compare_plot(
    original, repaired, labels, title, xlabel, ylabel="Count", size=(6, 6)
):
    width = 0.4
    plt.figure(figsize=size)
    x = np.arange(len(labels))
    plt.bar(x - width / 2, original, width, label="Original", color="skyblue", alpha=1)
    plt.bar(x + width / 2, repaired, width, label="Repaired", color="orange", alpha=1)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(x, labels, rotation=90)
    plt.legend()
    plt.show()


def prepare_different_series(series1, series2):
    all_index = sorted(set(series1.index).union(set(series2.index)))
    orig_counts = series1.reindex(all_index, fill_value=0)
    repaired_counts = series2.reindex(all_index, fill_value=0)
    return orig_counts, repaired_counts, all_index
