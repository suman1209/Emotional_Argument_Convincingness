import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import yaml
import pandas as pd
import scienceplots
import krippendorff
from typing import Optional
import params

from results import AnnotationEvaluator
import numpy as np
from scipy import stats
from matplotlib.lines import Line2D


plt.rcParams.update({
    "text.usetex": True,  # Set to True if using LaTeX
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 600,
    "savefig.dpi": 600,
    "lines.linewidth": 1.0,
    "lines.markersize": 4,
})

MODELS = {
    "deepseek-r1": 11,
    "gpt-4o": 12,
    "llama-3.1-70b": 13,
}

def prepare_plotting(n_columns=1, rows=1, cols=1):
    width_in_inches = 3.25 * n_columns  # Adjust width based on number of columns
    golden_ratio = (5 ** 0.5 - 1) / 2  # ~0.618
    height_in_inches = width_in_inches * golden_ratio
    fig, ax = plt.subplots(rows, cols, figsize=(width_in_inches, height_in_inches))

    return fig, ax

def get_annotator_ids(data_table):
    """
    Extract unique annotator IDs from the data table.
    
    Parameters:
        data_table (pd.DataFrame): DataFrame containing the results with annotator ratings in columns.
    
    Returns:
        list: List of unique annotator IDs.
    """
    return [col for col in data_table.columns if isinstance(col, int)]

def plot_rating_histogram(results_table, save_path=None):
    """
    Plot a histogram of all the convincingness ratings. 

    Parameters:
        results_table (pd.DataFrame): DataFrame containing the results with 'median_score' column.
        title (str): Title of the histogram.

    """
    annotator_ids = get_annotator_ids(results_table)
    ratings = results_table[annotator_ids].values.flatten()
    fig, _ = prepare_plotting()
    plt.grid(axis='y', alpha=0.75, zorder=0)
    bins = np.arange(min(ratings) - 0.5, max(ratings) + 1.5, 1)
    plt.hist(ratings, bins=bins, rwidth=0.8, align='mid')
    plt.xticks(np.arange(min(ratings), max(ratings) + 1))
    plt.xlabel('Convincingness Rating')
    plt.ylabel('Frequency')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def plot_annotator_rating_distribution(results_table, save_path=None, llm=False):
    """
    Plot a violin plot showing the distribution of ratings for each annotator.

    Parameters:
        results_table (pd.DataFrame): DataFrame containing the results with annotator ratings in columns.
        save_path (str, optional): Path to save the plot. If None, the plot will be displayed.
    """
    annotator_columns = get_annotator_ids(results_table)
    ratings = [results_table[col].dropna().tolist() for col in annotator_columns]

    prepare_plotting()
    parts = plt.violinplot(ratings, showmeans=True, showmedians=True, showextrema=False)
    # Change mean and median colors
    for b in parts['bodies']:
        b.set_facecolor('#87CEEB')
        b.set_alpha(0.7)
    if 'cmeans' in parts:
        parts['cmeans'].set_color('red')
        parts['cmeans'].set_linewidth(2)
    if 'cmedians' in parts:
        parts['cmedians'].set_color('green')
        parts['cmedians'].set_linewidth(2)
    # Add legend for mean and median
    legend_elements = [
        Line2D([0], [0], color='red', lw=2, label='Mean'),
        Line2D([0], [0], color='green', lw=2, label='Median')
    ]
    plt.legend(handles=legend_elements, loc='upper center')
    if llm:
        labels = MODELS.keys()
        plt.xticks(np.arange(1, len(labels) + 1), labels)
    else:
        plt.xticks(np.arange(1, len(annotator_columns) + 1), annotator_columns)
        plt.xlabel('Annotator ID')
    plt.ylabel('Convincingness Rating')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def plot_aggregated_ratings(results_table, argument_id: int, save_path=None):
    """
    Plot median, mean, and mode convincingness ratings for each argument.

    Parameters:
        results_table (pd.DataFrame): DataFrame containing 'argument_id', 'context_version', and annotator ratings in columns.
        argument_id (int): Argument ID to plot.
        save_path (str, optional): Path to save the plot. If None, the plot will be displayed.
    """
    annotator_ids = get_annotator_ids(results_table)
    filtered_results = results_table[results_table['argument_id'] == argument_id]
    ratings = filtered_results[annotator_ids]
    context_ids = filtered_results['context_version'].unique()
    medians = ratings.median(axis=1)
    means = ratings.mean(axis=1)
    modes = ratings.mode(axis=1)[0]  # Take the first mode if multiple

    prepare_plotting()
    width = 0.25
    x = np.arange(len(context_ids))

    plt.bar(x - width, medians, width=width, label='Median')
    plt.bar(x, means, width=width, label='Mean')
    plt.bar(x + width, modes, width=width, label='Mode')

    plt.xticks(x, context_ids)
    plt.xlabel('Context ID')
    plt.ylabel('Convincingness Rating')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def plot_unjustified_distribution(results_table, save_path=None):
    """
    Plot unjustified rating distribution for each argument.

    Parameters:
        results_table (pd.DataFrame): DataFrame containing 'argument_id', 'context_version', and annotator ratings in columns.
        save_path (str, optional): Path to save the plot
    """
    unjustified_results = results_table[results_table['context_version'].isin(range(2, 8))]
    prepare_plotting()
    for argument_id in unjustified_results['argument_id'].unique():
        filtered = unjustified_results[unjustified_results['argument_id'] == argument_id]
        parts = plt.violinplot(
            filtered['median_score'], positions=[argument_id], widths=0.9, showmeans=True, showmedians=True, showextrema=False
        )
        for b in parts['bodies']:
            b.set_facecolor('#87CEEB')
            b.set_alpha(0.7)
        if 'cmeans' in parts:
            parts['cmeans'].set_color('red')
            parts['cmeans'].set_linewidth(2)
        if 'cmedians' in parts:
            parts['cmedians'].set_color('green')
            parts['cmedians'].set_linewidth(2)
        
    argument_ids = unjustified_results['argument_id'].unique()
    argument_labels = [f"Argument {arg_id}" for arg_id in argument_ids]
    plt.xticks(argument_ids, argument_labels, rotation=45)
    plt.ylim(0.5, 5.5)
    plt.yticks(np.arange(1, 6, 1))
    plt.grid(axis='y', alpha=0.75, zorder=0)
    plt.ylabel('Convincingness Rating')
    # Add legend for mean and median
    legend_elements = [
        Line2D([0], [0], color='red', lw=2, label='Mean'),
        Line2D([0], [0], color='green', lw=2, label='Median')
    ]
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def plot_comparison(results_table, llm_results_table, save_path=None):
    """
    Plot a comparison between justified (context_id=1) and unjustified (context_version=2..7) ratings.
    """
    justified = results_table[results_table['context_version'] == 1]
    unjustified = results_table[results_table['context_version'].isin(range(2, 8))]

    prepare_plotting()
    for argument_id in unjustified['argument_id'].unique():
        filtered_unjustified = unjustified[unjustified['argument_id'] == argument_id]
        filtered_justified = justified[justified['argument_id'] == argument_id]
        plt.scatter(
            argument_id, 
            filtered_justified['median_score'], 
            color='tab:blue',
            edgecolors='w', 
            s=100
        )
        parts = plt.violinplot(
            filtered_unjustified['median_score'], positions=[argument_id], widths=0.9, showmeans=True, showmedians=True, showextrema=False
        )
        for b in parts['bodies']:
            b.set_facecolor('#87CEEB')
            b.set_alpha(0.7)
        if 'cmeans' in parts:
            parts['cmeans'].set_color('red')
            parts['cmeans'].set_linewidth(2)
        if 'cmedians' in parts:
            parts['cmedians'].set_color('green')
            parts['cmedians'].set_linewidth(2)
        
    argument_ids = unjustified['argument_id'].unique()
    argument_labels = [f"Argument {arg_id}" for arg_id in argument_ids]
    plt.xticks(argument_ids, argument_labels, rotation=45)
    plt.ylim(0.5, 5.5)
    plt.yticks(np.arange(1, 6, 1))
    plt.grid(axis='y', alpha=0.75, zorder=0)
    plt.ylabel('Convincingness Rating')
    # Add legend for mean and median
    legend_elements = [
        Line2D([0], [0], color='red', lw=2, label='Mean'),
        Line2D([0], [0], color='green', lw=2, label='Median')
    ]
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def plot_llm_annotator_agreement_heatmap(human_table, llm_table, save_path=None):
    """
    Plot heatmap of agreement scores between all annotations.
    Requires the AnnotationEvaluator class with compute_agreement_score(annotator_col, model_col).
    """
    # Prepare labels: "Human Median" + LLM names
    labels = ["Human"] + list(MODELS.keys())
    n = len(labels)

    # Prepare ratings for each source
    sources = []
    # Human median ratings
    human_median = human_table[get_annotator_ids(human_table)].median(axis=1)
    sources.append(human_median)
    # LLM ratings
    for llm_id in MODELS.values():
        sources.append(llm_table[llm_id])

    # Compute agreement scores for all pairs (including human median) using Spearman correlation
    heatmap_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            ratings_i = sources[i]
            ratings_j = sources[j]
            mask = ratings_i.notna() & ratings_j.notna()
            if mask.sum() > 0 and j <= i:
                # Spearman correlation
                corr, _ = stats.spearmanr(ratings_i[mask], ratings_j[mask])
                agreement = corr
            else:
                agreement = np.nan
            heatmap_matrix[i, j] = agreement

    df_heatmap = pd.DataFrame(heatmap_matrix, index=labels, columns=labels)

    fig, ax = prepare_plotting()
    sns.heatmap(df_heatmap, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax, cbar_kws={"label": r"Rank Correlation $\rho$"}, vmin=0, vmax=1)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path)
    else:
        plt.show()

def plot_context_type_distribution(human_table, llm_table, save_path=None):
    """
    Plot convincingness ratings distribution for justified vs unjustified contexts.
    """
    annotator_ids = get_annotator_ids(human_table)
    melted_human = human_table.melt(
        id_vars=["argument_id", "context_version"],
        value_vars=annotator_ids,
        var_name="source",
        value_name="rating"
    )
    melted_human["source"] = "Human"

    # Load LLM ratings
    melted_llm = pd.DataFrame()
    # Compute median rating across the three LLMs for each row
    llm_cols = list(MODELS.values())
    temp = llm_table[["argument_id", "context_version"] + llm_cols].copy()
    temp["rating"] = temp[llm_cols].median(axis=1)
    temp["source"] = "LLMs"
    melted_llm = temp[["argument_id", "context_version", "rating", "source"]]

    df_all = pd.concat([melted_human, melted_llm], ignore_index=True)
    df_all = df_all.dropna(subset=["rating"])
    df_all["context_type"] = df_all["context_version"].apply(lambda x: "Justified" if x == 1 else "Unjustified")

    fig, ax = prepare_plotting()
    sns.violinplot(x="context_type", y="rating", hue="source", data=df_all, ax=ax, split=True, inner="quartile", density_norm='width')

    # Axes setup
    ax.set_xlabel("Context Type")
    ax.set_ylabel("Convincingness Rating")
    ax.set_ylim(0.5, 5.5)
    ax.set_yticks(np.arange(1, 6, 1))
    ax.grid(axis='y', alpha=0.5)
    ax.legend()

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path)
    else:
        plt.show()

def plot_context_type_distribution_per_annotator(human_table, llm_table=None, save_path=None):
    """
    Plot convincingness ratings distribution for justified vs unjustified contexts per annotator.
    """
    human_annotator_ids = get_annotator_ids(human_table)
    llms = list(MODELS.keys())
    annotator_labels = human_annotator_ids + llms

    # Per annotator data preparation
    plot_rows = []
    # Human ratings
    for annot_id in human_annotator_ids:
        justified_ratings = human_table[human_table['context_version'] == 1][annot_id].dropna().tolist()
        for r in justified_ratings:
            plot_rows.append({'annotator': str(annot_id), 'context_type': 'Justified', 'rating': r, 'source': 'Human'})
        unjustified_ratings = human_table[human_table['context_version'] != 1][annot_id].dropna().tolist()
        for r in unjustified_ratings:
            plot_rows.append({'annotator': str(annot_id), 'context_type': 'Unjustified', 'rating': r, 'source': 'Human'})
    # LLM ratings
    if llm_table is not None:
        for llm_name, llm_col in zip(llms, MODELS.values()):
            justified_ratings = llm_table[llm_table['context_version'] == 1][llm_col].dropna().tolist()
            for r in justified_ratings:
                plot_rows.append({'annotator': llm_name, 'context_type': 'Justified', 'rating': r, 'source': 'LLM'})
            unjustified_ratings = llm_table[llm_table['context_version'] != 1][llm_col].dropna().tolist()
            for r in unjustified_ratings:
                plot_rows.append({'annotator': llm_name, 'context_type': 'Unjustified', 'rating': r, 'source': 'LLM'})
    df_per_annotator = pd.DataFrame(plot_rows)
    
    if llm_table is not None:
        fig_width = 7.16
    else:
        fig_width = 3.25 
    golden = (1 + 5 ** 0.5) / 2
    fig_height = 3.6 / golden
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))

    sns.violinplot(
        x='annotator',
        y='rating',
        hue='context_type',
        data=df_per_annotator,
        inner="quartile",
        split=True,
        ax=ax,
        legend=False,
        density_norm='count',
    )
    fig.tight_layout()
    handles = [
            mpatches.Patch(facecolor=sns.color_palette()[0], label='Justified', linewidth=1, edgecolor='#3F3F3F'),
            mpatches.Patch(facecolor=sns.color_palette()[1], label='Unjustified', linewidth=1, edgecolor='#3F3F3F')
        ]
    # Axes setup
    if llm_table is not None:
        ax.set_xlabel("Annotator / LLM Model")
        fig.legend(
            handles=handles,
            loc='upper center',
            bbox_to_anchor=(0.523, 0.15),  # Move legend below the plot
            fontsize=9,
            ncol=2,
            frameon=False
        )
        fig.subplots_adjust(bottom=0.3)  # Add space for legend
    else:
        ax.set_xlabel("Annotator ID")
        fig.legend(
            handles=handles,
            loc='upper center',
            bbox_to_anchor=(0.58, 0.225),  # Move legend below the plot
            fontsize=9,
            ncol=2,
            frameon=False
        )
        fig.subplots_adjust(bottom=0.4)  # Add space for legend
    ax.set_ylabel("Convincingness Rating")
    ax.set_ylim(0.5, 5.5)
    ax.set_yticks(np.arange(1, 6, 1))
    ax.grid(axis='y', alpha=0.5)

    if save_path:
        fig.savefig(save_path)
    else:
        plt.show()

def plot_rating_distribution_per_context_dimension(human_table, llm_table=None, save_path=None):
    """
    A violin plot for the rating distribution with each unjustified context version (e.g., context 2–7) on the x-axis. 
    Hue represents Human vs LLM ratings if LLM data is provided.

    Params:
        human_table (pd.DataFrame): The human ratings table.
        llm_table (pd.DataFrame, optional): The LLM ratings table.
        save_path (str, optional): The path to save the plot.

    Returns:
        None
    """
    human_annotator_ids = get_annotator_ids(human_table)
    llms = list(MODELS.keys())
    annotator_labels = human_annotator_ids + llms

    # Human ratings
    ratings_per_context = pd.DataFrame()
    for context_version in range(2, 8):
        ratings = human_table[human_table['context_version'] == context_version]
        ratings_per_context = pd.concat([ratings_per_context, pd.DataFrame({
            'context_version': context_version - 1,  # Shift by -1
            'rating': ratings['median_score'],
            'source': 'Human'
        })])

    # LLM ratings
    if llm_table is not None:
        for context_version in range(2, 8):
            ratings = llm_table[llm_table['context_version'] == context_version]
            ratings_per_context = pd.concat([ratings_per_context, pd.DataFrame({
                'context_version': context_version - 1,  # Shift by -1
                'rating': ratings['median_score'],
                'source': 'LLM'
            })])

    fig, ax = prepare_plotting()

    # Create a violin plot
    sns.violinplot(
        x='context_version',
        y='rating',
        hue='source' if llm_table is not None else None,
        data=ratings_per_context,
        inner="quartile",
        split=True if llm_table is not None else False,
        ax=ax,
        legend=True if llm_table is not None else False,
        density_norm='count',
    )
    fig.tight_layout()
    ax.set_xlabel("Context Version")
    ax.set_ylabel("Convincingness Rating")
    ax.set_ylim(0.5, 5.5)
    ax.set_yticks(np.arange(1, 6, 1))
    ax.grid(axis='y', alpha=0.5)
    ax.legend() if llm_table is not None else None

    if save_path:
        fig.savefig(save_path)
    else:
        plt.show()

if __name__ == "__main__":
    datasets = ['v1', 'v2']

    for dataset in datasets:
        # Human evaluation data
        human_data = pd.read_csv(f'data/{dataset}/annotations.csv', delim_whitespace=True)
        human_evaluator = AnnotationEvaluator(human_data, remove_context_ids=params.excluded_context_ids)
        human_data_table = human_evaluator.get_result_table()

        # Plot human ratings
        plot_rating_histogram(human_data_table, save_path=f'evaluation/{dataset}/fig/convincingness_histogram.png')
        plot_annotator_rating_distribution(human_data_table, save_path=f'evaluation/{dataset}/fig/annotator_rating_distribution.png')
        plot_aggregated_ratings(human_data_table, argument_id=1, save_path=f'evaluation/{dataset}/fig/aggregated_ratings.png')
        plot_unjustified_distribution(human_data_table, save_path=f'evaluation/{dataset}/fig/unjustified_distribution.png')

        if dataset == 'v1':
            # LLM evaluation data
            llm_data = pd.read_csv(f'data/{dataset}/annotations_llm.csv', delim_whitespace=True)
            llm_evaluator = AnnotationEvaluator(llm_data, remove_context_ids=params.excluded_context_ids)
            llm_data_table = llm_evaluator.get_result_table()

            # Plot LLM ratings
            plot_annotator_rating_distribution(llm_data_table, save_path=f'evaluation/{dataset}/fig/llm_annotator_rating_distribution.png', llm=True)
            # plot_comparison(human_data_table, llm_data_table, save_path='evaluation/fig/comparison_plot.png')
            plot_llm_annotator_agreement_heatmap(human_data_table, llm_data_table, save_path=f'evaluation/{dataset}/fig/llm_annotator_agreement_heatmap.png')
            plot_context_type_distribution(human_data_table, llm_data_table, save_path=f'evaluation/{dataset}/fig/context_type_distribution.png')
            plot_context_type_distribution_per_annotator(human_data_table, llm_data_table, save_path=f'evaluation/{dataset}/fig/context_type_distribution_per_annotator.png')
            plot_rating_distribution_per_context_dimension(human_data_table, llm_data_table, save_path=f'evaluation/{dataset}/fig/rating_distribution_per_context_dimension.png')
        else:
            plot_context_type_distribution_per_annotator(human_data_table, save_path=f'evaluation/{dataset}/fig/context_type_distribution_per_annotator.png')
            plot_rating_distribution_per_context_dimension(human_data_table, save_path=f'evaluation/{dataset}/fig/rating_distribution_per_context_dimension.png')