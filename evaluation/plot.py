import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import pandas as pd
import scienceplots
import krippendorff

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
    "figure.dpi": 300,
    "savefig.dpi": 300,
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
    Plot heatmap of agreement scores between human annotators and LLMs.
    Requires the AnnotationEvaluator class with compute_agreement_score(annotator_col, model_col).
    """
    annotators = get_annotator_ids(human_table)
    llm_ids = get_annotator_ids(llm_table)

    heatmap_matrix = np.zeros((len(annotators), len(llm_ids)))

    for i, human_id in enumerate(annotators):
        for j, llm_id in enumerate(llm_ids):
            # Use Krippendorff's alpha or other agreement metric
            ratings_1 = human_table[human_id]
            ratings_2 = llm_table[llm_id]
            # Only compare where both are not NaN
            mask = ratings_1.notna() & ratings_2.notna()
            if mask.sum() > 0:
                agreement = krippendorff.alpha(
                    reliability_data=np.vstack((ratings_1[mask], ratings_2[mask])),
                    level_of_measurement='ordinal'
                )
            else:
                agreement = np.nan
            heatmap_matrix[i, j] = agreement

    df_heatmap = pd.DataFrame(heatmap_matrix, index=[f"{i+1}" for i in range(len(annotators))],
                              columns=MODELS.keys())

    fig, ax = prepare_plotting()
    sns.heatmap(df_heatmap, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax, cbar_kws={"label": "Agreement Score"})
    ax.set_ylabel("Human Annotator ID")
    ax.set_title("Annotator vs LLM Agreement")
    fig.tight_layout()

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
    sns.violinplot(x="context_type", y="rating", hue="source", data=df_all, ax=ax, split=True, inner="quartile")

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

if __name__ == "__main__":
    # Human evaluation data
    human_data = pd.read_csv('data/annotations.csv', delim_whitespace=True)
    human_evaluator = AnnotationEvaluator(human_data)
    human_data_table = human_evaluator.get_result_table()

    # Plot human ratings
    plot_rating_histogram(human_data_table, save_path='evaluation/fig/convincingness_histogram.png')
    plot_annotator_rating_distribution(human_data_table, save_path='evaluation/fig/annotator_rating_distribution.png')
    plot_aggregated_ratings(human_data_table, argument_id=1, save_path='evaluation/fig/aggregated_ratings.png')
    plot_unjustified_distribution(human_data_table, save_path='evaluation/fig/unjustified_distribution.png')

    # LLM evaluation data
    llm_data = pd.read_csv('data/annotations_llm.csv', delim_whitespace=True)
    llm_evaluator = AnnotationEvaluator(llm_data)
    llm_data_table = llm_evaluator.get_result_table()

    # Plot LLM ratings
    plot_annotator_rating_distribution(llm_data_table, save_path='evaluation/fig/llm_annotator_rating_distribution.png', llm=True)
    # plot_comparison(human_data_table, llm_data_table, save_path='evaluation/fig/comparison_plot.png')
    plot_llm_annotator_agreement_heatmap(human_data_table, llm_data_table, save_path='evaluation/fig/llm_annotator_agreement_heatmap.png')
    plot_context_type_distribution(human_data_table, llm_data_table, save_path='evaluation/fig/context_type_distribution.png')