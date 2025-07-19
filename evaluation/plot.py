import matplotlib.pyplot as plt
import yaml
import pandas as pd

from results import AnnotationEvaluator
import numpy as np
from scipy import stats
from matplotlib.lines import Line2D

def get_annotator_ids(data_table):
    """
    Extract unique annotator IDs from the data table.
    
    Parameters:
        data_table (pd.DataFrame): DataFrame containing the results with annotator ratings in columns.
    
    Returns:
        list: List of unique annotator IDs.
    """
    return [col for col in data_table.columns if isinstance(col, int)]

def plot(x, y,
        x_label='argument_id',
        y_label='convincingness_rating',
        title='Convincingness Ratings',
        xlabel='Argument ID',
        ylabel='Convincingness Rating',
):
    """
    Plot the convincingness ratings of arguments.

    Parameters:
        x (list or pd.Series): Data for the x-axis.
        y (list or pd.Series): Data for the y-axis.
        x_label (str): Label for the x-axis data.
        y_label (str): Label for the y-axis data.
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
    """
    plt.figure(figsize=(10, 6))
    plt.bar(x, y, color='skyblue')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_rating_histogram(results_table, save_path=None):
    """
    Plot a histogram of all the convincingness ratings. 

    Parameters:
        results_table (pd.DataFrame): DataFrame containing the results with 'median_score' column.
        title (str): Title of the histogram.

    """
    annotator_ids = get_annotator_ids(results_table)
    ratings = results_table[annotator_ids].values.flatten()
    fig = plt.figure()
    plt.grid(axis='y', alpha=0.75, zorder=0)
    bins = np.arange(min(ratings) - 0.5, max(ratings) + 1.5, 1)
    plt.hist(ratings, bins=bins, rwidth=0.8, align='mid')
    plt.xticks(np.arange(min(ratings), max(ratings) + 1))
    plt.title('Convincingness Ratings Histogram')
    plt.xlabel('Convincingness Rating')
    plt.ylabel('Frequency')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def plot_annotator_rating_distribution(results_table, save_path=None):
    """
    Plot a violin plot showing the distribution of ratings for each annotator.

    Parameters:
        results_table (pd.DataFrame): DataFrame containing the results with annotator ratings in columns.
        save_path (str, optional): Path to save the plot. If None, the plot will be displayed.
    """
    annotator_columns = get_annotator_ids(results_table)
    ratings = [results_table[col].dropna().tolist() for col in annotator_columns]
    annotator_labels = [f'Annotator {annotator_id}' for annotator_id in annotator_columns]

    plt.figure()
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
    plt.legend(handles=legend_elements)
    plt.xticks(range(1, len(annotator_labels) + 1), annotator_labels)
    plt.title('Annotator Rating Distribution')
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

    # Just use 

    plt.figure()
    width = 0.25
    x = np.arange(len(context_ids))

    plt.bar(x - width, medians, width=width, label='Median')
    plt.bar(x, means, width=width, label='Mean')
    plt.bar(x + width, modes, width=width, label='Mode')

    plt.xticks(x, context_ids)
    plt.title('Convincingness Ratings per Context Version')
    plt.xlabel('Context ID')
    plt.ylabel('Convincingness Rating')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

if __name__ == "__main__":
    # Load the data from the CSV file
    print("Loading data...")
    data = pd.read_csv('evaluation/annotations.csv', delim_whitespace=True)
    evaluator = AnnotationEvaluator(data)

    # Extract data for plotting
    data_table = evaluator.get_result_table()

    # Plot the convincingness ratings
    plot_rating_histogram(data_table, save_path='evaluation/convincingness_histogram.png')
    plot_annotator_rating_distribution(data_table, save_path='evaluation/annotator_rating_distribution.png')
    plot_aggregated_ratings(data_table, argument_id=1, save_path='evaluation/aggregated_ratings.png')