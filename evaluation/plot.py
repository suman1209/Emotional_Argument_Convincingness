import matplotlib.pyplot as plt
import yaml
import pandas as pd

from results import AnnotationEvaluator
import numpy as np
from scipy import stats

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

def plot_rating_histogram(results_table, title='Convincingness Ratings Histogram'):
    """
    Plot a histogram of all the convincingness ratings. 

    Parameters:
        results_table (pd.DataFrame): DataFrame containing the results with 'median_score' column.
        title (str): Title of the histogram.
    """
    ratings = results_table[1].tolist() + results_table[2].tolist() + results_table[3].tolist()
    plt.figure(figsize=(10, 6))
    counts, bins, patches = plt.hist(ratings, bins=range(int(min(ratings)), int(max(ratings)) + 2), align='mid', rwidth=0.8)
    plt.xticks([(bins[i] + bins[i+1]) / 2 for i in range(len(bins)-1)])
    plt.title(title)
    plt.xlabel('Convincingness Rating')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    plt.tight_layout()
    plt.show()

def plot_annotator_rating_distribution(results_table):
    """
    Plot a violin plot showing the distribution of ratings for each annotator.

    Parameters:
        results_table (pd.DataFrame): DataFrame containing the results with annotator ratings in columns 1, 2, 3.
    """
    annotator_columns = [1, 2, 3]
    ratings = [results_table[col].dropna().tolist() for col in annotator_columns]
    annotator_labels = [f'Annotator {i+1}' for i in range(len(annotator_columns))]

    plt.figure(figsize=(8, 6))
    plt.violinplot(ratings, showmeans=True, showmedians=True)
    plt.xticks(range(1, len(annotator_labels) + 1), annotator_labels)
    plt.title('Annotator Rating Distribution')
    plt.ylabel('Convincingness Rating')
    plt.tight_layout()
    plt.show()

def plot_aggregated_ratings(results_table):
    """
    Plot median, mean, and mode convincingness ratings for each argument.

    Parameters:
        results_table (pd.DataFrame): DataFrame containing 'argument_id', 'context_version', and annotator ratings in columns 1, 2, 3.
    """

    argument_ids = results_table['argument_id'].astype(str) + '-' + results_table['context_version'].astype(str)
    ratings = results_table[[1, 2, 3]]

    medians = ratings.median(axis=1)
    means = ratings.mean(axis=1)
    modes = ratings.mode(axis=1)[0]  # Take the first mode if multiple

    plt.figure(figsize=(12, 6))
    width = 0.25
    x = np.arange(len(argument_ids))

    plt.bar(x - width, medians, width=width, label='Median')
    plt.bar(x, means, width=width, label='Mean')
    plt.bar(x + width, modes, width=width, label='Mode')

    plt.xticks(x, argument_ids)
    plt.title('Convincingness Ratings per Argument')
    plt.xlabel('Argument ID')
    plt.ylabel('Convincingness Rating')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Load the data from the CSV fi
    data = pd.read_csv('evaluation/test.csv')
    evaluator = AnnotationEvaluator(data)

    # Extract data for plotting
    data_table = evaluator.get_result_table() 
    median_ratings = data_table['median_score']
    argument_ids = data_table['argument_id'].astype(str) + '-' + data_table['context_version'].astype(str)

    # Plot the convincingness ratings
    # plot_rating_histogram(data_table)
    # plot_annotator_rating_distribution(data_table)
    plot_aggregated_ratings(data_table)