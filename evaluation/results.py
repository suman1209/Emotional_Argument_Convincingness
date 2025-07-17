import numpy as np
import pandas as pd
import yaml
import krippendorff

from scipy.stats import mode, zscore

class AnnotationEvaluator:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.rating_matrix = None

    def pivot_annotations(self):
        self.rating_matrix = self.data.pivot_table(
            index=['argument_id', 'context_version'],
            columns='annotator_id',
            values='convincingness_rating'
        )
        return self.rating_matrix

    def compute_krippendorffs_alpha_ordinal(self):
        if self.rating_matrix is None:
            self.pivot_annotations()
        
        return krippendorff.alpha(
            self.rating_matrix.values.T,
            level_of_measurement='ordinal'
        )
        
    def aggregate_median(self):
        """
        Aggregate the ratings by computing the median for each argument_id.
        
        Returns:
            pd.DataFrame: A DataFrame with argument_id and their corresponding median scores.
        """
        if self.rating_matrix is None:
            self.pivot_annotations()
        median_scores = self.rating_matrix.median(axis=1, skipna=True).reset_index(name='median_score')
        return median_scores

    def aggregate_mode(self):
        """
        Aggregate the ratings by computing the mode for each argument_id.
        
        Returns:
            pd.DataFrame: A DataFrame with argument_id and their corresponding mode scores.
        """
        if self.rating_matrix is None:
            self.pivot_annotations()
        
        mode_scores = self.rating_matrix.apply(lambda x: mode(x, nan_policy='omit').mode, axis=1).reset_index(name='mode_score')
        return mode_scores
        
    def aggregate_mean(self):
        """
        Aggregate the ratings by computing the mean for each argument_id.
        
        Returns:
            pd.DataFrame: A DataFrame with argument_id and their corresponding mean scores.
        """
        if self.rating_matrix is None:
            self.pivot_annotations()
        
        mean_scores = self.rating_matrix.mean(axis=1, skipna=True).reset_index(name='mean_score')
        return mean_scores
        
    def normalize_scores(self):
        if self.rating_matrix is None:
            self.pivot_annotations()
        
        # Z-score normalization
        normalized_matrix = self.rating_matrix.apply(zscore, axis=1, nan_policy='omit')
        return normalized_matrix
    
    def get_result_table(self):
        """
        Get the results of the evaluation including Krippendorff's alpha, median, mode, and mean scores.
        
        Returns:
            pd.DataFrame: A DataFrame containing the argument_id, context_version, median_score,
                          mode_score, and mean_score.
        """
        # Add median etc as column to the rating matrix
        if self.rating_matrix is None:
            self.pivot_annotations()

        median_scores = self.aggregate_median()
        mode_scores = self.aggregate_mode()
        mean_scores = self.aggregate_mean()

        result_table = self.rating_matrix.copy()
        result_table = result_table.merge(median_scores, on=['argument_id', 'context_version'], how='left')
        result_table = result_table.merge(mode_scores, on=['argument_id', 'context_version'], how='left')
        result_table = result_table.merge(mean_scores, on=['argument_id', 'context_version'], how='left')

        # result_table['krippendorff_alpha'] = self.compute_krippendorffs_alpha_ordinal()
        return result_table.reset_index(drop=True)
        


def main():
    data = pd.read_csv('evaluation/test.csv')
    evaluator = AnnotationEvaluator(data)

    # Perform all steps
    rating_matrix = evaluator.pivot_annotations()
    alpha = evaluator.compute_krippendorffs_alpha_ordinal()
    median_scores = evaluator.aggregate_median()
    mode_scores = evaluator.aggregate_mode()
    mean_scores = evaluator.aggregate_mean()

    # save the result to a YAML file
    result = {
        'krippendorff_alpha_tot': float(alpha),
        'rating_matrix': rating_matrix.to_dict(),
        'median_scores': [float(x) for x in median_scores['median_score'].values],
        'mode_scores': [float(x) for x in mode_scores['mode_score'].values],
        'mean_scores': [float(x) for x in mean_scores['mean_score'].values]
    }

    print(evaluator.get_result_table())

    with open('evaluation/results.yaml', 'w') as file:
        yaml.dump(result, file)

if __name__ == "__main__":
    main()

