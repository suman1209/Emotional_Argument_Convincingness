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
        
        print(self.rating_matrix.values.T.shape)
        return krippendorff.alpha(
            reliability_data=self.rating_matrix.values.T,
            level_of_measurement='ordinal'
        )
    
    def compute_krippendorffs_alpha_per_argument(self):
        if self.rating_matrix is None:
            self.pivot_annotations()
        
        alpha_per_argument = {}
        for argument_id in self.rating_matrix.index.get_level_values('argument_id').unique():
            argument_data = self.rating_matrix.xs(argument_id, level='argument_id')
            # print(argument_id, argument_data)
            alpha_per_argument[argument_id] = float(krippendorff.alpha(
                argument_data.values.T,
                level_of_measurement='ordinal'
            ))
        return alpha_per_argument
        
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

        result_table['krippendorff_alpha'] = self.compute_krippendorffs_alpha_ordinal()
        alpha_per_argument = self.compute_krippendorffs_alpha_per_argument()
        # Compute Krippendorff's alpha for each argument
        result_table['krippendorff_alpha_per_argument'] = result_table['argument_id'].map(alpha_per_argument)

        return result_table.reset_index(drop=True)
    
    def process_data(self, save_path: str = None):
        """
        Process the data to compute all necessary statistics and return a summary DataFrame.
        
        Returns:
            pd.DataFrame: A DataFrame containing the results of the evaluation.
        """
        rating_matrix = self.pivot_annotations()
        alpha = self.compute_krippendorffs_alpha_ordinal()
        alpha_per_argument = self.compute_krippendorffs_alpha_per_argument()
        median_scores = self.aggregate_median()
        mode_scores = self.aggregate_mode()
        mean_scores = self.aggregate_mean()

        # alpha_per_argument_float = {k: float(v) for k, v in alpha_per_argument.items()}

        result = {
            'krippendorff_alpha_tot': float(alpha),
            'krippendorff_alpha_per_argument': alpha_per_argument,
            'rating_matrix': rating_matrix.to_dict(),
            'median_scores': [float(x) for x in median_scores['median_score'].values],
            'mode_scores': [float(x) for x in mode_scores['mode_score'].values],
            'mean_scores': [float(x) for x in mean_scores['mean_score'].values]
        }

        if save_path:
            with open(save_path, 'w') as file:
                yaml.dump(result, file)
        
        else:
            return result
        


def main():
    data_human = pd.read_csv('data/annotations.csv', delim_whitespace=True)
    evaluator_human = AnnotationEvaluator(data_human)
    evaluator_human.process_data(save_path='evaluation/results_human.yaml')

    data_llm = pd.read_csv('data/annotations_llm.csv', delim_whitespace=True)
    evaluator_llm = AnnotationEvaluator(data_llm)
    evaluator_llm.process_data(save_path='evaluation/results_llm.yaml')

if __name__ == "__main__":
    main()

