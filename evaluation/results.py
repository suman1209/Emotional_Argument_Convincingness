import numpy as np
import pandas as pd
import yaml
import krippendorff
import params

from scipy.stats import mode, zscore, wilcoxon

class AnnotationEvaluator:
    def __init__(self, data: pd.DataFrame, remove_context_ids: list[int] = None):
        self.data = data
        # Remove all rows with the specified context_id if provided
        if remove_context_ids is not None:
            self.data = self.data[~self.data['context_version'].isin(remove_context_ids)]
        self.rating_matrix = self.init_rating_matrix()

    def init_rating_matrix(self):
        rating_matrix = self.data.pivot_table(
            index=['argument_id', 'context_version'],
            columns='annotator_id',
            values='convincingness_rating'
        )
        median_scores = self.aggregate_median(rating_matrix)
        mode_scores = self.aggregate_mode(rating_matrix)
        mean_scores = self.aggregate_mean(rating_matrix)
        rating_matrix = rating_matrix.merge(median_scores, on=['argument_id', 'context_version'], how='left')
        rating_matrix = rating_matrix.merge(mode_scores, on=['argument_id', 'context_version'], how='left')
        rating_matrix = rating_matrix.merge(mean_scores, on=['argument_id', 'context_version'], how='left')

        rating_matrix['krippendorff_alpha'] = self.compute_krippendorffs_alpha_ordinal(rating_matrix)

        alpha_per_argument = self.compute_krippendorffs_alpha_per_argument(rating_matrix)
        rating_matrix['krippendorff_alpha_per_argument'] = rating_matrix['argument_id'].map(alpha_per_argument)

        return rating_matrix
    
    def compute_wilcoxon_signed_rank(self, rating_matrix: pd.DataFrame):
        """
        Compute the Wilcoxon signed-rank test for paired samples.

        Args:
            rating_matrix (pd.DataFrame): A DataFrame where rows are argument_id and context_version
                                          and columns are annotator_id with their ratings.
        
        Returns:
            tuple: A tuple containing the test statistic and p-value.
        """        
        # Group by justified and unjustified contexts
        justified_context_version = 1
        unjustified_versions = list(range(2, 8))

        # Prepare paired data
        paired_justified = []
        paired_unjustified = []

        for arg_id in rating_matrix['argument_id'].unique():
            arg_rows = rating_matrix[rating_matrix['argument_id'] == arg_id]
            justified_row = arg_rows[arg_rows['context_version'] == justified_context_version]
            if justified_row.empty:
                continue
            justified_score = justified_row['median_score'].values[0]

            # Get all unjustified context scores for this argument
            unjustified_rows = arg_rows[arg_rows['context_version'].isin(unjustified_versions)]
            unjustified_scores = unjustified_rows['median_score'].dropna().values
            if len(unjustified_scores) == 0:
                continue

            # Use average of unjustified scores
            unjustified_score = unjustified_scores.mean()
            paired_justified.append(justified_score)
            paired_unjustified.append(unjustified_score)

        print(f"Paired justified scores: {paired_justified}",
              f"\nPaired unjustified scores: {paired_unjustified}")

        # Run the one-sided Wilcoxon test (H1: justified > unjustified)
        stat, p_value = wilcoxon(paired_justified, paired_unjustified, alternative='greater')
        print(f"Wilcoxon test statistic: {stat}, p-value: {p_value}")

        return stat, p_value

    def compute_krippendorffs_alpha_ordinal(self, rating_matrix: pd.DataFrame):
        """
        Compute Krippendorff's alpha for ordinal data.

        Args:
            rating_matrix (pd.DataFrame): A DataFrame where rows are argument_id and context_version
                                          and columns are annotator_id with their ratings.
        
        Returns:
            float: Krippendorff's alpha value.
        """
        annotations = rating_matrix.loc[:, [col for col in rating_matrix.columns if isinstance(col, int)]]
        alpha = krippendorff.alpha(
            reliability_data=annotations.values.T,
            level_of_measurement='ordinal'
        )
        return float(alpha)
    
    def compute_krippendorffs_alpha_per_argument(self, rating_matrix: pd.DataFrame):
        """
        Compute Krippendorff's alpha for each argument_id.

        Args:
            rating_matrix (pd.DataFrame): A DataFrame where rows are argument_id and context_version
                                          and columns are annotator_id with their ratings.

        Returns:
            dict: A dictionary with argument_id as keys and their corresponding Krippendorff's alpha values.
        """
        alpha_per_argument = {}
        for argument_id, group_df in rating_matrix.groupby('argument_id'):
            annotations = group_df.loc[:, [col for col in rating_matrix.columns if isinstance(col, int)]]
            alpha_per_argument[argument_id] = float(krippendorff.alpha(
                annotations.values.T,
                level_of_measurement='ordinal'
            ))
        return alpha_per_argument
        
    def aggregate_median(self, rating_matrix: pd.DataFrame):
        """
        Aggregate the ratings by computing the median for each argument_id.

        Args:
            rating_matrix (pd.DataFrame): A DataFrame where rows are argument_id and context_version
                                          and columns are annotator_id with their ratings.
        
        Returns:
            pd.DataFrame: A DataFrame with argument_id and their corresponding median scores.
        """
        median_scores = rating_matrix.median(axis=1, skipna=True).astype(float).reset_index(name='median_score')
        return median_scores

    def aggregate_mode(self, rating_matrix: pd.DataFrame):
        """
        Aggregate the ratings by computing the mode for each argument_id.

        Args:
            rating_matrix (pd.DataFrame): A DataFrame where rows are argument_id and context_version
                                          and columns are annotator_id with their ratings.
        
        Returns:
            pd.DataFrame: A DataFrame with argument_id and their corresponding mode scores.
        """
        mode_scores = rating_matrix.apply(lambda x: mode(x, nan_policy='omit').mode, axis=1).astype(float).reset_index(name='mode_score')
        return mode_scores

    def aggregate_mean(self, rating_matrix: pd.DataFrame):
        """
        Aggregate the ratings by computing the mean for each argument_id.

        Args:
            rating_matrix (pd.DataFrame): A DataFrame where rows are argument_id and context_version
                                          and columns are annotator_id with their ratings.
        
        Returns:
            pd.DataFrame: A DataFrame with argument_id and their corresponding mean scores.
        """
        mean_scores = rating_matrix.mean(axis=1, skipna=True).astype(float).reset_index(name='mean_score')
        return mean_scores
        
    def normalize_scores(self, rating_matrix: pd.DataFrame):
        """
        Normalize the scores in the rating matrix using z-score normalization.

        Args:
            rating_matrix (pd.DataFrame): A DataFrame where rows are argument_id and context_version
                                          and columns are annotator_id with their ratings.

        Returns:
            pd.DataFrame: A DataFrame with normalized scores.
        """
        normalized_matrix = rating_matrix.apply(zscore, axis=1, nan_policy='omit')
        return normalized_matrix
    
    def get_result_table(self):
        """
        Get the results of the evaluation including Krippendorff's alpha, median, mode, and mean scores.
        
        Returns:
            pd.DataFrame: A DataFrame containing the argument_id, context_version, median_score,
                          mode_score, and mean_score, along with Krippendorff's alpha values.
        """
        return self.rating_matrix.reset_index(drop=True)
    
    def process_data(self, save_path: str = None):
        """
        Process the data to compute all necessary statistics and return a summary DataFrame.
        
        Returns:
            pd.DataFrame: A DataFrame containing the results of the evaluation.
        """

        
        alpha = self.rating_matrix['krippendorff_alpha'].iloc[0]
        alpha_per_argument = self.rating_matrix['krippendorff_alpha_per_argument'].to_dict()
        median_scores = list(self.rating_matrix['median_score'].values)
        mode_scores = list(self.rating_matrix['mode_score'].values)
        mean_scores = list(self.rating_matrix['mean_score'].values)
        stat, p_value = self.compute_wilcoxon_signed_rank(self.rating_matrix)

        # alpha_per_argument_float = {k: float(v) for k, v in alpha_per_argument.items()}

        result = {
            'krippendorff_alpha_tot': float(alpha),
            'krippendorff_alpha_per_argument': alpha_per_argument,
            'rating_matrix': self.rating_matrix.to_dict(),
            'median_scores': [float(x) for x in median_scores],
            'mode_scores': [float(x) for x in mode_scores],
            'mean_scores': [float(x) for x in mean_scores],
            'wilcoxon_stat': float(stat),
            'wilcoxon_p_value': float(p_value)
        }

        if save_path:
            with open(save_path, 'w') as file:
                yaml.dump(result, file)
        
        else:
            return result
        


def main():
    datasets = ['v1', 'v2'] 
    for dataset in datasets:
        # Human evaluation data
        human_data = pd.read_csv(f'data/{dataset}/annotations.csv', delim_whitespace=True)
        human_evaluator = AnnotationEvaluator(human_data, remove_context_ids=params.excluded_context_ids)  
        human_evaluator.process_data(save_path=f'evaluation/{dataset}/results_human.yaml')

        # LLM evaluation data
        if dataset == 'v1':
            llm_data = pd.read_csv(f'data/{dataset}/annotations_llm.csv', delim_whitespace=True)
            llm_evaluator = AnnotationEvaluator(llm_data, remove_context_ids=params.excluded_context_ids)
            llm_evaluator.process_data(save_path=f'evaluation/{dataset}/results_llm.yaml')

if __name__ == "__main__":
    main()

