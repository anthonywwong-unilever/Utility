from typing import List, Dict, Union, Callable

import pandas as pd
import numpy as np


def deduplicate_table(
    df: pd.DataFrame, 
    group = List[str], 
    rank_col_method = Dict[str, Union[str, Callable]]) -> pd.DataFrame:
    """
    The dataframe <df> has duplicates in <dup_col> for each column group in <group>.
    
    <rank_col_method> specifies a hierarchy of columns and their corresponding 
    methods for ranking <dup_col> in case of ties. 

    Each key value in <rank_col_method> must be either 'max', 'min'.

    e.g: 

    group = ['region', 'commodity', 'date']
    dup_col = 'price'
    rank_col_method = {
        'data_points': 'max',
        'volume': 'min',
        'price': custom_transformation_for_collapsing_duplicates
    }
    
    Rank 'price' by 'data_points' in descending order and choose the price with maximum
    ranking at 'data_points'. In case of a tie in 'data_points', rank 'price' by 'volume' 
    in ascending order and choose the price with minimum ranking at 'volume. In case of 
    a tie in 'volume' too, apply 'custom_transformation_for_collapsing_duplicates' on 'price'
    and 'volume' to calculate a final value to replace 'price'.

    The last key-value pair in <rank_col_method> must satisfy the following:
    - The value should be an aggregation or transformation applied to the column(s) in the key
    - The key could be a comma-separated value indicating multiple columns to pass into the function.

    """
    df_rank = df.copy()
    rank_grouping = group
    
    for col, method in rank_col_method.items():
        if isinstance(method, str):
            df_group = df_rank.groupby(by=rank_grouping)[col]
            if method == 'max':
                df_rank[f'{col}_rank'] = df_group.rank(method='dense', ascending=False)
            else:
                df_rank[f'{col}_rank'] = df_group.rank(method='dense', ascending=True)
        else:
            df_group = df_rank.groupby(by=group, as_index=False)
            custom_transformed_df = df_group.apply(lambda row: pd.Series({col: method(row)}), include_groups=False)
            df_rank = df_rank.merge(custom_transformed_df, on=group, how='outer', indicator=True)
            df_rank[col] = df_rank.apply(lambda row: row[f'{col}_x'] if row['_merge'] == 'both' else row[f'{col}_y'], axis=1)
            df_rank = df_rank.drop(columns=[f'{col}_x', f'{col}_y', '_merge'])

        rank_grouping.append(col)

    rank_cols = [f'{col}_rank' for col in rank_col_method.keys()]
    dedup_condition = (df_rank[rank_cols] == 1).all(axis=1)

    return df_rank.loc[dedup_condition, ~df_rank.columns.isin(rank_cols)]


def dfs_diff(df1: pd.DataFrame, df2: pd.DataFrame, join_on: List[str] = None) -> pd.DataFrame:
    """Return the difference between <df1> and <df2>. (i.e. records in <df1> but not in <df2>)
    """
    if join_on is None:
        compare_df = df1.merge(df2, on=join_on, how='left', indicator=True)
        df1_diff_df2 = compare_df[compare_df['_merge'] == 'left_only'].drop(columns=['_merge'])
    else:
        value_cols = list(filter(lambda col: col not in join_on, df1.columns.to_list()))
        compare_df = df1.merge(df2, on=join_on, how='left', indicator=True)
        df1_diff_df2 = compare_df.loc[compare_df['_merge'] == 'left_only', join_on + [f'{col}_x' for col in value_cols]]

        df1_diff_df2.rename(columns={f'{col}_x': col for col in value_cols}, inplace=True)

    return df1_diff_df2

    
def top_n_rows_by_group(
    df: pd.DataFrame, 
    partitions: List[str], 
    order_by: List[str],    
    n_rows: int) -> pd.DataFrame:
    """
    """
    df['row_number'] = df.sort_values(order_by, ascending=False).groupby(partitions).cumcount()
    return df.loc[df['row_number'] < n_rows, ~df.columns.isin(['row_number'])]


def last_n_rows_by_group(
    df: pd.DataFrame, 
    partitions: List[str], 
    order_by: List[str],    
    n_rows: int) -> pd.DataFrame:
    """
    """
    df['row_number'] = df.sort_values(order_by, ascending=True).groupby(partitions).cumcount()
    return df.loc[df['row_number'] < n_rows, ~df.columns.isin(['row_number'])]


def drop_duplicates(
    df: pd.DataFrame, 
    subsets: List[str], 
    order_by: List[str],
    keep: str = 'max') -> pd.DataFrame:
    """
    """
    if keep == 'min':
        return last_n_rows_by_group(df, subsets, order_by, 1)
    return top_n_rows_by_group(df, subsets, order_by, 1)


def data_assessment(df: pd.DataFrame) -> pd.DataFrame:
    """Find all the unique values of each column of <df>.
    """
    column_uniques = {column: (list(df[column].unique()), df[column].nunique()) for column in df.columns}
    max_nuqniue = np.max([df[column].nunique() for column in df.columns])
    column_data = {column: unique_list + (max_nuqniue - nunique) * [np.nan] for column, (unique_list, nunique) in column_uniques.items()}
    unique_df = pd.DataFrame(column_data).transpose().rename(columns={index: f'value_{index + 1}' for index in range(max_nuqniue)})
    return unique_df