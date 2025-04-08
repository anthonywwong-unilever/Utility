from typing import List, Dict, Union, Callable
from scipy.stats import chisquare

import pandas as pd
import numpy as np
import re


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


def get_group_counts(df: pd.DataFrame, groups: List[str], ascending: bool = False) -> pd.DataFrame:
    """Return the total number of rows for each grouping in <groups> in <df>.
    """
    return df.groupby(groups, as_index=False).size().rename(columns={'size': 'total_count'}).sort_values('total_count', ascending=ascending)


def get_group_min_max(df: pd.DataFrame, groups: List[str], value_col: str) -> pd.DataFrame:
    """Return the minimum and maximum value of <value_col> under for each grouping in <groups> in <df>.
    """
    df_min_max = df.groupby(groups, as_index=False).agg({value_col: ['min', 'max']})
    df_min_max.columns = [f'{col_2}_{col_1}' if col_2 != '' else col_1 for col_1, col_2 in df_min_max.columns.values]
    return df_min_max

def data_assessment(df: pd.DataFrame) -> pd.DataFrame:
    """Find all the unique values of each column of <df>.
    """
    column_uniques = {}
    df = df.fillna(np.nan)
    for column in df.columns:
        unique_values = list(df[column].dropna().unique())
        n_uniques = len(unique_values)
        if n_uniques == 0 and len(unique_values) == 1:
            column_uniques[column] = ([], 0)
        else:
            column_uniques[column] = (unique_values, n_uniques)
    max_nuqniue = np.max([df[column].nunique() for column in df.columns])
    column_data = {column: unique_list + (max_nuqniue - nunique) * [np.nan] for column, (unique_list, nunique) in column_uniques.items()}
    unique_df = pd.DataFrame(column_data).transpose().rename(columns={index: f'value_{index + 1}' for index in range(max_nuqniue)})
    return unique_df


def unpivot(
    df: pd.DataFrame, 
    reg: str, 
    var_name: str, 
    value_name: str) -> pd.DataFrame:
    """Unpivot the pivot dataframe <df> using the columns matching
    the regular expression in <reg>. Pivoted variables will be under 
    the column <var_name>, and values under another column <value_name>.
    """
    pivot_df = df.copy()
    unpivot_variables = list(filter(lambda col: re.search(rf'{reg}', col) is None, pivot_df.columns))
    unpivot_values = list(filter(lambda col: re.search(rf'{reg}', col) is not None, pivot_df.columns))
    unpivot_df = pd.melt(
        pivot_df, 
        id_vars=unpivot_variables, 
        value_vars=unpivot_values, 
        var_name=var_name, 
        value_name=value_name)
    return unpivot_df


def categorical_frequency(df: pd.DataFrame, cat_cols: List[str]) -> pd.DataFrame:
    """Assess if the categorical groups in <cat_cols> from <df> have evenly balanced
    class frequencies (i.e. good data coverage).
    """
    group_df = df.copy()
    cat_col_group_counts = get_group_counts(group_df, cat_cols)
    cat_col_group_counts = cat_col_group_counts.rename(columns={'total_count': 'observed_frequency'})

    n_rows, n_groups = group_df.shape[0], cat_col_group_counts.shape[0]
    exp_freq = n_rows // n_groups
    cat_col_group_counts = cat_col_group_counts.assign(
        percentage=cat_col_group_counts['observed_frequency'] / n_rows,
        expected_frequency=[exp_freq if index < n_groups else exp_freq + (n_rows - index * (exp_freq)) for index in range(1, n_groups + 1, 1)]
    )
    # Test if group classes are balanced (i.e. uniformly distributed)
    _, pvalue = chisquare(f_obs=cat_col_group_counts['observed_frequency'], f_exp=cat_col_group_counts['expected_frequency'])
    if pvalue <= 0.05:
        cat_col_group_counts['high_coverage'] = cat_col_group_counts['observed_frequency'] >= cat_col_group_counts['expected_frequency']
    else:
        cat_col_group_counts['high_coverage'] = True

    column_order = cat_cols + ['percentage', 'observed_frequency', 'expected_frequency', 'high_coverage']
    return cat_col_group_counts[column_order].set_index(cat_cols)


def write_to_excel(df: pd.DataFrame, excel_path: str, sheet_name: str) -> None:
    with pd.ExcelWriter(excel_path, mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)
    print(f"Successfully written to the '{sheet_name}' tab in '{excel_path}'")


def get_group_duplicates_queries(df: pd.DataFrame, groups: List[str]) -> List[str]:
    """Return the set of queries filtering for the duplicate rows in <df>
    under the grouping <groups>.
    """
    df_duplicates = get_group_counts(df, groups).query('total_count > 1')[groups]

    if df_duplicates.empty:
        return []
    
    df_duplicates_tr = df_duplicates.transpose()
    dup_queries = []

    for col in df_duplicates_tr.columns:
        col_query = []
        dup_col_values = df_duplicates_tr[col].to_dict()
        for dup_col, dup_value in dup_col_values.items():
            col_query.append(f"{dup_col} == '{dup_value}'")
        dup_queries.append(' & '.join(col_query))
    
    return dup_queries


def get_group_duplicates(df: pd.DataFrame, groups: List[str]) -> pd.DataFrame:
    """Return all duplicate records in <df> under the grouping <groups>.
    """
    dup_queries = get_group_duplicates_queries(df, groups)
    final_dup_query = ' | '.join([f'({query})' for query in dup_queries])
    return df.query(final_dup_query)