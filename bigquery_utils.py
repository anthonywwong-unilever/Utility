from pandas import DataFrame
from pandas_gbq import read_gbq
from typing import Optional, List, Dict, Union
from utils.dataframe_utils import dfs_diff
from utils.general_utils import filter_not_contain, lists_to_dict
from IPython.display import display
from google.api_core.exceptions import NotFound
from google.cloud import bigquery
from google.cloud.bigquery import (
    Client, 
    Table, SchemaField, 
    QueryJobConfig, LoadJobConfig, WriteDisposition,
    SourceFormat, DecimalTargetType
)

import numpy as np


pd_to_bq_dtypes = {
    'bool': 'BOOLEAN',
    'byte': 'BYTES',
    'datetime': 'DATETIME',
    'date': 'DATE',
    'dbdate': 'DATE',
    'string': 'STRING',
    'int': 'INTEGER',
    'float': 'FLOAT',
    'double': 'FLOAT',
    'long': 'BIGINT',
    'timestamp': 'TIMESTAMP',
    'short': 'SMALLINT'
}

bq_to_pd_dtypes = {
    'BOOLEAN': 'bool',
    'BYTES': 'byte',
    'DATETIME': 'datetime64[ns]',
    'DATE': 'dbdate',
    'STRING': 'string',
    'INTEGER': 'int64',
    'FLOAT': 'float64',
    'BIGINT': 'long',
    'TIMESTAMP': 'timestamp[ns][pyarrow]',
    'SMALLINT': 'short'
}


def get_bq_dtypes(
    pd_dtype: str, 
    pd_to_bq_dict: dict[str, str] = pd_to_bq_dtypes) -> Optional[str]:
    for from_dtype, to_dtype in pd_to_bq_dict.items():
        if pd_dtype.startswith(from_dtype):
            return to_dtype
        

def get_pd_dtypes(
    bq_dtype: str, 
    bq_to_pd_dict: dict[str, str] = bq_to_pd_dtypes) -> Optional[str]:
    if bq_dtype in bq_to_pd_dict:
        return bq_to_pd_dict[bq_dtype]
        

def get_bq_schema(df: DataFrame) -> List[SchemaField]:
    """Convert the column data types in <df> to their BigQuery 
    counterparts.
    --------------------------------------------------------------
    Each column of <df> should be one of the following data types:
    1. bool
    2. byte
    3. datetime
    4. date
    4. dbdate
    5. string
    6. int
    7. float
    8. double
    9. long
    10. timestamp
    11. short
    --------------------------------------------------------------
    All columns in <table> are assumed to be nullable by default.
    """
    schema = []
    for column, dtype in df.dtypes.items():
        bq_dtype = get_bq_dtypes(pd_dtype=dtype.name)
        bq_field = SchemaField(name=column, field_type=bq_dtype)
        schema.append(bq_field) 
    return schema

def insert_bq_table_values(
    bq_table: Table, 
    dataset: str, 
    table: str, 
    df: DataFrame, 
    client: Client, 
    iter: int = 1) -> None:

    if iter <= 5:
        try:
            bq_table = client.get_table(bq_table)
            client.insert_rows_from_dataframe(table=bq_table, dataframe=df)
            print(f'Table {dataset}.{table} successfully created and populated.')
        except NotFound:
            print(f'Cannot get table {dataset}.{table}. Trying again...')
            insert_bq_table_values(bq_table, dataset, table, df, client, iter + 1)
    else:      
        print(f"Error! BQ table {dataset}.{table} either doesn't exist or is being created. Investigation required.")
        

def create_gbq_from_parquet(
    project: str, 
    dataset_id: str, 
    table_id: str,
    parquet_gcs_uri: str) -> None:
    """Create a new BigQuery table <project>.<dataset_id>.<table_id> 
    from the Parquet file sitting in the GCS location <parquet_gcs_uri>.
    """
    client = Client(project=project)
    load_job = client.load_table_from_uri(
        parquet_gcs_uri,
        f'{dataset_id}.{table_id}',
        job_config=LoadJobConfig(
            source_format=SourceFormat.PARQUET,
            decimal_target_types=[DecimalTargetType.BIGNUMERIC]
        )
    )
    load_job.result()
    print(f'BigQuery table {project}.{dataset_id}.{table_id} successfully loaded from Parquet.')
    

def create_gbq_from_pd(
    project: str, 
    dataset_id: str, 
    table_id: str,
    df: DataFrame) -> None:
    """Create a new BigQuery table <project>.<dataset>.<table> from the pandas dataframe <df>. 
    Populate the BigQuery table using <df> after its creation. If <project>.<dataset>.<table> 
    already exists, then insert into it using <df>, assuming duplicates are allowed.
    """
    client = Client(project=project)
    table_id = f'{project}.{dataset_id}.{table_id}'
    schema = get_bq_schema(df)

    bq_table = Table(table_ref=table_id, schema=schema)

    try:
        client.get_table(bq_table)
        table_exists = True
    except:
        table_exists = False

    if not table_exists:
        client.create_table(table=bq_table)
        insert_bq_table_values(bq_table, dataset_id, table_id, df, client)
    else:
        job_config = bigquery.LoadJobConfig(
            schema=schema,
            write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE
        )
        job = client.load_table_from_dataframe(df, bq_table, job_config)
        job.result()
        print(f'Table {dataset_id}.{table_id} successfully truncated and re-populated.')


def read_gbq_to_pd(
    project: str, 
    dataset_id: str, 
    table_id: str, 
    cast_pd_dtypes: bool = True) -> DataFrame:
    """Read the BigQuery table <project>.<dataset>.<table> into a Pandas Dataframe.
    Cast all columns to the same data types as in the BigQuery table. 
    """
    client = Client(project=project)
    bq_table_ref = Table(table_ref=f'{project}.{dataset_id}.{table_id}')
    bq_table = client.get_table(bq_table_ref)
    pd_dataframe = read_gbq(f'SELECT * FROM `{project}.{dataset_id}.{table_id}`', project) 

    if cast_pd_dtypes:
        return pd_dataframe.fillna(np.nan).astype({
            schema_field.name: get_pd_dtypes(schema_field.field_type) for schema_field in bq_table.schema})
    return pd_dataframe.fillna(np.nan)


def get_all_tables(
    project: str, 
    dataset_id: str, 
    name_only: bool = False) -> Union[List[str], Dict[str, DataFrame]]:
    """
    """
    client = Client(project=project)
    tables = client.list_tables(dataset_id)
    if name_only:
        return [table.table_id for table in tables]
    return {table.table_id: read_gbq_to_pd(project, dataset_id, table.table_id) for table in tables}


def get_all_tables_startswith(
    project: str, 
    dataset_id: str, 
    startswith: str, 
    name_only: bool = False) -> Union[List[str], Dict[str, DataFrame]]:
    """
    """
    client = Client(project=project)
    tables = client.list_tables(dataset_id)
    if name_only:
        return [table.table_id for table in tables]
    get_table = lambda table_id: read_gbq_to_pd(project, dataset_id, table_id)
    return {table.table_id: get_table(table.table_id) for table in tables if table.table_id.startswith(startswith)}


def get_all_tables_endswith(
    project: str, 
    dataset_id: str, 
    startswith: str, 
    name_only: bool = False) -> Union[List[str], Dict[str, DataFrame]]:
    """
    """
    client = Client(project=project)
    tables = client.list_tables(dataset_id)
    if name_only:
        return [table.table_id for table in tables]
    get_table = lambda table_id: read_gbq_to_pd(project, dataset_id, table_id)
    return {table.table_id: get_table(table.table_id) for table in tables if table.table_id.endswith(startswith)}


def get_all_tables_contains(
    project: str, 
    dataset_id: str, 
    contains: str, 
    name_only: bool = False) -> Union[List[str], Dict[str, DataFrame]]:
    """
    """
    client = Client(project=project)
    tables = client.list_tables(dataset_id)
    try:
        if name_only:
            return [table.table_id for table in tables if contains in table.table_id]
        get_table = lambda table_id: read_gbq_to_pd(project, dataset_id, table_id, False)
        return {table.table_id: get_table(table.table_id) for table in tables if contains in table.table_id}
    except Exception as error:
        print(error)


def upsert_to_bigquery( 
    project: str, 
    dataset_id: str, 
    table_id: str,
    dataframe: DataFrame,
    match_on: List[str] = None,
    when_matched_then_update: List[str] = None) -> None:
    """Upsert the BigQuery table `<project>.<dataset_id>.<table_id>` 
    using records from <dataframe> with a 'MERGE' operation. 

    The operation would match on the columns in <match_on>. If 
    <match_on> isn't given, then match on all columns.

    If <match_on> is given, the matched records from 
    `<project>.<dataset_id>.<table_id>` will be updated 
    using the values from <dataframe> at the columns specified in 
    <when_matched_then_update>. If <when_matched_then_update>
    isn't given, then update all values at columns other than
    those in <match_on>.

    When <match_on> isn't given, then simply insert new records
    from <dataframe> into `<project>.<dataset_id>.<table_id>`.
    
    All unmatched records from <dataframe> will be inserted
    into `<project>.<dataset_id>.<table_id>`.
    """

    # Define the full table ID
    table_ref = f"{project}.{dataset_id}.{table_id}"

    client = Client(project=project)
    schema = client.get_table(Table(table_ref)).schema
    columns = [schema_field.name for schema_field in schema]
    dataframe = dataframe[columns].fillna(np.nan).astype(
        {schema_field.name: get_pd_dtypes(schema_field.field_type) for schema_field in schema}
    )
    
    # Load the dataframe into a temporary table
    temp_table_ref = f"{project}.{dataset_id}.{table_id}_temp"

    job_config = bigquery.LoadJobConfig(
        schema=schema,
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE
    )
    job = client.load_table_from_dataframe(dataframe[columns], temp_table_ref, job_config=job_config)
    job.result()  # Wait for the job to complete

    if match_on is None:
        # When no primary keys specified, then match on all columns
        # Insert new records only (when not matched)
        on_condition = ' AND '.join([f'T.{col} = S.{col}' for col in columns])
        update_condition = None
    elif when_matched_then_update is None:
        # When some primary keys specified, then match on primary keys
        # When matched, update the values for the rest of non-primary key columns
        # Insert new records (when not matched)
        on_condition = ' AND '.join([f'T.{pk} = S.{pk}' for pk in match_on])
        update_condition = ', '.join([f'T.{col} = S.{col}' for col in columns if col not in match_on])
    else:
        # When some primary keys and update keys are specified
        # When matched on primary keys, update the values at the updated keys only
        # Insert new records (when not matched)
        on_condition = ' AND '.join([f'T.{pk} = S.{pk}' for pk in match_on])
        update_condition = ', '.join([f'T.{col} = S.{col}' for col in when_matched_then_update])

    # Define the query
    merge_query = f"""
        MERGE `{table_ref}` T
        USING `{temp_table_ref}` S
        ON {on_condition} 
        {f'WHEN MATCHED THEN UPDATE SET {update_condition}' if update_condition is not None else ''}
        WHEN NOT MATCHED THEN
        INSERT ({', '.join(columns)})
        VALUES ({', '.join([f'S.{col}' for col in columns])})
    """ 

    bq_job_config = QueryJobConfig(use_legacy_sql=False)

    # Execute the merge query
    query_job = client.query(merge_query, job_config=bq_job_config)
    query_job.result()  # Wait for the job to complete

    # Delete the temporary table
    client.delete_table(temp_table_ref)

    print(f"Upserted records into {table_ref}.")


def upsert_explain(
    project: str, 
    dataset_id: str, 
    table_id: str,
    new_df: DataFrame,
    match_on: List[str] = None,
    when_matched_then_update: List[str] = None) -> None:
    """Get a preview of the changes that would happen when
    upserting the BigQuery table `<project>.<dataset_id>.<table_id>` 
    using the dataframe <new_df> with the 'MERGE' operation.

    Print details of records to be updated and inserted, respectively.
    """
    # Conform the column order and data types between the BQ and Pandas dataframes
    old_df = read_gbq_to_pd(project, dataset_id, table_id)
    dtypes_order = [dtype.name for _, dtype in old_df.dtypes.items()]
    updated_df = (new_df[old_df.columns.tolist()]
                  .fillna(np.nan)
                  .astype(lists_to_dict(old_df.columns.tolist(), dtypes_order)))

    old_diff_updated = dfs_diff(old_df, updated_df)
    updated_diff_old = dfs_diff(updated_df, old_df)

    records_to_insert = dfs_diff(updated_diff_old, old_diff_updated, match_on)
    records_to_update = old_diff_updated.merge(updated_df, on=match_on, how='inner')
     
    if records_to_update.empty:
        print(f"No records need to be updated from {project}.{dataset_id}.{table_id}.")
    else:
        if match_on is None:
            value_cols = old_diff_updated.columns.to_list() 
            select_cols = np.array([[f'{col}_from', f'{col}_to'] for col in value_cols]).flatten().tolist()
        elif when_matched_then_update is None:
            value_cols = filter_not_contain(old_diff_updated.columns.to_list(), match_on)
            select_cols = match_on + np.array([[f'{col}_from', f'{col}_to'] for col in value_cols]).flatten().tolist()
        else:
            value_cols = when_matched_then_update
            select_cols = match_on + np.array([[f'{col}_from', f'{col}_to'] for col in value_cols]).flatten().tolist()

        records_to_update = (records_to_update
            .rename(columns={f'{col}_x': f'{col}_from' for col in value_cols})
            .rename(columns={f'{col}_y': f'{col}_to' for col in value_cols}))

        print(f"{records_to_update.shape[0]} old records from {project}.{dataset_id}.{table_id} to update: ")
        display(records_to_update[select_cols])

    if records_to_insert.empty:
        print(f'No new records to insert into {project}.{dataset_id}.{table_id}')
    else:
        print(f'{records_to_insert.shape[0]} new records to insert into {project}.{dataset_id}.{table_id}: ')
        display(records_to_insert)
        
