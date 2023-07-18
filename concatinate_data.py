import pandas as pd
import os
import csv

"""
    Takes the three main data sets and merges them into 
    one, outputting the result as a csv file
"""
def concatinate_data():
    # Basic directory setup
    HOME = os.getcwd()
    DATA_SET_PATH = HOME + '/' + 'Data_sets'
    # File paths to data set files
    NAmatch_Path = DATA_SET_PATH + '/' + 'NAmatch.csv'
    EUmatch_Path = DATA_SET_PATH + '/' + 'EUmatch.csv'
    KRmatch_Path = DATA_SET_PATH + '/' + 'KRmatch.csv'

    data_sets =[EUmatch_Path, KRmatch_Path, NAmatch_Path]
    region_label = ['region.EU', 'region.KR', 'region.NA']

    # Combines all regions datasets into when main one
    regions_df = []

    for i in range(3):
        region = region_label[i]
        df = pd.read_csv(data_sets[i])
        df['region'] = region
        regions_df.append(df)

    """
        For the sake of consistency in final results, this has been disabled.
        This code does work though, so it is optional to use this to replace
        the code above, should there be extra data to consider. The code above 
        only accounts for the data we're considering for this assignment

    for data_set in os.listdir(DATA_SET_PATH):
        if os.path.isfile(DATA_SET_PATH + '/' + data_set):
            region = 'region.' + data_set[0:2]
            df = pd.read_csv(DATA_SET_PATH + '/' + data_set)
            df['region'] = region
            regions_df.append(df)      
    """

    regions_df = pd.concat(regions_df, ignore_index=True)
    regions_df.to_csv('CSVFILES/ALL_REGIONS.csv', index=False)

    # Clarifying State of the Data Set: NA
    na_df = pd.read_csv(NAmatch_Path)
    na_sort_by_champs = na_df.sort_values('champion')

    na_missing_files = na_sort_by_champs.isna().sum()
    na_missing_files.name = 'missing_entry_count'
    na_missing_files.to_csv('Data_sets/Counts/NA_missing_entries.csv')

    na_champ_counts = na_sort_by_champs['champion'].value_counts()
    na_champ_counts.name = 'champion_count'
    na_champ_counts.to_csv('Data_sets/Counts/NA_champ_counts.csv')

    # Clarifying State of the Data Set: EU
    eu_df = pd.read_csv(EUmatch_Path)
    eu_sort_by_champs = eu_df.sort_values('champion')

    eu_missing_files = eu_sort_by_champs.isna().sum()
    eu_missing_files.name = 'missing_entry_count'
    eu_missing_files.to_csv('Data_sets/Counts/EU_missing_entries.csv')

    eu_champ_counts = eu_sort_by_champs['champion'].value_counts()
    eu_champ_counts.name = 'champion_count'
    eu_champ_counts.to_csv('Data_sets/Counts/EU_champ_counts.csv')

    # Clarifying State of the Data Set: KR
    kr_df = pd.read_csv(KRmatch_Path)
    kr_sort_by_champs = kr_df.sort_values('champion')

    kr_missing_files = kr_sort_by_champs.isna().sum()
    kr_missing_files.name = 'missing_entry_count'
    kr_missing_files.to_csv('Data_sets/Counts/KR_missing_entries.csv')

    kr_champ_counts = kr_sort_by_champs['champion'].value_counts()
    kr_champ_counts.name = 'champion_count'
    kr_champ_counts.to_csv('Data_sets/Counts/KR_champ_counts.csv')
    return
