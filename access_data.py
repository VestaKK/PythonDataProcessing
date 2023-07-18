import os
import csv

"""
    Accesses necessary data in Ed database and stores it in a
    more easily accessible folder
"""
def access_data():
    
    # accessing and copying data set files
    DATA_PATH = "/course/data/a2/games"
    data_files = ['KRmatch.csv', 'EUmatch.csv', 'NAmatch.csv']

    for files in os.listdir(DATA_PATH):
        if files in data_files:

            file_table = open(DATA_PATH + '/' + files, 'r')
            data = csv.reader(file_table)
            new_file = []

            for row in list(data):
                new_file.append(row)

            file_table.close()

            csv_file = open('Data_sets/' + files, 'w')
            writer = csv.writer(csv_file)
            writer.writerows(new_file)
            csv_file.close()

    return

