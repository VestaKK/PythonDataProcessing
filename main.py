import os
import sys

def verify_access_data():

    try:
        from access_data import access_data
    except ImportError:
        print("File access function not found.")
        return

    print("=" * 80)
    print("Accessing and copying files...\n")
    access_data()

    print("Checking output...\n")
    if os.path.isfile("Data_sets/EUmatch.csv"):
        print("\tEUmatch.csv output was found.\n")
    else:
        print("\tEUmatch.csv output was NOT found. Please check your code.\n")

    if os.path.isfile("Data_sets/KRmatch.csv"):
        print("\tKRmatch.csv output was found.\n")
    else:
        print("\tKRmatch.csv output was NOT found. Please check your code.\n")

    if os.path.isfile("Data_sets/NAmatch.csv"):
        print("\tNAmatch.csv output was found.\n")
    else:
        print("\tNAmatch.csv output was NOT found. Please check your code.\n")
    
    print("Finished accessing and copying data.")
    print("=" * 80)


def verify_concatinate_data():
    try:
        from concatinate_data import concatinate_data
    except ImportError:
        print("Data concatination function not found")
        return

    print("=" * 80)
    print("Concatinating Data from all regions...\n")
    
    concatinate_data()

    print("Checking output...\n")
    if os.path.isfile("CSVFILES/ALL_REGIONS.csv"):
        print("\tALL_REGIONS.csv was found\n")
    else:
        print("\tALL_REGIONS.csv was NOT found. Please check your code.\n")


    print("Finished data concatination")
    print("=" * 80)


def verify_initial_impute_clean_data():

    try:
        from initial_impute_clean_data import initial_impute_clean_data
    except ImportError:
        print("Inital imputation and cleaning file not found")
        return

    print("=" * 80)
    print("Executing initial imputation and cleaning...")

    initial_impute_clean_data()

    print("Checking output...\n")
    if os.path.isfile("CSVFILES/CLEAN_REGION_DATA.csv"):
        print("\tCLEAN_REGION_DATA.csv was found.\n")
    else:
        print("\tCLEAN_REGION_DATA.csv was NOT found. Please check your code.\n")

    if os.path.isfile("Graphs/pre_summoner_spells.png"):
        print("\tpre_summoner_spells.png was found.\n")
    else:
        print("\tpre_summoner_spells.png was NOT found. Please check your code.\n")

    if os.path.isfile("Graphs/post_summoner_spells.png"):
        print("\tpost_summoner_spells.png was found.\n")
    else:
        print("\tpost_summoner_spells.png was NOT found. Please check your code.\n")
        
    print("Finished initial imputation and cleaning of data")
    print("=" * 80)


def verify_encode_bin_data():

    try:
        from encode_bin_data import encode_bin_data
    except ImportError:
        print("Data binning function not found.")
        return

    print("=" * 80)
    print("Executing data encoding and binning...\n")
    encode_bin_data()

    print("Checking output...\n")

    if os.path.isfile("CSVFILES/BINNED_DATA.csv"):
        print("\tBINNED_DATA.csv was found.\n")
    else:
        print("\tBINNED_DATA.csv was NOT found. Please check your code.\n")

    if os.path.isfile("CSVFILES/BINNED_DATA_ENCODES.json"):
        print("\tBINNED_DATA_ENCODES.json was found.\n")
    else:
        print("\tBINNED_DATA_ENCODES.json was NOT found. Please check your code.\n")

    print("Finished encoding and binning data.")
    print("=" * 80)


def verify_data_analysis():

    try:
        from data_analysis import data_analysis
    except ImportError:
        print("Data analysis function not found.")
        return

    print("=" * 80)
    print("Executing data_analysis...\n")

    data_analysis()

    print("Checking output...\n")
    
    if os.path.isfile("Graphs/confusion_matrix.png"):
        print("\tconfusion_matrix.png was found.\n")
    else:
        print("\tconfusion_matrix.png was NOT found. Please check your code.\n")

    if os.path.isfile("Graphs/decision_tree.png"):
        print("\tdecision_tree.png was found.\n")
    else:
        print("\tdecision_tree.png was NOT found. Please check your code.\n")

    if os.path.isfile("TXT_FILES/final_model_evaluation.txt"):
        print("\tfinal_model_evaluation.txt was found.\n")
    else:
        print("\tfinal_model_evaluation.txt was NOT found. Please check your code.\n")

    print("Finished executing data analysis.")
    print("=" * 80)


def verify_scatter_plots():

    try:
        from scatter_plots import scatter_plots
    except ImportError:
        print("Scatter_plot function not found.")
        return

    print("=" * 80)
    print("Executing scatter plot plotting..\n")
    scatter_plots()

    print("Finished scattering plots :)")
    print("=" * 80)


def main():

    args = sys.argv
    assert len(args) >= 2, "Please provide a task."
    task = args[1]
    assert task in ['access_data','concatinate_data', 'initial_impute_clean_data', 'encode_bin_data', 'data_analysis', 'all', 'alt_all'], "Invalid task."
    if task == 'access_data':
        verify_access_data()
    elif task == 'concatinate_data':
        verify_concatinate_data()
    elif task == 'scatter_plots':
        verify_scatter_plots()
    elif task == 'initial_impute_clean_data':
        verify_initial_impute_clean_data()
    elif task == 'encode_bin_data':
        verify_encode_bin_data()
    elif task == 'data_analysis':
        verify_data_analysis()
    elif task == "all":
        verify_access_data()
        verify_concatinate_data()
        verify_scatter_plots()
        verify_initial_impute_clean_data()
        verify_encode_bin_data()
        verify_data_analysis()
        
    elif task == "alt_all":
        verify_concatinate_data()
        verify_scatter_plots()
        verify_initial_impute_clean_data()
        verify_encode_bin_data()
        verify_data_analysis()

if __name__ == "__main__":
    main()
