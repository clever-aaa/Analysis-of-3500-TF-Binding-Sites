import pandas as pd

def read_raw_data(results_file:str, seq_file:str =None) -> pd.DataFrame:
    """
    param files: address of result file and sequence file,
                if the second is none or wrong, then return results file only
    -------
    return: merged data from the files with right correspondence
    """
    if not(os.path.exists(results_file)): #if file exist
        print(f"file {results_file} not exists")
        return pd.DataFrame()
    results_df = pd.read_csv(results_file)
    # if there is seq_file, then read and merge
    if os.path.exists(seq_file):
        seq_df=pd.read_csv(seq_file)
        # merge the two dataframe with right correspondence
        data = pd.merge(results_df, seq_df, on='RNAME', how='left', indicator=True)
        missing_data = data[data['_merge'] == 'left_only']
        if len(missing_data):
            print(f'Warning: {missing_data} have no corresponding sequences')
        print('get all data as:')
        print(data.head())
        return data
    # if seq_file is not given, then check if seqs are already in df
    if seq_file is None :
        if not('CREs' in results_df.columns): # if not, then warning
            print(f'Warning: no sequence file to corresponding to results')
    else : # if the seq_file is wrong, then waring
        print(f'file {seq_file} not exists')
    return results_df

def visualize_scatter(test_y:pd.DataFrame, pred_y: pd.DataFrame, plot_name: str, group: str) -> None:
    '''
    :param test_y: experimental data include column 'odds ratio'
    :param pred_y: predicted data include column 'odds ratio'
    :param group: group of test data (T2, T3,...) - for plot title
    :param plot_name: name of the plot to be saved
    -------
    :return: None
    '''
    #calculate the predict accuracy
    r2 = r2_score(test_y['odds_ratio'], pred_y['odds_ratio'])
    print(f'r2: {r2:.2f}')
    x_range=range(1,len(test_y)+1)
    plt.figure(figsize=(15, 6))
    plt.scatter(x_range, test_y['odds_ratio'], color='red', s=3, alpha=0.5, label='test data')
    plt.plot(x_range, pred_y['odds_ratio'], color='blue', alpha=0.5, label='predict data')
    plt.xlabel('Index')
    plt.ylabel('Odds Ratio')
    plt.title(f'Real and Predict Odds Ratio of {group}\n'+f'R²={r2:.2f}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    addr=f'plots/{plot_name}.jpg'
    plt.savefig(addr, dpi=300, bbox_inches='tight')
    plt.close()

def visualize_raw_data(data: pd.DataFrame,addr='plots/raw_data/raw_data_or.png') -> None:
    '''
    scatter data to get general analysis
    :param data: all data (contain T2, T3 and T4)
    -------
    :return:None
    '''
    T2_data = data[data['sample'] == 'T2'].reset_index()
    T3_data = data[data['sample'] == 'T3'].reset_index()
    T4_data = data[data['sample'] == 'T4'].reset_index()

    plt.figure(figsize=(20, 6))
    plt.scatter(range(1,len(T2_data)+1), T2_data['odds_ratio'], label='T2', color='red', s=2, alpha=0.5)
    plt.scatter(range(1,len(T3_data)+1), T3_data['odds_ratio'], label='T3', color='blue', s=2, alpha=0.5)
    plt.scatter(range(1,len(T4_data)+1), T4_data['odds_ratio'], label='T4', color='green', s=2, alpha=0.5)
    plt.xlabel('Index')
    plt.ylabel('Odds Ratio')
    plt.title(f'T2, T3, T4 Odds Ratio')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.savefig(addr, dpi=300,bbox_inches='tight')
    plt.close()
