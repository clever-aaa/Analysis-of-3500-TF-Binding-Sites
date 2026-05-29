import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
import optuna
import time
from sklearn.metrics import accuracy_score, log_loss
import matplotlib.pyplot as plt
import os
from transformers import AutoTokenizer, AutoModel
import torch

def read_raw_data(results_file:str, seq_file:str = None) -> pd.DataFrame:
    """
    read experiment data from csv file, if the second is None or wrong, then return results file only
    :param results_file: address of result file
    :param seq_file: address of sequence information file
    :return: merged data from the files with right correspondence
    """
    if not(os.path.exists(results_file)): #if results file not exist, return empty DataFrame
        print(f"file {results_file} not exists")
        return pd.DataFrame()
    results_df = pd.read_csv(results_file)
    # if there is seq_file, then read and merge
    if os.path.exists(seq_file):
        seq_df=pd.read_csv(seq_file)
        # merge the two dataframe with right correspondence
        all_data = pd.merge(results_df, seq_df, on='RNAME', how='left', indicator=True)
        missing_data = all_data[all_data['_merge'] == 'left_only']
        if len(missing_data):
            print(f'Warning: {missing_data} have no corresponding sequences')
        print('get all data as:')
        print(all_data.head())
        return all_data
    # if seq_file is not given, then check if seqs are already in df
    if seq_file is None :
        if not('CREs' in results_df.columns): # if not, then warning
            print(f'Warning: no sequence file to corresponding to results')
    else : # if the seq_file is wrong, then waring
        print(f'file {seq_file} not exists')
    return results_df

def DNA_BERT_2(sequence:str) -> tuple[pd.DataFrame,pd.DataFrame]:
    """
    get embedded vector through DNABERT2 model
    :param sequence: DNA sequence such as 'ACTACAATGG'
    :return: vector (1x768 dim) of mean pooling and max pooling
    """
    tokenizer = AutoTokenizer.from_pretrained("DNABERT-2-117M", local_files_only=True, trust_remote_code=True)
    model = AutoModel.from_pretrained("DNABERT-2-117M", local_files_only=True, trust_remote_code=True)

    inputs = tokenizer(sequence, return_tensors='pt')["input_ids"]
    hidden_states = model(inputs)[0]

    # embedding with mean pooling
    embedding_mean = torch.mean(hidden_states[0], dim=0)
    # embedding with max pooling
    embedding_max = torch.max(hidden_states[0], dim=0)[0]

    return pd.DataFrame(embedding_mean.detach().numpy().reshape(1, -1)), pd.DataFrame(embedding_max.detach().numpy().reshape(1, -1))

def get_embedded_data(group: str, results_addr: str) -> None:
    """
    get BERT2 embedded vector and other information, and save in
    "data/embedded data for {group} (full).csv" while "data/Processed Full Data.csv" is unembedded data
    :param group: T2, T3, T4,..., the group to be selected and embedded and save
    :param results_addr: address of results file (data/full_enrichment_results_t1_t4.csv or data/full_enrichment_results_t5_t8.csv)
    :return: None
    """
    # read unembedded data
    seqs_addr = 'data/Copy of strict3500.csv'
    sheet2 = 'oligo_pool_strict'
    data = read_raw_data(results_addr, seqs_addr)
    data['flank_length'] = data["5'_flank"].str.len()
    data2 = pd.read_excel('data/Copy of strict3500.xlsx', sheet_name=sheet2)
    # merge data
    data = pd.merge(data, data2, on='CREs', how='left', indicator='merge_gc')  # merge gc_content mainly
    missing_data = data[data['merge_gc'] == 'left_only']
    if len(missing_data):
        print(f'Warning: {missing_data} have no corresponding sequences')
    else:
        data = data.drop('merge_gc', axis=1)
    # extract cols we need
    results = data[
        ['sample', 'odds_ratio', 'original_sequence', 'CREs', 'RNAME', 'gc_content', 'flank_length', 'n_repeats', 'p_adj']].copy()
    # order for different sequence
    duplicated = data.drop_duplicates(subset=['original_sequence'])#data[data['original_sequence'].duplicated(keep=False)]
    seq2id=duplicated[['original_sequence']].reset_index(drop=True)
    seq2id['tf_id']=seq2id.index+1
    seq2id.to_csv('data/seq2id.csv', index=False)  # save the corresponding seq and id
    n_repeats = results['n_repeats'].max()  # the max repeat num
    for idx, row in results.iterrows():
        cur_count = row['n_repeats']
        for i in range(n_repeats):
            col_name = 'tf' + str(i + 1)
            if i + 1 <= cur_count:
                mask = seq2id['original_sequence'] == row['original_sequence']
                if not(mask.any()):
                    new_id = seq2id['tf_id'].max() + 1 if not seq2id.empty else 0
                    seq2id.loc[len(seq2id)] = [row['original_sequence'], new_id]
                tf_id=seq2id.loc[seq2id['original_sequence']==row['original_sequence'],'tf_id'].values[0]
                results.loc[idx, col_name] = tf_id  # in this data, every position is the same seq
            else:
                results.loc[idx, col_name] = 0  # 0 is None
    # save unembedded data
    save_path = "data/Processed Full Data.csv"
    results.to_csv(save_path, index=False, encoding='utf-8')
    print('-' * 50)
    print(results.head())

    # divide data into group T2 T3 and T4
    selected_data = results[results['sample'] == group].reset_index(drop=True)

    # get embedded data and save
    embedded_seq = []
    for idx, row in selected_data.iterrows():
        seq = row['CREs']
        RNAME = row['RNAME']
        embedded_temp = DNA_BERT_2(seq)[0]
        embedded_temp['RNAME'] = RNAME
        embedded_seq.append(embedded_temp)
    embedded_seq = pd.concat(embedded_seq)
    embedded_data = pd.merge(embedded_seq, selected_data, on='RNAME', how='left', indicator='merge_embedded')
    missing_data = embedded_data[embedded_data['merge_embedded'] == 'left_only']
    if len(missing_data):
        print(f'Warning: \n{missing_data.head()} \nhave no corresponding sequences')
    else:
        embedded_data = embedded_data.drop('merge_embedded', axis=1)
    print(embedded_data.head())
    save_path = f"data/embedded data for {group} (full).csv"
    embedded_data.to_csv(save_path, index=False, encoding='utf-8')
    return

def get_xy(data_addr: str, threshold: list, split_rate: float = 0.1) -> tuple:
    """
    get x and y data from csv file (saved data can be directly used to training model)
    :param data_addr: address of data (.csv)
    :param threshold: 1x4 dim, threshold to split y to 5 classes
    :param split_rate: split rate of training and test set (=test set rate)
    :return: train_x, train_y, test_x, test_y (DataFrame)
    """
    # directly read processed data
    file_path = data_addr
    embedded_data = pd.read_csv(file_path)
    embedded_data = embedded_data[embedded_data['p_adj'] < 0.05] # select data has suitable p value

    # manual division
    seq_embedded = [str(i) for i in range(768)]
    x_embed = embedded_data[seq_embedded]
    tf_ids = [col for col in embedded_data.columns if str(col).startswith('tf') and str(col)[2:].isdigit()]
    x_grammar = embedded_data[['gc_content', 'flank_length']]
    x_tf_ids = embedded_data[tf_ids]
    x_tf_ids_sorted = x_tf_ids.apply(lambda row: sorted(row, key=lambda x: x != 0), axis=1, result_type='expand')
    x_tf_ids_sorted.columns = x_tf_ids.columns
    x = pd.concat([x_embed, x_grammar, x_tf_ids_sorted], axis=1)
    embedded_data.loc[embedded_data['odds_ratio'] < threshold[0], 'y'] = 0
    embedded_data.loc[(embedded_data['odds_ratio'] >= threshold[0]) & (embedded_data['odds_ratio'] < threshold[1]), 'y'] = 1
    embedded_data.loc[(embedded_data['odds_ratio'] >= threshold[1]) & (embedded_data['odds_ratio'] < threshold[2]), 'y'] = 2
    embedded_data.loc[(embedded_data['odds_ratio'] >= threshold[2]) & (embedded_data['odds_ratio'] < threshold[3]), 'y'] = 3
    embedded_data.loc[embedded_data['odds_ratio'] >= threshold[3], 'y'] = 4

    y = embedded_data[['y', 'odds_ratio']]
    T2_data_0 = embedded_data[embedded_data['y'] == 0]
    print(f'class 0 number: {T2_data_0.shape[0]}')
    T2_data_1 = embedded_data[embedded_data['y'] == 1]
    print(f'class 1 number: {T2_data_1.shape[0]}')
    T2_data_2 = embedded_data[embedded_data['y'] == 2]
    print(f'class 2 number: {T2_data_2.shape[0]}')
    T2_data_3 = embedded_data[embedded_data['y'] == 3]
    print(f'class 3 number: {T2_data_3.shape[0]}')
    T2_data_4 = embedded_data[embedded_data['y'] == 4]
    print(f'class 4 number: {T2_data_4.shape[0]}')

    print('get x:\n', x.head())
    print('get y:\n', y.head())

    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=split_rate, random_state=42)
    return train_x, train_y, test_x, test_y

def visualize_scatter(test_y: pd.DataFrame, pred_y: pd.DataFrame, plot_name: str, group: str) -> None:
    '''
    visualize predict y and real y through scatter plot, it will be saved in directory "plots"
    :param test_y: experimental data include column 'odds ratio'
    :param pred_y: predicted data include column 'odds ratio'
    :param group: group of test data (T2, T3,...) - for plot title
    :param plot_name: name of the plot to be saved
    :return: None
    '''
    # calculate the predict accuracy
    accuracy = accuracy_score(test_y['odds_ratio'], pred_y['odds_ratio'])
    print(f'accuracy: {accuracy:.2f}')
    x_range = range(1, len(test_y) + 1)
    plt.figure(figsize=(15, 6))
    plt.scatter(x_range, test_y['odds_ratio'], color='red', s=5, alpha=0.5, label='test data')
    plt.scatter(x_range, pred_y['odds_ratio'], color='blue', s=2, alpha=0.5, label='predict data')
    plt.xlabel('Index')
    plt.ylabel('Odds Ratio')
    plt.title(f'Real and Predict Odds Ratio of {group}\n' + f'accuracy={accuracy:.2f}')
    plt.grid(True, alpha=0.3)
    plt.legend()

    os.makedirs('results/plots', exist_ok=True)
    addr = f'results/plots/{plot_name}.jpg'
    plt.savefig(addr, dpi=300, bbox_inches='tight')
    plt.close()

def predict_save(data: dict, model, csv_name: str) -> None:
    """
    save more details about predict for classify model,
    including predict score for every class, predict class, correct class
    :param data: data['x'] is a DataFrame of input x, while data['y'] is a DataFrame of output y
    :param model: trained model
    :param csv_name: name of the csv file to be saved, it better include dir address
    :return: None
    """
    x = data['x']
    predict_y = model.predict(x)
    predict_y = pd.DataFrame(predict_y, columns=['predict_y'])
    proba_y = model.predict_proba(x)
    proba_y = pd.DataFrame(proba_y, columns=['proba_0', 'proba_1', 'proba_2', 'proba_3', 'proba_4'])
    details = pd.concat([data['y'].reset_index(drop=True), data['origin_y'].reset_index(drop=True), predict_y, proba_y],
                        axis=1)
    details.to_csv(csv_name, index=True)
    return None

def run_XGB_training(data: dict, parameters: dict) ->tuple:
    """
    train a XGBoost model use input data and parameters
    :param data: dict contains x,y respectively for training and test, and odds ratio of test set
    :param parameters: dict contains parameters like n_estimators, learning_rate,...,
    :return: logloss and trained model
    """
    n_estimators = parameters['n_estimators']
    learning_rate = parameters['learning_rate']
    max_depth = parameters['max_depth']
    reg_lambda = parameters['reg_lambda']
    reg_alpha = parameters['reg_alpha']
    min_child_weight = parameters['min_child_weight']
    subsample = parameters['subsample']
    gamma = parameters['gamma']
    colsample_bytree = parameters['colsample_bytree']

    early_stop = xgb.callback.EarlyStopping(
        rounds=100, metric_name='mlogloss', data_name='validation_1', save_best=True, maximize=False
    )
    clf = xgb.XGBClassifier(
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        max_depth=max_depth,
        reg_lambda=reg_lambda,
        reg_alpha=reg_alpha,
        min_child_weight=min_child_weight,
        gamma=gamma,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        callbacks=[early_stop]
    )
    train_x = data['train_x']
    train_y = data['train_y']
    test_x = data['test_x']
    test_y = data['test_y']
    clf = clf.fit(train_x, train_y,
                  eval_set=[(train_x, train_y), (test_x, test_y)],
                  verbose=100
                  )
    print(f"Best iteration: {clf.best_iteration}")
    print(f"Best score: {clf.best_score}")
    y_pred_proba = clf.predict_proba(test_x)
    logloss = log_loss(test_y, y_pred_proba)

    print(f"Multi-class Log Loss: {logloss:.4f}")
    return logloss, clf

def XGB_objective(trial)->float:
    """
    XGBoost objective function for Bayesian hyperparameter optimization
    :return: logloss
    """
    gamma = trial.suggest_float(name="gamma", low=0, high=5, step=0.1)
    learning_rate = trial.suggest_float(name="learning_rate", low=1e-4, high=0.3, log=True)
    n_estimators = trial.suggest_int(name="n_estimators", low=10, high=2000, step=1)
    max_depth = trial.suggest_int(name="max_depth", low=3, high=10, step=1)
    reg_lambda = trial.suggest_int(name="reg_lambda", low=0, high=2, step=1)
    reg_alpha = trial.suggest_int(name="reg_alpha", low=0, high=2, step=1)
    min_child_weight = trial.suggest_float(name="min_child_weight", low=1e-3, high=10, log=True)
    subsample = trial.suggest_float(name="subsample", low=0.5, high=1.0, step=0.1)
    colsample_bytree = trial.suggest_float(name="colsample_bytree", low=0.3, high=1.0,step=0.1)

    parameters={
        'n_estimators': n_estimators,
        'learning_rate': learning_rate,
        'max_depth': max_depth,
        'reg_lambda': reg_lambda,
        'reg_alpha': reg_alpha,
        'min_child_weight': min_child_weight,
        'gamma':gamma,
        'subsample': subsample,
        'colsample_bytree': colsample_bytree
    }

    logloss, XGBmodel = run_XGB_training(data, parameters=parameters)

    return logloss

def optimizer_optuna(n_iters:int, algo:str, optuna_objective):
    """
    main function of optimize parameters
    :param n_iters: iterations to search
    :param algo: choose optimizer to use ('TPE' or 'GP')
    :param optuna_objective: objective function to optimize
    :return: optimizer -> study (to use: study.best_trial.params and study.best_trial.values)
    """
    if algo == 'TPE':
        algo = optuna.samplers.TPESampler(n_startup_trials=15,n_ei_candidates=24)
    elif algo == 'GP':
        from optuna.integration import SkoptSampler
        algo = SkoptSampler(skopt_kwargs={
            'base_estimator':'GP',
            'n_initial_points':30,
            'acq_func':'EI'
        })

    study=optuna.create_study(
        sampler=algo,
        direction='minimize'
    )
    study.optimize(
        optuna_objective,
        n_trials=n_iters,
        show_progress_bar=True
    )
    print(f"best params: {study.best_trial.params}")
    print(f"best score: {study.best_trial.values}")
    return study

if "__main__"==__name__:
    start_time = time.time()
    group='T2' # or T3, T4,...,
    results_addr = 'data/full_enrichment_results_t1_t4.csv' # or 'data/full_enrichment_results_t5_t8.csv'
    split_rate = 0.2 # test set rate
    threshold = [0.8, 1, 1.2, 1.5] # for splitting y to 5 class (0,1,2,3,4)
    embedded_path = f'data/embedded data for {group} (full).csv' # default path save in function"get_embedded_data"
    n_trials = 100 # iterations for optimize parameters
    algo = 'TPE'

    get_embedded_data(group,results_addr)

    train_x, train_y, test_x, test_y = get_xy(embedded_path, threshold, split_rate)
    test_origin = test_y['odds_ratio']
    train_origin = train_y['odds_ratio']
    test_y=test_y['y']
    train_y=train_y['y']
    data = {
        'train_x': train_x,
        'train_y': train_y,
        'test_x' : test_x,
        'test_y' : test_y,
        'test_y_origin': test_origin
    }

    # XGBoost
    study = optimizer_optuna(n_trials, algo, XGB_objective)
    parameters = study.best_trial.params
    auc, XGBmodel = run_XGB_training(data, parameters)

    os.makedirs('results/model', exist_ok=True)
    import joblib
    joblib.dump(XGBmodel, 'results/model/XGBoost_model.pkl')
    # or directly read model has been trained, e.g. XGBmodel = joblib.load('model/XGBoost_model.pkl')

    # visualize logloss change in training
    results = XGBmodel.evals_result()
    plt.figure(figsize=[10, 6])
    x_axis = range(len(results['validation_1']['mlogloss']))
    plt.plot(x_axis, results['validation_0']['mlogloss'], label='train')
    plt.plot(x_axis, results['validation_1']['mlogloss'], label='test')
    plt.xlabel('epochs')
    plt.ylabel('logloss')
    plt.title('XGBoost Logloss Curve')
    plt.legend()
    os.makedirs('results/plots', exist_ok=True)
    plt.savefig(f'results/plots/Logloss.png', transparent=False, dpi=300, format='png',
                bbox_inches='tight')
    plt.show()
    plt.close()

    # visualization prediction result of test set
    if isinstance(test_y, pd.Series):
        test_y = pd.DataFrame(test_y, columns=[test_y.name])
    test_y = test_y.rename(columns={'y': 'odds_ratio'})
    pre_y = XGBmodel.predict(test_x)
    pre_y = pd.DataFrame(pre_y, columns=['odds_ratio'])
    visualize_scatter(test_y, pre_y, 'XGBoost-TestT2', group)
    # visualization prediction result of train set
    if isinstance(train_y, pd.Series):
        train_y = pd.DataFrame(train_y, columns=[train_y.name])
    train_y = train_y.rename(columns={'y': 'odds_ratio'})
    pre_y = XGBmodel.predict(train_x)
    pre_y = pd.DataFrame(pre_y, columns=['odds_ratio'])
    visualize_scatter(train_y, pre_y, 'XGBoost-TrainT2', group)

    os.makedirs('results', exist_ok=True)
    test_data = {
        'x': test_x,
        'y': test_y,
        'origin_y': test_origin
    }
    predict_save(test_data, XGBmodel, f'results/{group}-predict_details.csv')

    end_time=time.time()
    print(f'time: {(end_time-start_time)/60} minutes')
