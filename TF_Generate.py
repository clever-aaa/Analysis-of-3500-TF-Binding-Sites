import os
from typing import Any
import cell_state as cst
import pandas as pd
import joblib
import random
import time
import numpy as np
import math
import json

def get_iterate_origin(target_class:Any, tf_columns:list) -> list:
    """
    :param target_class: 0 or 1 (0 is select tfs with the lowest odds ratio, while 1 is the highest tfs)
    :param tf_columns: column names of tfs
    :return: [] or [lists] e.g.[[1,3,76,43,0,0],[...],...,]
    """
    if target_class==[]:
        return []
    id_data = pd.read_csv('../data/embedded data for T8 (full).csv', parse_dates=False)
    id_data = id_data[['tf_id', "odds_ratio",*tf_columns]]
    print(id_data.head())
    if target_class==0:
        result = id_data.nsmallest(QUANTITY, 'odds_ratio')[tf_columns].values.tolist()
    elif target_class==1:
        result = id_data.nlargest(QUANTITY, 'odds_ratio')[tf_columns].values.tolist()
    else:
        print('Target is not available (0 or 1)')
        result=[]
    return result

def handle_one_tf(tf:list, all_tf_ids: list, poss: float = 0.5, n_tf: int = 6)->list:
    """
    handle one tf, if it exists, change it, else generate a new one.
    :param tf: tf, if it is empty, then generate a new one.
    :param all_tf_ids: ids of all tfs to be chosen.
    :param poss: possibility to change tf, 1-poss is the possibility to change an order. default=0.5
    :param n_tf: number of tfs to choose. default=6
    :return: a 6 dimensional list includes tf_ids, e.g.[4,6,9,0,0,0] (0 will at the bottom of the list)
    """
    if len(tf)==0: # if no tf, generate a new one
        choose_tf=np.random.choice(all_tf_ids, size=n_tf, replace=True).tolist()
        result = [x for x in choose_tf if x != 0] + [x for x in choose_tf if x == 0]
        log['operate'].append('None')
        return result
    # if it is a tf, handle an iteration
    replace=0
    exchange=0
    if random.random()<poss:
        replace=1
    if random.random()>poss:
        exchange=1
    if not (replace+exchange):
        replace=1
    if not(0 in all_tf_ids):
        all_tf_ids=np.append(all_tf_ids,0)
    if replace: # change tf
        choose_tf=np.random.choice(all_tf_ids, size=1).tolist()[0]
        choose_pos=np.random.choice([i for i in range(N_tfs)], size=1)[0]
        log['operate'].append(f'{tf}[{choose_pos}]: {tf[choose_pos]} -> {choose_tf}')
        tf[choose_pos]=choose_tf
    if exchange: # change order
        choose_pos=np.random.choice([i for i in range(N_tfs)], size=2).tolist()
        if replace:
            log['operate'][-1]+=f'|exchange {tf}:{choose_pos[0]} <-> {choose_pos[1]}'
        else:
            log['operate'].append(f'exchange {tf}:{choose_pos[0]} <-> {choose_pos[1]}')
        if tf[max(choose_pos)]!=0:
            temp=tf[choose_pos[0]]
            tf[choose_pos[0]]=tf[choose_pos[1]]
            tf[choose_pos[1]]=temp
    result = [x for x in tf if x != 0] + [x for x in tf if x == 0]
    return result

def calculate_gc_content(seq: str) -> float:
    """Calculate GC content of a sequence."""
    seq = seq.upper()
    gc_count = seq.count('G') + seq.count('C')
    return gc_count / len(seq) if len(seq) > 0 else 0

def evaluate_seq(all_data: pd.DataFrame, tf_ids: list, model) -> dict:
    """
    get other information from given combination (tf_ids)
    :param all_data: all basic data needed, including columns 'original_sequence', 'tfs' and 'tf_id'
    :param tf_ids: a 6 dimensional list includes tf_ids, e.g.[4,6,9,0,0,0]
    :param model: trained model
    :return: a dict of a tf combination, includes sequence, predict class,...,
    """
    tfs=[]
    tf_detail=[]
    origin_scores=[]
    for tf_id in tf_ids:
        if tf_id==0:
            break
        #print(tf_id)
        tf_seq=all_data[all_data['tf_id'] == tf_id]['original_sequence'].iloc[0]
        tf_name=all_data[all_data['tf_id'] == tf_id]['tfs'].iloc[0]
        origin_score=all_data[all_data['tf_id'] == tf_id]['odds_ratio'].iloc[0]
        tf_detail.append(tf_name)
        tfs.append(tf_seq)
        origin_scores.append(origin_score)
    current_gc=calculate_gc_content(''.join(tfs))
    if current_gc>Target_GC:
        spacer=SPACERS["low_gc"]
    elif current_gc<Target_GC:
        spacer=SPACERS["high_gc"]
    else:
        spacer=SPACERS["medium_gc"]
    sequence=spacer.join(tfs)

    X_embedded=cst.DNA_BERT_2(sequence)[0]
    X_grammar=pd.DataFrame({
        'gc_content':[calculate_gc_content(sequence)],
        'flank_length':[27]
    })
    col_ids=[f'tf{i+1}' for i in range(len(tf_ids))]
    X_ids=pd.DataFrame([tf_ids],columns=col_ids)
    X=pd.concat([X_embedded,X_grammar,X_ids],axis=1)
    print(f'X:\n{X}')
    y=model.predict(X)
    y_proba=model.predict_proba(X)

    tf = {
        'sequence': sequence,
        'predict_class':int(y[0]),
        'predict_proba':[round(float(x),3) for x in y_proba[0]],
        'tf_order':tf_ids,
        'origin_score':origin_scores,
        'detail_order':tf_detail
    }
    return tf

def main(tfs: list):
    """Input origin tfs and optimize by annealing algorithm"""
    # parameters of annealing
    STEPS = 600
    MAX_T = 100
    MIN_T = 0.001
    LOGMAX = np.log10(MAX_T)
    LOGMIN = np.log10(MIN_T)
    LOGINC = (LOGMAX - LOGMIN) / STEPS
    random.seed(42)
    # get data we need
    tf_data = pd.read_excel('data/Copy of strict3500.xlsx', sheet_name='oligo_pool_strict',dtype={'tfs': str})
    tf_data = tf_data[["original_sequence", "CREs", "tfs"]]
    id_data=pd.read_csv(embedded_data_path, parse_dates=False)
    id_data = id_data[['CREs', 'tf_id', "odds_ratio"]]
    data = pd.merge(id_data, tf_data, on='CREs', how='left', indicator='merge')
    missing_data = data[data['merge'] == 'left_only']
    if len(missing_data):
        print(f'Warning: {missing_data} have no corresponding sequences')
        data = data[data['merge'] == 'both'].drop('merge', axis=1)
    else:
        data = data.drop('merge', axis=1)

    # Start Annealing
    results=[]
    tf_ids = np.array(data['tf_id'])
    for tf_n in range(QUANTITY):
        no_remarkable_better = 0
        step = 0
        if tfs:
            curr_tf=evaluate_seq(data, tfs[tf_n], XGBmodel)
        else:
            curr_tf=handle_one_tf([], tf_ids)
            curr_tf=evaluate_seq(data, curr_tf, XGBmodel)

        for i in range(N_tfs):
            log[tf_columns[i]].append(curr_tf['tf_order'][i])
        log['score'].append(curr_tf['predict_proba'][Target])
        log['iter'].append(step)
        log['tf_id'].append(tf_n + 1)
        log['operate'].append('None')

        while no_remarkable_better<20: #or step<MAX_T:
            if step < STEPS:
                logT = LOGMAX - step * LOGINC
                t = math.pow(10, logT)  # /100？
            else:
                t = 0

            new_tf = handle_one_tf(curr_tf['tf_order'], tf_ids)
            new_tf = evaluate_seq(data, new_tf, XGBmodel)
            Better = False
            # now use the target class predict probability to calculate if it is better
            if new_tf['predict_proba'][Target] > curr_tf['predict_proba'][Target]:
                better_diff=new_tf['predict_proba'][Target] - curr_tf['predict_proba'][Target]
                if better_diff < remarkable_threshold:
                    no_remarkable_better+=1
                else:
                    no_remarkable_better=0
                Better = True
            else:
                no_remarkable_better+=1

            if Better:
                curr_tf = new_tf
            else:
                if random.random() < np.exp((new_tf['predict_proba'][Target]-curr_tf['predict_proba'][Target])/t):
                    curr_tf = new_tf
            step += 1

            for i in range(N_tfs):
                log[tf_columns[i]].append(curr_tf['tf_order'][i])
            log['score'].append(curr_tf['predict_proba'][Target])
            log['iter'].append(step)
            log['tf_id'].append(tf_n+1)

        results.append(curr_tf)
    results = sorted(results, key=lambda x: (x['predict_class'], x['predict_proba'][Target]),reverse=True)
    for n, item in enumerate(results, start=1):
        item['id'] = f'comb{n}'
    os.makedirs('results',exist_ok=True)
    with open(f'../results/generate-class{Target}.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

if __name__ == '__main__':
    start_time=time.time()

    SPACERS = {
        "high_gc": "ACGTATGTCGAGTTTAC",  # GC content: 0.53
        "medium_gc": "ACGTATGTCGAGTTTA",  # GC content: 0.47
        "low_gc": "ACACGTTCTAGC"  # GC content: 0.42
    }
    Target_GC = 0.5
    QUANTITY = 20
    N_tfs = 6
    Target=0 # the value want to maximize, while 0 means class 0 predict probability
    remarkable_threshold = 0.005
    tf_columns = [f'tf{i + 1}' for i in range(N_tfs)]
    embedded_data_path='../data/embedded data for T2 (full).csv'

    tfs=get_iterate_origin(0,tf_columns) # iterate origin (if input is empty, then start with random)
    print(f'get iterate origin: {tfs}')

    # load model
    XGBmodel = joblib.load('../results/model/XGBoost_model.pkl')
    log={
        'iter':[],
        'tf_id': [],
        **{f'tf{i + 1}': [] for i in range(N_tfs)},
        'score':[],
        'operate':[]
    }

    main(tfs)

    log=pd.DataFrame(log)
    log.to_csv(f'../results/generate-class{Target}.csv', index=False)

    end_time = time.time()
    print(f'Total time: {(end_time - start_time)/60} minutes')