import pandas as pd
import numpy as np
import logging
from ast import literal_eval
from sklearn.utils import resample
import scipy.stats as st
from scipy.stats import rankdata


logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)


def get_trajec_actions(trajectories, column):
    trajec_actions = pd.DataFrame()
    for index, row in trajectories.iterrows():
        if row['t'] == 0 and index > 0:
            trajec_actions = trajec_actions.append({'actions': actions, 'gender': gender, 'age': age, column: col_traj}, ignore_index=True)
            actions = [row['A_t']]
            col_traj = [row[column]]
            gender = row['gender']
            age = row['age']
        elif index == 0:
            age = row['age']
            actions = [row['A_t']]
            col_traj = [row[column]]
            gender = row['gender']
        else:
            age = row['age']
            actions.append(row['A_t'])
            col_traj.append(row[column])
            gender = row['gender']
    trajec_actions = trajec_actions.append({'actions': actions, 'gender': gender, 'age': age, column: col_traj}, ignore_index=True)
    return trajec_actions


def get_pulse_trajec_actions(pulse_trajecs, column):
    pulse_trajec_actions = pd.DataFrame()
    for index, row in pulse_trajecs.iterrows():
        if row['t'] == 0 and index > 0:
            pulse_trajec_actions = pulse_trajec_actions.append({'actions': actions, 'gender': gender, 'age': age, column: col_traj}, ignore_index=True)
            age = row['age']
            actions = [row['A_t']]
            col_traj = [row[column]]
            gender = row['gender']
        elif index == 0:
            age = row['age']
            actions = [row['A_t']]
            col_traj = [row[column]]
            gender = row['gender']
        else:
            age = row['age']
            actions.append(row['A_t'])
            col_traj.append(row[column])
            gender = row['gender']
    pulse_trajec_actions = pulse_trajec_actions.append({'actions': actions, 'gender': gender, 'age': age, column: col_traj}, ignore_index=True)
    pulse_trajec_actions['icustay_id'] = pulse_trajecs['icustay_id'].unique()
    return pulse_trajec_actions


def find_elements(series, element):
    return series.apply(lambda x: literal_eval(str(x)) == element)


def find_elements_starting_with(series, element):
    return series.apply(lambda x: literal_eval(str(x))[:len(element)] == element)


def find_elements_containing(series, element):
    return series.apply(lambda x: literal_eval(str(element)) in literal_eval(str(x)))


def compute_probs(trajec_actions, column):
    for index, row in trajec_actions.iterrows():
        prob_a_den = 0
        gamma = []
        for i in range(len(row['actions'])):
            prob_a_num = (find_elements(trajec_actions['gender'],row['gender']) & find_elements(trajec_actions['age'],row['age']) & find_elements_starting_with(trajec_actions[column],row[column][:i+1]) & find_elements_starting_with(trajec_actions['actions'],row['actions'][:i+1])).sum()
            if i == 0:
                prob_a_den += (find_elements(trajec_actions['gender'],row['gender']) & find_elements(trajec_actions['age'],row['age']) & find_elements_starting_with(trajec_actions[column],row[column][:i+1])).sum()
            else:
                prob_a_den += ((find_elements_starting_with(trajec_actions[column],row[column][:i+1])) & (find_elements(trajec_actions['gender'],row['gender']))&(find_elements(trajec_actions['age'],row['age']))&(find_elements_starting_with(trajec_actions['actions'],row['actions'][:i]))).sum() -\
                ((find_elements_starting_with(trajec_actions[column],row[column][:i])) & (find_elements(trajec_actions['gender'],row['gender']))&(find_elements(trajec_actions['age'],row['age']))&(find_elements_starting_with(trajec_actions['actions'],row['actions'][:i]))).sum()
            gamma.append(prob_a_num/prob_a_den)
            trajec_actions.at[index, f'prob_a_{i}'] = prob_a_num/prob_a_den
        trajec_actions.at[index, 'prob_a'] = prob_a_num/prob_a_den
    return trajec_actions


def bootstrap_distribution(col, gender, age, action, column_v, trajec_actions, pulse_trajec_actions, n_iter=100):
    df = pd.DataFrame()
    max_y = trajec_actions.loc[find_elements(trajec_actions['gender'], gender) & find_elements(trajec_actions['age'], age) & find_elements_containing(trajec_actions[col], max(column_v)), f'{col}_raw'].max()
    min_y = trajec_actions.loc[find_elements(trajec_actions['gender'], gender) & find_elements(trajec_actions['age'], age) & find_elements_containing(trajec_actions[col], min(column_v)), f'{col}_raw'].min()
    sim_filtered = pulse_trajec_actions[find_elements_starting_with(pulse_trajec_actions[col], column_v) & find_elements(pulse_trajec_actions['gender'], gender) & find_elements(pulse_trajec_actions['age'], age) & find_elements_starting_with(pulse_trajec_actions['actions'], action)].copy()
    real_filtered = trajec_actions[find_elements(trajec_actions[col], column_v) & find_elements(trajec_actions['gender'], gender) & find_elements(pulse_trajec_actions['age'], age) & find_elements_starting_with(trajec_actions['actions'], action)].copy()
    if len(real_filtered) > 1 and len(sim_filtered) > 1:
        for i in range(n_iter):
            real_train = resample(real_filtered, n_samples=len(real_filtered))
            exp_y = real_train[f'{col}_raw'].mean()
            prob = real_train['prob_a'].max()
            sim_train = resample(sim_filtered, n_samples=len(sim_filtered))
            exp_y_sim = sim_train[f'{col}_raw'].mean()
            df = df.append({'Exp_y': exp_y, 'UB': prob*exp_y + (1-prob)*max_y, 'LB': prob*exp_y + (1-prob)*min_y, 'Sim_exp_y': exp_y_sim, 'max_y':max_y, 'min_y': min_y}, ignore_index=True)
        return df
    return None


def bootstrap_distribution_(col, gender, age, action, column_v, trajec_actions, pulse_trajec_actions, n_iter=100, i=3):
    global pulse_data, MIMICtable
    pulse_data = pulse_data.rename(columns={col: f'{col}_raw'})
    MIMICtable = MIMICtable.rename(columns={col: f'{col}_raw'})
    pulse = pulse_trajec_actions[['actions', 'gender', 'age', 'icustay_id', col]].merge(pulse_data[pulse_data['bloc'] == i+2], left_on=['icustay_id', 'gender'], right_on=['icustay_id', 'gender'])
    obs_data = trajec_actions[['actions', 'gender', 'age', 'icustay_id', col, f'prob_a_{i}']].merge(MIMICtable[MIMICtable['bloc'] == i+2], left_on=['icustay_id', 'gender', 'age'], right_on=['icustay_id', 'gender', 'age'])
    df = pd.DataFrame()
    max_y = obs_data.loc[find_elements(obs_data['gender'], gender) & find_elements(obs_data['age'], age) & find_elements_containing(obs_data[col], max(column_v)), f'{col}_raw'].max()
    min_y = obs_data.loc[find_elements(obs_data['gender'], gender) & find_elements(obs_data['age'], age) & find_elements_containing(obs_data[col], min(column_v)), f'{col}_raw'].min()
    sim_filtered = pulse[find_elements(pulse['gender'], gender) & find_elements(pulse['age'], age) & find_elements(pulse[col], column_v) & find_elements_starting_with(pulse['actions'], action)].copy()
    real_filtered = obs_data[find_elements(obs_data['gender'], gender) & find_elements(obs_data['age'], age) & find_elements(obs_data[col], column_v) & find_elements_starting_with(obs_data['actions'], action)].copy()
    if len(real_filtered) > 1 and len(sim_filtered) > 1:
        for j in range(n_iter):
            real_train = resample(real_filtered, n_samples=len(real_filtered))
            exp_y = real_train[f'{col}_raw'].mean()
            prob = real_train[f'prob_a_{i}'].max()
            sim_train = resample(sim_filtered, n_samples=len(sim_filtered))
            exp_y_sim = sim_train[f'{col}_raw'].mean()
            df = df.append({'Exp_y': exp_y, 'UB': prob*exp_y + (1-prob)*max_y, 'LB': prob*exp_y + (1-prob)*min_y, 'Sim_exp_y': exp_y_sim, 'max_y':max_y, 'min_y': min_y}, ignore_index=True)
        return df
    return None


def rejected_hypotheses_bootstrap(col, trajec_actions, pulse_trajec_actions):
    logging.info("calculating rejected hypotheses")
    state_actions = trajec_actions[['gender', 'age', 'actions', col]].copy()
    state_actions.loc[:,'a'] = state_actions['actions'].apply(tuple)
    state_actions.loc[:,'s'] = state_actions[col].apply(tuple)
    state_actions = state_actions.drop_duplicates(['gender', 'age', 'a', 's'])
    total_hypotheses = len(state_actions)
    logging.info(f"Total hypotheses: {total_hypotheses}")
    p_values = pd.DataFrame()
    for index, row in state_actions.iterrows():
        logging.info(f"On index {index}/{total_hypotheses}")
        df = bootstrap_distribution(col, row['gender'], row['age'], row['actions'], row[col], trajec_actions, pulse_trajec_actions)
        if df is not None:
            sigma_ub = (df['UB']-df['Sim_exp_y']).var()
            exp_ub = (df['UB']-df['Sim_exp_y']).mean()
            p_ub = st.norm.cdf(exp_ub/np.sqrt(sigma_ub))
            sigma_lb = (df['Sim_exp_y']-df['LB']).var()
            exp_lb = (df['Sim_exp_y']-df['LB']).mean()
            p_lb = st.norm.cdf(exp_lb/np.sqrt(sigma_lb))
            p_values = p_values.append({'gender': row['gender'], 'age': row['age'], 'actions': row['actions'], col: row[col], 'p_lb': p_lb, 'p_ub': p_ub}, ignore_index=True)
    rej_hyps = p_values[(p_values['p_lb']<0.05/total_hypotheses) ^ (p_values['p_ub']<0.05/total_hypotheses)].copy()
    for index, row in rej_hyps.iterrows():
        rej_hyps.loc[index, 'n_real'] = (find_elements(trajec_actions['gender'], row['gender']) & find_elements(trajec_actions['age'], row['age']) & find_elements(trajec_actions['actions'], row['actions']) & find_elements(trajec_actions[col], row[col])).sum()
        rej_hyps.loc[index, 'n_sim'] = (find_elements(pulse_trajec_actions['gender'], row['gender']) & find_elements(pulse_trajec_actions['age'], row['age']) & find_elements(pulse_trajec_actions['actions'], row['actions']) & find_elements(pulse_trajec_actions[col], row[col])).sum()
    return len(rej_hyps), p_values, rej_hyps


def do_hypothesis_testing(column, MIMICtable, pulse_data, col_bins_num, actionbloc):
    logging.info("doing hypothesis testing")
    pulse_data = pulse_data.rename(columns={column: f'{column}_raw'})
    
    col_ranked = rankdata(MIMICtable[column])/len(MIMICtable)
    col_bins = np.floor((col_ranked + 1/(col_bins_num + 0.0000001))*col_bins_num)
    median_col = [MIMICtable.loc[col_bins==1, column].median(), MIMICtable.loc[col_bins==2, column].median(), MIMICtable.loc[col_bins==3, column].median(), MIMICtable.loc[col_bins==4, column].median()]
    
    MIMICtable = MIMICtable.rename(columns={column: f'{column}_raw'})
    MIMICtable[column] = col_bins
    pulse_data = pulse_data.merge(MIMICtable[['age', 'icustay_id', 'bloc', column]], left_on=['icustay_id', 'bloc'], right_on=['icustay_id', 'bloc'])
    
    trajectories = pd.DataFrame()
    trajectories['t'] = np.arange(len(MIMICtable))%5
    trajectories['icustay_id'] = MIMICtable['icustay_id']
    trajectories['gender'] = MIMICtable['gender']
    trajectories['age'] = MIMICtable['age']
    trajectories[column] = MIMICtable[column]
    trajectories['A_t'] = actionbloc['action_bloc']
    trajectories = trajectories[trajectories['t']!=4]
    
    pulse_trajecs = pd.DataFrame()
    pulse_trajecs['t'] = np.arange(len(pulse_data))%5
    pulse_trajecs['icustay_id'] = pulse_data['icustay_id']
    pulse_trajecs = pulse_trajecs[pulse_trajecs['t']!=4]
    pulse_trajecs = pulse_trajecs.merge(trajectories[['t','icustay_id', 'A_t', 'gender', 'age', column]], left_on=['icustay_id', 't'], right_on=['icustay_id', 't'])
    
    trajec_actions = get_trajec_actions(trajectories, column)
    pulse_trajec_actions = get_pulse_trajec_actions(pulse_trajecs, column)
    trajec_actions = compute_probs(trajec_actions, column)
    
    icustayids = MIMICtable['icustay_id'].unique()
    trajec_actions['icustay_id'] = icustayids
    
    mimic_data_last_time = MIMICtable[MIMICtable['bloc'] == 5].drop(columns=column)
    trajec_actions = trajec_actions.merge(mimic_data_last_time, left_on=['icustay_id', 'gender', 'age'], right_on=['icustay_id', 'gender', 'age'])

    pulse_data_last_time = pulse_data[pulse_data['bloc'] == 5].drop(columns=column)
    pulse_trajec_actions = pulse_trajec_actions.merge(pulse_data_last_time, left_on=['icustay_id', 'gender', 'age'], right_on=['icustay_id', 'gender', 'age'])
    
    num_rej_hyps, p_values, rej_hyps = rejected_hypotheses_bootstrap(column, trajec_actions, pulse_trajec_actions)
    return num_rej_hyps, p_values, rej_hyps, trajec_actions, pulse_trajec_actions