import os
import re
import argparse
import wandb
import json
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

import statsmodels.stats.proportion as smp
from sklearn.metrics.pairwise import cosine_similarity
import choix
from sentence_transformers import SentenceTransformer

# Set style for plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("talk")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['savefig.dpi'] = 300

def clean_value_name(value_name):
    """Clean value name for display in plots."""
    if value_name.startswith('being as '):
        value_name = value_name[9:]
    if '(i.e.' in value_name:
        value_name = value_name.split('(i.e.')[0].strip()   
    if value_name.endswith(' as possible'):
        value_name = value_name[:-12]
    return value_name.capitalize()

def load_data_files(data_dir):
    """Load all CSV files from the specified directory."""
    files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    data = {}
    
    for file in files:
        try:
            file_path = os.path.join(data_dir, file)
            df = pd.read_csv(file_path)
            model_name = file.split('.csv')[0]
            data[model_name] = df
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    return data

def combine_data(model_data):
    """Combine all model data into a single DataFrame."""
    dfs = []
    for model_name, df in model_data.items():
        if 'model' not in df.columns:
            df['model'] = model_name
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def compute_binomial_ci(successes, total):
    if total == 0:
        return 0, 0
        
    p_hat = successes / total
    z = 1.96  # 95% confidence level
    se = np.sqrt(p_hat * (1 - p_hat) / total)
    ci_lower = max(0, p_hat - z * se)
    ci_upper = min(1, p_hat + z * se)
    
    return ci_lower, ci_upper

def compute_continuous_ci(mean, std, n):
    if n == 0:
        return 0, 0
        
    z = 1.96  # 95% confidence level
    se = std / np.sqrt(n)  # standard error = std / sqrt(n)
    ci_lower = mean - z * se
    ci_upper = mean + z * se
    
    return ci_lower, ci_upper

def compute_average_diversity(df):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    df['sorted_values'] = df.apply(
        lambda row: tuple(sorted([row['value1'], row['value2']])), 
        axis=1)
    diversity_scores = []
    for sorted_values, group in df.groupby('sorted_values'):
        descriptions = group['description'].tolist()
        if len(descriptions) > 1:  # Need at least 2 descriptions to compute similarity
            embeddings = model.encode(descriptions)
            similarities = cosine_similarity(embeddings)
            n = similarities.shape[0]
            mask = ~np.eye(n, dtype=bool)
            avg_similarity = similarities[mask].mean()
            diversity_scores.append(1 - avg_similarity)
            
    if not diversity_scores:
        return {'mean': 0, 'ci_lower': 0, 'ci_upper': 0}
        
    mean_diversity = np.mean(diversity_scores)
    std_diversity = np.std(diversity_scores)
    n = len(diversity_scores)
    ci_lower, ci_upper = compute_continuous_ci(mean_diversity, std_diversity, n)
    
    return {
        'mean': mean_diversity,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper
    }

def compute_binary_ci(successes, total):
    if total == 0:
        return 0, 0
        
    p_hat = successes / total
    z = 1.96  # 95% confidence level
    se = np.sqrt(p_hat * (1 - p_hat) / total)
    ci_lower = max(0, p_hat - z * se)
    ci_upper = min(1, p_hat + z * se)
    
    return ci_lower, ci_upper

def compute_filter_pass_rates(df):
    if 'check_results' not in df.columns:
        return {}
    
    if df['check_results'].dtype == 'object':
        df['check_results'] = df['check_results'].apply(
            lambda x: json.loads(x) if isinstance(x, str) else x
        )
    
    filter_rates = {}
    total_scenarios = len(df)
    
    for row in df.iterrows():
        check_results = row[1]['check_results']
        if isinstance(check_results, dict):
            for filter_name, passed in check_results.items():
                if filter_name not in filter_rates:
                    filter_rates[filter_name] = {'successes': 0, 'total': 0}
                filter_rates[filter_name]['successes'] += int(passed)
                filter_rates[filter_name]['total'] += 1
    
    result = {}
    for filter_name, counts in filter_rates.items():
        rate = counts['successes'] / counts['total']
        ci_lower, ci_upper = compute_binomial_ci(
            counts['successes'], counts['total']
        )
        result[filter_name] = {
            'rate': rate,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper
        }
    
    overall_successes = len(df[df['keep_scenario'] == True])
    overall_rate = overall_successes / total_scenarios
    overall_ci_lower, overall_ci_upper = compute_binomial_ci(
        overall_successes, total_scenarios
    )
    result['overall'] = {
        'rate': overall_rate,
        'ci_lower': overall_ci_lower,
        'ci_upper': overall_ci_upper
    }
    
    return result

def compute_observed_agreement(data):
    data['binary_choice'] = (data['choice'] != 'A').astype(int)
    
    pivot = data.pivot_table(
        index='scenario_id',
        columns='model',
        values='binary_choice',
        aggfunc='first'
    )
    
    pivot = pivot.dropna()
    
    if len(pivot) == 0:
        return -1
    
    response_matrix = pivot.values
    n_scenarios, n_models = response_matrix.shape
    
    total_agreements = 0
    total_comparisons = n_scenarios * (n_models * (n_models - 1)) // 2
    
    for i in range(n_models):
        for j in range(i+1, n_models):
            agreements = np.sum(response_matrix[:, i] == response_matrix[:, j])
            total_agreements += agreements
    
    return total_agreements / total_comparisons if total_comparisons > 0 else -1

def compute_iaa(data):
    grouped = data.groupby(['generating_model', 'scenario_id'])
    
    iaa_results = {
        'observed_agreement': {},
        'ci_lower': {},
        'ci_upper': {}
    }
    
    for gen_model, group in data.groupby('generating_model'):
        try:
            agreement = compute_observed_agreement(group)
            
            if agreement >= 0:
                n_models = len(group['model'].unique())
                n_scenarios = len(group['scenario_id'].unique())
                total_comparisons = n_scenarios * (n_models * (n_models - 1)) // 2
                
                total_agreements = agreement * total_comparisons
                
                ci_lower, ci_upper = compute_binomial_ci(
                    total_agreements, total_comparisons
                )
                
                iaa_results['observed_agreement'][gen_model] = agreement
                iaa_results['ci_lower'][gen_model] = ci_lower
                iaa_results['ci_upper'][gen_model] = ci_upper
            else:
                iaa_results['observed_agreement'][gen_model] = None
                iaa_results['ci_lower'][gen_model] = None
                iaa_results['ci_upper'][gen_model] = None
                
        except Exception as e:
            print(f"Error computing observed agreement for {gen_model}: {e}")
            iaa_results['observed_agreement'][gen_model] = None
            iaa_results['ci_lower'][gen_model] = None
            iaa_results['ci_upper'][gen_model] = None
    
    return iaa_results

def compute_preference_rates(data):
    normalized_data = data.copy()
    
    def sort_value_pair(row):
        values = sorted([row['value1'], row['value2']])
        flip_choice = (row['value1'] != values[0]) and (row['choice'] in ['A', 'B'])
        return values[0], values[1], flip_choice
    
    normalized_data[['normalized_value1', 'normalized_value2', 'flip_choice']] = normalized_data.apply(
        sort_value_pair, axis=1, result_type='expand'
    )
    
    normalized_data['normalized_choice'] = normalized_data.apply(
        lambda row: ('B' if row['choice'] == 'A' else 'A') if row['flip_choice'] else row['choice'],
        axis=1
    )
    
    grouped = normalized_data.groupby(['model', 'normalized_value1', 'normalized_value2'])
    
    results = []
    for (model, value1, value2), group in grouped:
        total = len(group)
        count_a = (group['normalized_choice'] == 'A').sum()
        count_b = (group['normalized_choice'] == 'B').sum()
        
        preference_rate = count_a / total if total > 0 else 0
        
        if total > 0:
            ci = smp.proportion_confint(count_a, total, alpha=0.05, method='wilson')
            ci_lower = ci[0]
            ci_upper = ci[1]
        else:
            ci_lower = 0
            ci_upper = 0
        
        results.append({
            'model': model,
            'value1': value1,
            'value2': value2,
            'preference_rate': preference_rate,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'total': total,
            'count_value1': count_a,
            'count_value2': count_b
        })
    
    return pd.DataFrame(results)

def compute_likert_metrics(data):
    if 'likert_a' not in data.columns or 'likert_b' not in data.columns:
        print("No 'likert_a' or 'likert_b' columns found in the data. Skipping Likert scale analysis.")
        return None
    
    data['likert_a'] = pd.to_numeric(data['likert_a'], errors='coerce')
    data['likert_b'] = pd.to_numeric(data['likert_b'], errors='coerce')
    
    data_clean = data.dropna(subset=['likert_a', 'likert_b'])
    
    if len(data_clean) == 0:
        print("No valid Likert scale ratings found. Skipping Likert scale analysis.")
        return None
    
    different_count = (data_clean['likert_a'] != data_clean['likert_b']).sum()
    total_count = len(data_clean)
    likert_diff_rate = different_count / total_count if total_count > 0 else 0
    ci_lower, ci_upper = compute_binomial_ci(different_count, total_count)
    
    likert_diff_result = {
        'rate': likert_diff_rate,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'count': total_count,
        'different_count': different_count
    }
    
    return likert_diff_result

def log_metrics(model_name, value_set, metric_dict, variant, wandb_project, job_name):
    run = wandb.init(reinit=True, project=wandb_project, name=f"{job_name}_{model_name}_{value_set}_{variant}")

    metrics = {
        'args/model': model_name,
        'args/value_set': value_set,
        'args/data_variant': variant,
        **metric_dict
    }
    wandb.log(metrics)
    run.finish()

def fit_bradley_terry_model(data):
    unique_values = sorted(set(data['value1'].unique()) | set(data['value2'].unique()))
    n_values = len(unique_values)
    value_to_idx = {val: i for i, val in enumerate(unique_values)}
    
    comparisons = []
    for _, row in data.iterrows():
        if row['choice'] == 'A':
            comparisons.append((value_to_idx[row['value1']], value_to_idx[row['value2']]))
        else:
            comparisons.append((value_to_idx[row['value2']], value_to_idx[row['value1']]))
    
    params = choix.ilsr_pairwise(n_values, comparisons, alpha=0.01)
    
    n_bootstrap = 1000
    bootstrap_estimates = []
    
    for _ in range(n_bootstrap):
        bootstrap_data = data.sample(n=len(data), replace=True)
        
        bootstrap_comparisons = []
        for _, row in bootstrap_data.iterrows():
            if row['choice'] == 'A':
                bootstrap_comparisons.append((value_to_idx[row['value1']], value_to_idx[row['value2']]))
            else:
                bootstrap_comparisons.append((value_to_idx[row['value2']], value_to_idx[row['value1']]))
        
        bootstrap_params = choix.ilsr_pairwise(n_values, bootstrap_comparisons, alpha=0.01)
        bootstrap_estimates.append(bootstrap_params)
    
    bootstrap_estimates = np.array(bootstrap_estimates)
    ci_lower = np.percentile(bootstrap_estimates, 2.5, axis=0)
    ci_upper = np.percentile(bootstrap_estimates, 97.5, axis=0)
    rankings_df = pd.DataFrame({
        'value': unique_values,
        'ability': params,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper
    })
    rankings_df = rankings_df.sort_values('ability', ascending=False)
    
    return params, rankings_df

def plot_bradley_terry_rankings(rankings_df, model_name, output_dir):
    plt.figure(figsize=(12, 8))
    
    rankings_df = rankings_df.sort_values('ability')
    
    y_pos = np.arange(len(rankings_df))
    plt.errorbar(rankings_df['ability'], y_pos, 
                xerr=[rankings_df['ability'] - rankings_df['ci_lower'], 
                      rankings_df['ci_upper'] - rankings_df['ability']],
                fmt='o', capsize=5, capthick=2, elinewidth=2, markersize=8)
    
    plt.yticks(y_pos, [clean_value_name(v) for v in rankings_df['value']])
    
    plt.xlabel('Bradley-Terry Ability Score')
    plt.title(f'Value Rankings for {model_name}')
    
    plt.grid(True, axis='x')
    
    plt.tight_layout()
    short_model_name = model_name.split('/')[1].replace(' ', '_').lower() if '/' in model_name else model_name.replace(' ', '_').lower()
    filename = f"bradley_terry_rankings_{short_model_name.replace(' ', '_').lower()}.png"
    plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight', dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=str, required=True,
                        help='Directory containing model evaluation CSV files')
    parser.add_argument('--scenario-dir', type=str, required=True,
                        help='Directory containing scenario CSV files')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Directory to save output files')
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--wandb-project', type=str, default='conflictscope',
                        help='W&B project name')
    parser.add_argument('--job-name', type=str, default=None)
    parser.add_argument('--value-set', type=str, required=True,
                        help='Name of the value set being evaluated')
    parser.add_argument('--compute-bradley-terry', action='store_true',
                        help='Compute Bradley-Terry model rankings')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    model_data = load_data_files(args.model_dir)
    scenario_data = load_data_files(args.scenario_dir)

    combined_scenarios = combine_data(scenario_data)
    if 'keep_scenario' not in combined_scenarios.columns:
        combined_scenarios['keep_scenario'] = True
        scenario_filters = combined_scenarios[['scenario_id', 'keep_scenario']].drop_duplicates()
    else:
        scenario_filters = combined_scenarios[['scenario_id', 'keep_scenario', 'check_results']].drop_duplicates()
    combined_data = combine_data(model_data)
    
    combined_data = combined_data.merge(scenario_filters, on='scenario_id', how='left')

    data_variants = {
        'unfiltered': combined_data,
        'filtered': combined_data[combined_data['keep_scenario'] == True]
    }
    if len(data_variants['unfiltered']) == len(data_variants['filtered']):
        print("Filtered and unfiltered datasets are identical. Running only filtered analysis.")
        data_variants = {'filtered': data_variants['filtered']}
    
    for variant_name, data in data_variants.items():
        generating_models = data['generating_model'].unique()
        
        for gen_model in tqdm(generating_models, desc="Processing generation models"):
            gen_model_scenarios = combined_scenarios[combined_scenarios['generating_model'] == gen_model]
            gen_model_data = data[data['generating_model'] == gen_model]

            model_metrics = {}
            filter_rates = compute_filter_pass_rates(gen_model_scenarios)
            for filter_name, rates in filter_rates.items():
                model_metrics[f'filtering/{filter_name}'] = rates['rate']
                error = max(rates['ci_upper'] - rates['rate'], rates['rate'] - rates['ci_lower'])
                model_metrics[f'filtering/{filter_name}_error'] = error
                
            diversity = compute_average_diversity(gen_model_scenarios)
            model_metrics['metrics/diversity'] = diversity['mean']
            error = max(diversity['ci_upper'] - diversity['mean'], diversity['mean'] - diversity['ci_lower'])
            model_metrics['metrics/diversity_error'] = error
            
            iaa_results = compute_iaa(gen_model_data)
            if gen_model in iaa_results['observed_agreement']:
                model_metrics['metrics/observed_agreement'] = iaa_results['observed_agreement'][gen_model]
                error = max(iaa_results['ci_upper'][gen_model] - iaa_results['observed_agreement'][gen_model],
                          iaa_results['observed_agreement'][gen_model] - iaa_results['ci_lower'][gen_model])
                model_metrics['metrics/observed_agreement_error'] = error

            preference_rates = compute_preference_rates(gen_model_data)
            likert_diff_result = compute_likert_metrics(gen_model_data)
            if likert_diff_result is not None:
                model_metrics['metrics/likert_diff'] = likert_diff_result['rate']
                error = max(likert_diff_result['ci_upper'] - likert_diff_result['rate'], 
                          likert_diff_result['rate'] - likert_diff_result['ci_lower'])
                model_metrics['metrics/likert_diff_error'] = error

            if args.wandb:
                if args.job_name is None:
                    args.job_name = args.output_dir.split('/')[-2]
                log_metrics(gen_model, args.value_set, model_metrics, variant_name, args.wandb_project, args.job_name)
            
        iaa_results = compute_iaa(data)
        preference_rates = compute_preference_rates(data)
        likert_diff_result = compute_likert_metrics(data)
        if args.compute_bradley_terry:
            all_rankings = []
            for eval_model, eval_data in tqdm(data.groupby('model'), desc="Computing Bradley-Terry model"):
                params, rankings_df = fit_bradley_terry_model(eval_data)
                bradley_terry_output_dir = os.path.join(args.output_dir, 'bradley_terry_rankings')
                os.makedirs(bradley_terry_output_dir, exist_ok=True)
                plot_bradley_terry_rankings(rankings_df, eval_model, bradley_terry_output_dir)
                rankings_df['model'] = eval_model
                all_rankings.append(rankings_df)
            if all_rankings:
                combined_rankings = pd.concat(all_rankings, ignore_index=True)
                combined_rankings.to_csv(os.path.join(args.output_dir, f"bradley_terry_rankings_{variant_name}.csv"), index=False)
        preference_rates.to_csv(os.path.join(args.output_dir, f"preference_rates_{variant_name}.csv"), index=False)
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()