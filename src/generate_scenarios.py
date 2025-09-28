from model_wrappers import ModelWrapper
from utils import SYSTEM_PROMPT, CAUSAL_GRAPH_PROMPTS, STAGE_TWO_PROMPT, STAGE_ONE_BASE_PROMPT, STAGE_ONE_OUTPUT_FORMAT, STAGE_TWO_BASE_PROMPT, parse_json, load_value_dict
from argparse import ArgumentParser
from tqdm import tqdm
import os
import json
import random
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def parse_args():
    parser = ArgumentParser()

    parser.add_argument('--value1', '-v1', type=str, required=True)
    parser.add_argument('--value2', '-v2', type=str, required=True)
    parser.add_argument('--value-set', '-v', type=str, required=True)
    parser.add_argument('--num-scenarios', '-n', type=int, default=40)
    parser.add_argument('--batch-size', '-b', type=int, default=40)
    parser.add_argument('--max-extra-tries', '-e', type=int, default=5)
    parser.add_argument('--deduplicate', '-d', action='store_true')
    parser.add_argument('--deduplication_threshold', '-dt', type=float, default=0.9)
    parser.add_argument('--model', '-m', type=str, default='gpt-4o-mini')
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--max-tokens', type=int, default=4096)
    parser.add_argument('--output-dir', '-o', type=str, required=True)
    parser.add_argument('--add-to-existing', '-a', action='store_true')

    return parser.parse_args()

def standardize_dict_keys(d: dict) -> dict:
    return {k.lower().replace(' ', '_') : v for k, v in d.items()}

def generate_stage_one(v1, v2, client, num_scenarios, start_idx, value_dict):
    scenarios = {}
    prompt_types = list(CAUSAL_GRAPH_PROMPTS.keys())
    prompt_types_count = {prompt_type: int(num_scenarios / len(prompt_types)) for prompt_type in prompt_types}
    for i in range(num_scenarios - sum(prompt_types_count.values())):
        prompt_types_count[random.choice(prompt_types)] += 1
       
    for prompt_type in prompt_types:
        if prompt_types_count[prompt_type] == 0:
            continue
        scenario_length = len(scenarios)
        
        base_prompt = STAGE_ONE_BASE_PROMPT.format(
            v1=v1, 
            v2=v2, 
            value1_definition=value_dict[v1], 
            value2_definition=value_dict[v2]
        )
        base_prompt += "\n" + CAUSAL_GRAPH_PROMPTS[prompt_type].format(v1, v2)
        base_prompt += "\n" + STAGE_ONE_OUTPUT_FORMAT.format(
            num_scenarios=prompt_types_count[prompt_type],
            v1=v1,
            v2=v2,
            prompt_type=prompt_type
        )
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": base_prompt}]
        outputs = parse_json(client.generate(messages))

        for idx, output in enumerate(outputs):
            if idx >= prompt_types_count[prompt_type]:
                break
            try:
                outputs[output]['value1'] = value_dict[v1]
                outputs[output]['value2'] = value_dict[v2]
                outputs[output]['prompt_type'] = prompt_type
                scenarios[f'{v1}-{v2}-{prompt_type}-{start_idx + scenario_length + idx}'] = standardize_dict_keys(outputs[output])
            except TypeError:
                pass
    
    return scenarios

def generate_stage_two(scenarios, client, value_dict):
    scenario_list = []
    for scenario_id, scenario in scenarios.items():
        if 'description' in scenario:
            continue
        else:
            keys = [k for k in list(scenario.keys()) if k not in ['value1', 'value2', 'prompt_type']]
            template = '\n'.join(f'{k}: {scenario[k]}' for k in keys)
            reverse_dict = {v: k for k, v in value_dict.items()}

            base_prompt = STAGE_TWO_BASE_PROMPT.format(
                v1=reverse_dict[scenario['value1']],
                v2=reverse_dict[scenario['value2']],
                value1_definition=scenario['value1'],
                value2_definition=scenario['value2']
            )
            base_prompt += "\n" + STAGE_TWO_PROMPT.format(
                template = template,
                value1 = scenario['value1'],
                value2 = scenario['value2']
            )

            messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": base_prompt}]
            scenario_list.append((scenario_id, messages))
            
    outputs = [parse_json(output) for output in client.batch_generate([scenario[1] for scenario in scenario_list], verbose=True)]
    for (scenario_id, _), output in zip(scenario_list, outputs):
        try:
            keys = ['description', 'user_prompt', 'action1', 'action2', 'consequence1', 'consequence2']
            for key in keys:
                scenarios[scenario_id][key] = output[key]

        except KeyError:
            print(f'Removing the following scenario due to instantiation error: {scenarios[scenario_id]}')
            del scenarios[scenario_id]

    return scenarios

def deduplicate_scenarios(embeddings, generated_scenarios, embedding_model, threshold):
    new_embeddings = embedding_model.encode([scenario.get('context', '') + ' ' + scenario.get('action_opportunity', '') for scenario in generated_scenarios.values()])
    all_embeddings = np.vstack((embeddings, new_embeddings))
    all_similarities = cosine_similarity(all_embeddings)

    to_remove = set()
    for i in range(len(embeddings), len(all_embeddings)):
        try:
            if max(all_similarities[i][:i]) > threshold:
                to_remove.add(i)
        except ValueError:
            continue

    saved_ids = [k for idx, k in enumerate(generated_scenarios.keys()) if idx + len(embeddings) not in to_remove]
    saved_scenarios = {k:v for k,v in generated_scenarios.items() if k in saved_ids}
    saved_indices = list(range(len(embeddings))) + [i for i in range(len(embeddings), len(all_embeddings)) if i not in to_remove]
    return saved_scenarios, all_embeddings[saved_indices]
    
def main():
    args = parse_args()
    client = ModelWrapper.create(
        args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens
    )
    value_dict = load_value_dict(args.value_set)

    if args.deduplicate:
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = np.empty((0, 384))


    os.makedirs(args.output_dir, exist_ok=True)
    short_model_name = args.model.split('/')[-1]
    output_path = os.path.join(args.output_dir, f'{short_model_name}_{args.value1}_{args.value2}.json')

    if not os.path.exists(output_path):
        scenarios = {}
    else:
        if args.add_to_existing:
            with open(output_path, 'r') as f:
                scenarios = json.load(f)
        else:
            print(f'Output file {output_path} already exists. Exiting.')
            return
            
    target_length = args.num_scenarios + len(scenarios)
    tries = 0

    progress_bar = tqdm(total=target_length, desc="Generating scenarios: stage one")
    progress_bar.update(len(scenarios))

    while len(scenarios) < target_length:
        initial_count = len(scenarios)
        effective_batch_size = min(args.batch_size, target_length - len(scenarios))
        generated = generate_stage_one(
            args.value1, args.value2, client, effective_batch_size, 
            len(scenarios), value_dict
        )
        if args.deduplicate and len(generated) > 0:
            generated, embeddings = deduplicate_scenarios(embeddings, generated, embedding_model, args.deduplication_threshold)
    
        scenarios = scenarios | generated
        new_scenarios = len(scenarios) - initial_count
        progress_bar.update(new_scenarios)
        
        if new_scenarios == 0:
            tries += 1
        if tries >= args.max_extra_tries:
            break

    progress_bar.close()

    scenarios = generate_stage_two(scenarios, client, value_dict)
    with open(output_path, 'w') as f:
        json.dump(scenarios, f, indent=4)

if __name__ == '__main__':
    main()