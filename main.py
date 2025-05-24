from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold
from sklearn.multioutput import MultiOutputRegressor
from scipy.stats import spearmanr
import torch
import pickle
from tqdm import tqdm
from einops import rearrange
import matplotlib.pyplot as plt
import seaborn as sns
import json
import random
import warnings
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import transformers
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.neural_network import MLPRegressor
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from baukit import Trace, TraceDict
from custom_llama import llama # modified code to access attention head outputs
import argparse
HUGGINGFACE_TOKEN = "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"

#Extracting Activation
def extract_attention_head_activations(model, statements):
    HEADS = [f"model.layers.{i}.self_attn.head_out" for i in range(model.config.num_hidden_layers)]
    head_wise_hidden_states_list = []
    for prompt in tqdm(statements, total=len(statements)):
        with torch.no_grad():
            with TraceDict(model, HEADS) as ret:
                output = model(prompt.to('cuda'), output_hidden_states=True, output_attentions=True)
                head_wise_hidden_states = [ret[head].output.squeeze().detach().cpu() for head in HEADS]
                head_wise_hidden_states = torch.stack(head_wise_hidden_states, dim = 0).squeeze().numpy()
                head_wise_hidden_states_list.append(head_wise_hidden_states[:, :, :])
    features = []
    for head_wise_hidden_states, label in zip(head_wise_hidden_states_list, labels):
        features.append(rearrange([np.array(head_wise_hidden_states[:,-1,:])], 'b l (h d) -> b l h d', h = model.config.num_attention_heads))
    features = np.stack(features, axis=0)
    return features

# Probing
def probe_head(features, labels, model_config, probe_type='linear'):
    performance = np.zeros((model_config.num_hidden_layers, model_config.num_attention_heads))
    model_dict = {}

    for i in tqdm(range(model_config.num_hidden_layers)):
        model_dict[i] = {}
        for j in range(model_config.num_attention_heads):
            kf = KFold(n_splits=2, shuffle=True, random_state=42)
            scores = []

            for train_indices, test_indices in kf.split(range(features.shape[0])):
                X_train = features[train_indices, 0, i, j, :]
                X_test = features[test_indices, 0, i, j, :]
                y_train = np.array(labels)[train_indices]
                y_test = np.array(labels)[test_indices]

                if probe_type == 'linear':
                    model = Ridge(alpha=1, fit_intercept=False)
                elif probe_type == 'mlp':
                    model = MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42)
                else:
                    raise ValueError("Invalid probe_type. Choose 'linear' or 'mlp'.")

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                model_dict[i][j] = model
                scores.append(spearmanr(y_test, y_pred).statistic)

            performance[i, j] = np.mean(scores)

    return performance, model_dict



def lt_modulated_vector_add(head_output, layer_name):
    layer_index = layer_name[len('model.layers.'):]
    layer_index = int(layer_index[:layer_index.index('.')])
    head_output = rearrange(head_output.detach().cpu(), 'b s (h d) -> b s h d', h=model.config.num_attention_heads)
    for head_index in head_dict[layer_index]:
        head_output[:, -1, head_index, :] += alpha  * focal_ridge_dict[(layer_index, head_index)] * np.std(features[:, 0, layer_index, head_index, :], axis=0)
    head_output = rearrange(head_output, 'b s h d -> b s (h d)')
    return head_output.to('cuda')



# class Args:
#     pass

# args = Args()
# args.input_file = '/content/RepresentationPoliticalLLM/BWS_annotations_modified.csv'
# args.model = 'lmsys/vicuna-7b-v1.5'
# args.prob = 'linear'  # or 'mlp'
# args.axis = 'competence'  # or 'warmth'
# args.prompt = 'Generate a text like this'

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, help='input file')
    parser.add_argument('--model', type=str, help='NousResearch/Llama-2-7b-chat-hf or lmsys/vicuna-7b-v1.5')
    parser.add_argument('--prob', type=str, help='mlp or linear')
    parser.add_argument('--axis', type=str, help='competence or warmth')
    parser.add_argument('--prompt', type=str, help = 'Evaluate the competence of the word')
    args = parser.parse_args()

    df = pd.read_csv(args.input_file)


    #Prompting
    for model_name in [args.model]:
        tokenizer = llama.LlamaTokenizer.from_pretrained(model_name, cache_dir='./model', token=HUGGINGFACE_TOKEN)
        # Generate prompts for competence/warmth simulation
        statements = []
        for index, row in df.iterrows():
            statements.append(tokenizer(f'USER: {args.prompt} "{row["Sentence"]}".\nASSISTANT:', return_tensors="pt")['input_ids'])
        pickle.dump(statements, open(f'./results_replication/{model_name.replace("/", "_")}_{args.axis}.pkl', 'wb'))

    # Extract attention output
    for model_name in [args.model]: #
        model = llama.LlamaForCausalLM.from_pretrained(model_name, cache_dir='./model', low_cpu_mem_usage=True, torch_dtype=torch.float16, token=HUGGINGFACE_TOKEN).to('cuda:0')
        statements = pickle.load(open(f'./results_replication/{model_name.replace("/", "_")}_{args.axis}.pkl', 'rb'))
        labels = np.array(df[args.axis].astype(float))
        features = extract_attention_head_activations(model, statements)
        pickle.dump((features, labels), open(f"./results_replication/{model_name.replace('/','_')}_{args.axis}_features.pkl", 'wb'))

    # Probing
    for model_name in [args.model]:
        features, labels = pickle.load(open(f"./results_replication/{model_name.replace('/','_')}_{args.axis}_features.pkl", 'rb'))
        probe_type = args.prob
        performance, model_dict = probe_head(features, labels, model.config, probe_type=probe_type)
        suffix = f"{'_mlp' if probe_type == 'mlp' else ''}"
        pickle.dump(performance, open(f"./results_replication/{model_name.replace('/','_')}_{args.axis}_performance{suffix}.pkl", 'wb'))
        pickle.dump(model_dict, open(f"./results_replication/{model_name.replace('/','_')}_{args.axis}_ridge{suffix}.pkl", 'wb'))


    #Intervention
    for model_name in [args.model]:
        tokenizer = llama.LlamaTokenizer.from_pretrained(model_name, cache_dir='./model', token=HUGGINGFACE_TOKEN)
        model = llama.LlamaForCausalLM.from_pretrained(model_name, cache_dir='./model', low_cpu_mem_usage=True, torch_dtype=torch.float16, 
                                                      token=HUGGINGFACE_TOKEN).to('cuda:0')
        performance = pickle.load(open(f"./results_replication/{model_name.replace('/','_')}_{args.axis}_performance.pkl", 'rb'))
        features, labels = pickle.load(open(f"./results_replication/{model_name.replace('/','_')}_{args.axis}_features.pkl", 'rb'))
        trained_ridge_dict = pickle.load(open(f"./results_replication/{model_name.replace('/','_')}_{args.axis}_ridge{suffix}.pkl", 'rb'))
        topics = ['These people are active', 'these people are lazy', 'these people are determined', 'these people are warm',  'these people are cold']
        results = []
        for k in reversed([16, 32, 48, 64, 80, 96]):
            for alpha in tqdm(reversed([-30, -20, -10, 0, 10, 20, 30])): # Add -50, -40, 40, 50 for the coherence tests
                for topic in topics:
                    original_prompt = f"Write a statement like {topic}."
                    top_indices = np.dstack(np.unravel_index(np.argsort(performance.ravel()), (32, 32)))[0][-k:, :][::-1]
                    focal_ridge_dict = {}
                    for i in top_indices:
                        ridge_model = trained_ridge_dict[i[0]][i[1]]
                        focal_ridge_dict[tuple(i)] = ridge_model.coef_
                    head_dict = {}
                    for i in top_indices:
                        if i[0] not in head_dict:
                            head_dict[i[0]] = [i[1]]
                        else:
                            head_dict[i[0]].append(i[1])
                    with TraceDict(model, [f'model.layers.{i}.self_attn.head_out' for i in sorted(list(set(top_indices[:,0])))], edit_output=lt_modulated_vector_add) as ret: 
                        input_ids = tokenizer(f"USER: {original_prompt}\nASSISTANT: ", return_tensors="pt")['input_ids']
                        model_gen_tokens = model.generate(input_ids.to('cuda')[0][:-1].unsqueeze(0), max_length=200)
                    model_gen_str = tokenizer.decode(model_gen_tokens[0], skip_special_tokens=True)
                    model_gen_str = model_gen_str.strip()
                    results.append([k, alpha, topic, model_gen_str])
                    with open(f'intervention_results_{args.axis}.txt', 'a') as f:  # 'a' means append, 'w' means overwrite
                        print(k, alpha, topic, model_gen_str, '\n', file=f)

        pickle.dump(results, open(f"./results_replication/{model_name.replace('/','_')}_intervention_results_{args.axis}.pkl", 'wb'))






    #Plot heatmap

    for model_name in [args.model]:
        performance = pickle.load(open(f"./results_replication/{model_name.replace('/', '_')}_{args.axis}_performance.pkl", 'rb'))

        num_layers = performance.shape[0]
        num_heads = performance.shape[1]

        performance_flipped = np.flipud(performance)

        plt.figure(figsize=(14, 6))
        sns.heatmap(performance_flipped, cmap="viridis",
                    xticklabels=[f"H{h}" for h in range(num_heads)],
                    yticklabels=[f"L{l}" for l in reversed(range(num_layers))],
                    annot=True, fmt=".2f")

        plt.title(f"Spearman Performance for {model_name} ({args.axis} Prediction)")
        plt.xlabel("Attention Head")
        plt.ylabel("Layer")
        plt.tight_layout()
        plt.savefig(f"./results_replication/{model_name.replace('/', '_')}_{args.axis}_performance_heatmap_{args.prob}.png")
        plt.show()
