import argparse
import torch
import pickle
from tqdm import tqdm
from einops import rearrange
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModelForCausalLM, AutoTokenizer
from baukit import Trace, TraceDict
from custom_llama import llama  # Your custom code for Llama
import warnings
warnings.filterwarnings("ignore")

HUGGINGFACE_TOKEN = "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"

# --- Extract Activations ---
def extract_attention_head_activations(model, statements, labels):
    HEADS = [f"model.layers.{i}.self_attn.head_out" for i in range(model.config.num_hidden_layers)]
    head_wise_hidden_states_list = []
    for prompt in tqdm(statements, total=len(statements)):
        with torch.no_grad():
            with TraceDict(model, HEADS) as ret:
                output = model(prompt.to('cuda'), output_hidden_states=True, output_attentions=True)
                head_wise_hidden_states = [ret[head].output.squeeze().detach().cpu() for head in HEADS]
                head_wise_hidden_states = torch.stack(head_wise_hidden_states, dim=0).squeeze().numpy()
                head_wise_hidden_states_list.append(head_wise_hidden_states[:, :, :])
    features = []
    for head_wise_hidden_states in head_wise_hidden_states_list:
        features.append(rearrange([np.array(head_wise_hidden_states[:,-1,:])], 'b l (h d) -> b l h d', h=model.config.num_attention_heads))
    features = np.stack(features, axis=0)
    return features

# --- Probing (multi-output) ---
def probe_head(features, labels, model_config, probe_type='linear'):
    performance = np.zeros((model_config.num_hidden_layers, model_config.num_attention_heads, labels.shape[1]))
    model_dict = {}
    for i in tqdm(range(model_config.num_hidden_layers)):
        model_dict[i] = {}
        for j in range(model_config.num_attention_heads):
            kf = KFold(n_splits=2, shuffle=True, random_state=42)
            scores = []
            for train_indices, test_indices in kf.split(range(features.shape[0])):
                X_train = features[train_indices, 0, i, j, :]
                X_test = features[test_indices, 0, i, j, :]
                y_train = labels[train_indices]   # (n_train, 2)
                y_test = labels[test_indices]     # (n_test, 2)
                if probe_type == 'linear':
                    model = Ridge(alpha=1, fit_intercept=False)
                elif probe_type == 'mlp':
                    model = MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42)
                else:
                    raise ValueError("Invalid probe_type. Choose 'linear' or 'mlp'.")
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                model_dict[i][j] = model
                # Compute spearman for each output
                scores_dim = []
                for dim in range(labels.shape[1]):
                    # Handle NaN spearman
                    score = spearmanr(y_test[:, dim], y_pred[:, dim]).statistic
                    score = score if not np.isnan(score) else 0
                    scores_dim.append(score)
                scores.append(scores_dim)
            performance[i, j, :] = np.mean(scores, axis=0)  # Mean across folds for both axes
    return performance, model_dict




# --- Main ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, help='input file')
    parser.add_argument('--model', type=str, help='e.g. NousResearch/Llama-2-7b-chat-hf or lmsys/vicuna-7b-v1.5')
    parser.add_argument('--prob', type=str, help='mlp or linear')
    parser.add_argument('--prompt', type=str, help='Base prompt for user')
    args = parser.parse_args()
    probe_type = args.prob
    suffix = f"{'_mlp' if probe_type == 'mlp' else ''}"
    df = pd.read_csv(args.input_file)

    # Prepare multi-dimensional labels
    labels = df[['competence', 'warmth']].astype(float).values  # (N, 2)

    # ---- Prompting ----
    for model_name in [args.model]:
        tokenizer = llama.LlamaTokenizer.from_pretrained(model_name, cache_dir='./model', token=HUGGINGFACE_TOKEN)
        statements = []
        for index, row in df.iterrows():
            statements.append(tokenizer(f'USER: {args.prompt} "{row["Sentence"]}".\nASSISTANT:', return_tensors="pt")['input_ids'])
        pickle.dump(statements, open(f'./results_replication/{model_name.replace("/", "_")}_both_{args.prob}.pkl', 'wb'))

    # ---- Extract activations ----
    for model_name in [args.model]:
        model = llama.LlamaForCausalLM.from_pretrained(model_name, cache_dir='./model', low_cpu_mem_usage=True, torch_dtype=torch.float16, token=HUGGINGFACE_TOKEN).to('cuda:0')
        statements = pickle.load(open(f'./results_replication/{model_name.replace("/", "_")}_both_{args.prob}.pkl', 'rb'))
        features = extract_attention_head_activations(model, statements, labels)
        pickle.dump((features, labels), open(f"./results_replication/{model_name.replace('/','_')}_both_{args.prob}{suffix}_features.pkl", 'wb'))

    # ---- Probing ----
    for model_name in [args.model]:
        features, labels = pickle.load(open(f"./results_replication/{model_name.replace('/','_')}_both_{args.prob}{suffix}_features.pkl", 'rb'))
        performance, model_dict = probe_head(features, labels, model.config, probe_type=probe_type)
        pickle.dump(performance, open(f"./results_replication/{model_name.replace('/','_')}_both_{args.prob}{suffix}_performance.pkl", 'wb'))
        pickle.dump(model_dict, open(f"./results_replication/{model_name.replace('/','_')}_both_{args.prob}{suffix}_ridge.pkl", 'wb'))

    # ---- Plotting: both axes ----
    for model_name in [args.model]:
        performance = pickle.load(open(f"./results_replication/{model_name.replace('/','_')}_both_{args.prob}{suffix}_performance.pkl", 'rb'))
        axes_names = ['Competence', 'Warmth']
        for dim, axis_name in enumerate(axes_names):
            perf_flipped = np.flipud(performance[:, :, dim])
            num_layers = performance.shape[0]
            num_heads = performance.shape[1]
            # Find the best performance and its indices
            max_perf = perf_flipped.max()
            max_idx = np.unravel_index(np.argmax(perf_flipped), perf_flipped.shape)
            best_layer = num_layers - 1 - max_idx[0]
            best_head = max_idx[1]
            print(f"\nBest {axis_name} performance: {max_perf:.3f} at Layer {best_layer}, Head {best_head}")
            plt.figure(figsize=(14, 6))
            ax = sns.heatmap(
                perf_flipped,
                cmap="viridis",
                xticklabels=[f"H{h}" for h in range(num_heads)],
                yticklabels=[f"L{l}" for l in reversed(range(num_layers))],
                annot=True, fmt=".2f"
            )
            plt.title(f"Spearman Performance for {model_name} ({axis_name} Prediction {args.prob} {suffix})")
            plt.xlabel("Attention Head")
            plt.ylabel("Layer")
            ax.add_patch(plt.Rectangle(
                (best_head, max_idx[0]), 1, 1, fill=False, edgecolor='red', lw=3
            ))
            plt.tight_layout()
            plt.savefig(f"./results_replication/{model_name.replace('/', '_')}_{axis_name}_performance_heatmap_{args.prob}{suffix}_{args.prompt}.png")
            plt.show()
