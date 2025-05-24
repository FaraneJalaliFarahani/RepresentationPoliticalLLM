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
from sklearn.multioutput import MultiOutputRegressor
HUGGINGFACE_TOKEN = "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"


# Load competence dataset
competence_df = pd.read_csv('/content/RepresentationPoliticalLLM/BWS_annotations_modified.csv')
#competence_df = competence_df.dropna(subset=['Sentence', 'competence', 'warmth']).reset_index(drop=True)

# --- Prompting (Skip if you already have tokenized prompts) ---
# Replace with your tokenizer/model if necessary




HUGGINGFACE_TOKEN = "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"

for model_name in ['lmsys/vicuna-7b-v1.5']:
    tokenizer = llama.LlamaTokenizer.from_pretrained(model_name, cache_dir='./model', token=HUGGINGFACE_TOKEN)

    # Generate prompts for competence simulation
    statements_competence = []
    for index, row in competence_df.iterrows():
        statements_competence.append(tokenizer(f'USER:Generate a text like this statement "{row["Sentence"]}".\nASSISTANT:', return_tensors="pt")['input_ids'])
    pickle.dump(statements_competence, open(f'./results_replication/{model_name.replace("/", "_")}_competence.pkl', 'wb'))

# --- Extracting Activations ---
def predictions_to_dataframe(predictions_dict, n_targets=2):
    rows = []
    for layer, heads in predictions_dict.items():
        for head, folds in heads.items():
            for fold_info in folds:
                fold = fold_info['fold']
                test_indices = fold_info['test_indices']
                texts = fold_info['input_texts']
                y_true = fold_info['y_test']
                y_pred = fold_info['y_pred']
                for idx_in_fold, global_idx in enumerate(test_indices):
                    row = {
                        'layer': layer,
                        'head': head,
                        'fold': fold,
                        'sample_index': global_idx,
                        'input_text': texts[idx_in_fold]
                    }
                    # Add true and predicted labels for each target (e.g., competence, warmth)
                    for target in range(n_targets):
                        row[f'label_{target}'] = y_true[idx_in_fold][target]
                        row[f'prediction_{target}'] = y_pred[idx_in_fold][target]
                    rows.append(row)
    return pd.DataFrame(rows)

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
    for head_wise_hidden_states in head_wise_hidden_states_list:
        features.append(rearrange([np.array(head_wise_hidden_states[:,-1,:])], 'b l (h d) -> b l h d', h = model.config.num_attention_heads))
    features = np.stack(features, axis=0)
    return features

for model_name in ['lmsys/vicuna-7b-v1.5']:
    model = llama.LlamaForCausalLM.from_pretrained(
        model_name, cache_dir='./model', low_cpu_mem_usage=True,
        torch_dtype=torch.float16, token=HUGGINGFACE_TOKEN
    ).to('cuda:0')

    statements_competence = pickle.load(open(f'./results_replication/{model_name.replace("/", "_")}_competence.pkl', 'rb'))

    # Prepare multi-labels (competence, warmth)
    labels = np.vstack([
        competence_df['competence'].astype(float).to_numpy(),
        competence_df['warmth'].astype(float).to_numpy()
    ]).T  # shape: (n_samples, 2)

    features = extract_attention_head_activations(model, statements_competence)
    pickle.dump((features, labels), open(f"./results_replication/{model_name.replace('/','_')}_competence_features.pkl", 'wb'))

# --- Probing ---

def probe_head(features, labels, model_config, texts, probe_type='linear'):
    """
    Args:
        features: (n_samples, 1, n_layers, n_heads, feature_dim)
        labels: (n_samples, n_targets)
        model_config: huggingface model config
        texts: list/array of original sentences, shape (n_samples,)
        probe_type: 'linear' or 'mlp'
    """
    n_targets = labels.shape[1]  # 2: competence, warmth
    performance = np.zeros((model_config.num_hidden_layers, model_config.num_attention_heads, n_targets))
    model_dict = {}
    predictions_dict = {}  # NEW: Collect predictions

    for i in tqdm(range(model_config.num_hidden_layers)):
        model_dict[i] = {}
        predictions_dict[i] = {}
        for j in range(model_config.num_attention_heads):
            kf = KFold(n_splits=2, shuffle=True, random_state=42)
            scores = []
            fold_predictions = []

            for fold_idx, (train_indices, test_indices) in enumerate(kf.split(range(features.shape[0]))):
                X_train = features[train_indices, 0, i, j, :]
                X_test = features[test_indices, 0, i, j, :]
                y_train = labels[train_indices]
                y_test = labels[test_indices]
                texts_test = [texts[idx] for idx in test_indices]   # <--- Grab original texts

                if probe_type == 'linear':
                    base_model = Ridge(alpha=1, fit_intercept=False)
                elif probe_type == 'mlp':
                    base_model = MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42)
                else:
                    raise ValueError("Invalid probe_type. Choose 'linear' or 'mlp'.")

                model = MultiOutputRegressor(base_model)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                model_dict[i][j] = model

                # Spearman for each target
                fold_scores = [spearmanr(y_test[:, k], y_pred[:, k]).statistic for k in range(n_targets)]
                scores.append(fold_scores)

                # --- Save predictions, input texts, labels, and predictions for this fold
                fold_predictions.append({
                    'fold': fold_idx,
                    'test_indices': test_indices.tolist(),
                    'input_texts': texts_test,    # <-- Add the actual input texts here!
                    'X_test': X_test,
                    'y_test': y_test,
                    'y_pred': y_pred,
                })

            # Average across folds
            scores = np.array(scores)
            performance[i, j, :] = scores.mean(axis=0)

            predictions_dict[i][j] = fold_predictions  # Store per head

    return performance, model_dict, predictions_dict

def lt_modulated_vector_add(head_output, layer_name):
            layer_index = int(layer_name.split('.')[2])
            head_output = rearrange(head_output.detach().cpu(), 'b s (h d) -> b s h d', h=model.config.num_attention_heads)
            for head_index in head_dict[layer_index]:
                std_comp = np.std(features[:, 0, layer_index, head_index, :], axis=0)
                std_warmth = std_comp  # or use a different std if you want
                # Apply both interventions
                head_output[:, -1, head_index, :] += (
                    alpha_comp * focal_ridge_dict_comp[(layer_index, head_index)] * std_comp +
                    alpha_warmth * focal_ridge_dict_warmth[(layer_index, head_index)] * std_warmth
                )
            head_output = rearrange(head_output, 'b s h d -> b s (h d)')
            return head_output.to('cuda')


probe_type = 'linear'  # or 'mlp'
suffix = f"{'_mlp' if probe_type == 'mlp' else ''}"
texts = competence_df['Sentence'].tolist()  # Extract original texts

for model_name in ['lmsys/vicuna-7b-v1.5']:
    # Load your extracted features and labels
    features, labels = pickle.load(open(f"./results_replication/{model_name.replace('/','_')}_competence_features{suffix}.pkl", 'rb'))

    # Call probe_head and pass texts
    performance, model_dict, predictions_dict = probe_head(features, labels, model.config, texts, probe_type=probe_type)

    # Save everything with suffix
    pickle.dump(performance, open(f"./results_replication/{model_name.replace('/','_')}_competence_performance{suffix}.pkl", 'wb'))
    pickle.dump(model_dict, open(f"./results_replication/{model_name.replace('/','_')}_competence_models{suffix}.pkl", 'wb'))
    pickle.dump(predictions_dict, open(f"./results_replication/{model_name.replace('/','_')}_competence_predictions{suffix}.pkl", 'wb'))
    # Flatten predictions_dict and save to CSV
    df_predictions = predictions_to_dataframe(predictions_dict, n_targets=labels.shape[1])
    csv_path = f"./results_replication/{model_name.replace('/','_')}_competence_predictions{suffix}.csv"
    df_predictions.to_csv(csv_path, index=False)
    print(f"Saved predictions to {csv_path}")






for model_name in ['lmsys/vicuna-7b-v1.5']:
    tokenizer = llama.LlamaTokenizer.from_pretrained(model_name, cache_dir='./model', token=HUGGINGFACE_TOKEN)
    model = llama.LlamaForCausalLM.from_pretrained(
        model_name, cache_dir='./model', low_cpu_mem_usage=True,
        torch_dtype=torch.float16, token=HUGGINGFACE_TOKEN
    ).to('cuda:0')

    # Load performance and model_dict from probing step
    performance = pickle.load(open(f"./results_replication/{model_name.replace('/','_')}_competence_performance{suffix}.pkl", 'rb'))
    features, labels = pickle.load(open(f"./results_replication/{model_name.replace('/','_')}_competence_features{suffix}.pkl", 'rb'))
    model_dict = pickle.load(open(f"./results_replication/{model_name.replace('/','_')}_competence_models{suffix}.pkl", 'rb'))
    topics = ['these people are always active', 'these people are always lazy', 'these people are always warm', 'these people are always cold', 'these people are always warm and active', 'these people are always cold and inactive' , 'these people are always cold and active' , 'these people are always warm and inactive']
    k_values = [8, 16, 32]
    alphas_comp = [-30, -20, -10, 0, 10, 20, 30]
    alphas_warmth = [-30, -20, -10, 0, 10, 20, 30]  # You can use different values or the same

    results = []
    for k in k_values:
        # Precompute top-k heads for competence and warmth
        perf_flat_comp = performance[:, :, 0].ravel()
        perf_flat_warmth = performance[:, :, 1].ravel()
        # You could also use a different criterion for choosing heads for each trait, or just use competence
        top_indices = np.dstack(np.unravel_index(np.argsort(perf_flat_comp), performance[:, :, 0].shape))[0][-k:, :][::-1]

        # Get the coefficients for the top heads for both competence and warmth
        focal_ridge_dict_comp = {}
        focal_ridge_dict_warmth = {}
        for i in top_indices:
            head_model = model_dict[i[0]][i[1]]  # i[0] is layer, i[1] is head
            # If MultiOutputRegressor, .estimators_[0] is for comp, .estimators_[1] is for warmth
            if hasattr(head_model, 'estimators_'):
                focal_ridge_dict_comp[tuple(i)] = head_model.estimators_[0].coef_
                focal_ridge_dict_warmth[tuple(i)] = head_model.estimators_[1].coef_
            else:
                # Fallback: just use .coef_ (shouldn't happen if using MultiOutputRegressor for both traits)
                focal_ridge_dict_comp[tuple(i)] = head_model.coef_
                focal_ridge_dict_warmth[tuple(i)] = np.zeros_like(head_model.coef_)

        head_dict = {}
        for i in top_indices:
            head_dict.setdefault(i[0], []).append(i[1])

        for alpha_comp in alphas_comp:
            for alpha_warmth in alphas_warmth:
                print(f"\n--- Running k={k}, alpha_comp={alpha_comp}, alpha_warmth={alpha_warmth} ---")
                for topic in topics:
                    original_prompt = f'Paraphrase this text: "{topic}"'
                    with TraceDict(
                        model,
                        [f'model.layers.{i}.self_attn.head_out' for i in sorted(set(top_indices[:, 0]))],
                        edit_output=lt_modulated_vector_add
                    ):
                        input_ids = tokenizer(f"USER: {original_prompt}\nASSISTANT:", return_tensors="pt")['input_ids']
                        gen_tokens = model.generate(input_ids.to('cuda')[0][:-1].unsqueeze(0), max_length=128)
                    gen_text = tokenizer.decode(gen_tokens[0], skip_special_tokens=True).strip()
                    print(f"\n[k={k} | alpha_comp={alpha_comp} | alpha_warmth={alpha_warmth}] Prompt: {topic}\nGenerated: {gen_text}\n")
                    results.append({
                        'k': k,
                        'alpha_comp': alpha_comp,
                        'alpha_warmth': alpha_warmth,
                        'topic': topic,
                        'generated': gen_text
                    })

    pickle.dump(results, open(f"./results_replication/{model_name.replace('/','_')}_10sent_kalpha2_intervention_results.pkl", 'wb'))
    print(f"Saved intervention results for {model_name}")


for model_name in ['lmsys/vicuna-7b-v1.5']:
    performance = pickle.load(open(f"./results_replication/{model_name.replace('/', '_')}_competence_performance{suffix}.pkl", 'rb'))

    # ---- Load predictions_dict and write CSV once per model ----
    predictions_dict = pickle.load(open(f"./results_replication/{model_name.replace('/','_')}_competence_predictions{suffix}.pkl", 'rb'))
    def predictions_to_dataframe(predictions_dict, n_targets=2):
        rows = []
        for layer, heads in predictions_dict.items():
            for head, folds in heads.items():
                for fold_info in folds:
                    fold = fold_info['fold']
                    test_indices = fold_info['test_indices']
                    texts = fold_info['input_texts']
                    y_true = fold_info['y_test']
                    y_pred = fold_info['y_pred']
                    for idx_in_fold, global_idx in enumerate(test_indices):
                        row = {
                            'layer': layer,
                            'head': head,
                            'fold': fold,
                            'sample_index': global_idx,
                            'input_text': texts[idx_in_fold]
                        }
                        for target in range(n_targets):
                            row[f'label_{target}'] = y_true[idx_in_fold][target]
                            row[f'prediction_{target}'] = y_pred[idx_in_fold][target]
                        rows.append(row)
        return pd.DataFrame(rows)

    df_predictions = predictions_to_dataframe(predictions_dict, n_targets=2)  # 2 = competence, warmth
    csv_path = f"./results_replication/{model_name.replace('/','_')}_competence_predictions{suffix}.csv"
    df_predictions.to_csv(csv_path, index=False)
    print(f"Saved predictions to {csv_path}")
    # -----------------------------------------------------------

    num_layers = performance.shape[0]
    num_heads = performance.shape[1]

    for idx, target_name in enumerate(['Competence', 'Warmth']):
        perf_flipped = np.flipud(performance[:, :, idx])

        # Find the best performance and its indices
        max_perf = perf_flipped.max()
        max_idx = np.unravel_index(np.argmax(perf_flipped), perf_flipped.shape)
        # The flipped indices must be mapped back to original layer indices
        best_layer = num_layers - 1 - max_idx[0]
        best_head = max_idx[1]
        print(f"\nBest {target_name} performance: {max_perf:.3f} at Layer {best_layer}, Head {best_head}")

        plt.figure(figsize=(14, 6))
        ax = sns.heatmap(
            perf_flipped,
            cmap="viridis",
            xticklabels=[f"H{h}" for h in range(num_heads)],
            yticklabels=[f"L{l}" for l in reversed(range(num_layers))],
            annot=True, fmt=".2f"
        )
        plt.title(f"Spearman Performance for {model_name} ({target_name} Prediction)")
        plt.xlabel("Attention Head")
        plt.ylabel("Layer")

        # Optional: Highlight the best cell with a red box
        ax.add_patch(plt.Rectangle(
            (best_head, max_idx[0]), 1, 1, fill=False, edgecolor='red', lw=3
        ))

        plt.tight_layout()
        plt.savefig(f"./results_replication/{model_name.replace('/', '_')}_performance_heatmap_{target_name.lower()}.png")
        plt.show()
