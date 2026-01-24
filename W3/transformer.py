import os
import torch
from transformer import AutoTokenizer

# This will be implmented after
from helper import WeightManager, apply_rope, extract_model_weights

weight_path = os.environ.get("TRANSFROMER_WEIGHT_PATH", "/data/Meta-Llama-3-8B-instruct")

layers = 32
head_dim = 128 # 
num_qo_heads = 32
num_kv_heads = 4  # 


weight_manager = WeightManager()
weight_manager.loader_from_safe_tensor(weight_manager.weights_map, layers)

weights = None # TODO

# Load weights

embedding = weights["embedding"]

# ...
layernormAttn_weight = None
self_attn_q_proj_weight = None

# FFN

# Final layer normalization

# Fianly vocab projection


tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

epsilon = 1e-5

def run_one_iteration(input_ids: list[int]) -> int:
    # Multi-head causal attention
    input_tensor = torch.tensor(input_ids, dtype=torch.int32, device='cuda')

    hiddent_state = embedding[input_tensor]

    for current_layer in range(layers):
        # Layer normalization(RMSNorm) for each vector of user requests
        # I: (seq_len, hidden_dim): (seq_len, 4096)
        # O: (seq_len, hidden_dim): (seq_len, 4096)
        # dim -1: walk over columns in particular row -> row-wise
        rms = torch.sqrt(torch.mean(hiddent_state**2, dim=-1, keepdim=True) + epsilon)
        normalized_x = hiddent_state / rms

        # Element-size multiplication with layernorm weights, hidden_dim 4096
        # Broadcast Semantics: https://docs.pytorch.org/docs/stable/notes/broadcasting.html
        # I_1 / x_1: (seq_len, hidden_dim): (seq_len, 4096) 
        # I_2 / w: (hidden_dim) - (automaticaaly broadcasted) -> (seq_len, hidden_dim)
        x = normalized_x * layernormAttn_weight[current_layer]


        # Multi-head causal attention
        # I_1 / x: (seq_len, hidden_dim): (seq_len, 4096)
        # I_2 / w: (num_qo_heads * head_dim, hidden_dim): (32 * 128, 4096)
        # O: (seq_len, num_qo_heads * head_dim): (seq_len, 32 * 128)
        # num_qo_heads: quary heads
        q = x.matmul(self_attn_q_proj_weight[current_layer].t())

        # For k, v, so w is (num
        )
        pass

    pass

def _demo_generation() -> None:

    input_string = "The University of Washington is a public research university in Seattle, Washington."
    input_ids = tokenizer.encode(input_string)

    output_ids = input_ids
    interations = 10

    for round in range(interations):
        new_token = run_one_iteration(output_ids)
        output_ids.append(new_token)

    output_string = tokenizer.decode(output_ids, skip_special_tokens=True)

    print(output_string)
if __name__ == "__main__":
    _demo_generation()