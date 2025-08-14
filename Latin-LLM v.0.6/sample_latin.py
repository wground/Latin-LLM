"""
Sample from a trained Latin GPT model
Uses system detection for optimal hardware configuration.
"""
import os
import json
import pickle
import argparse
import time
from contextlib import nullcontext
import torch
from tokenizers import ByteLevelBPETokenizer

# Import local model
from model import GPTConfig, GPT

def load_system_config(config_path: str = "latin_training_config.json"):
    """Load system configuration from detect_system.py output."""
    if not os.path.exists(config_path):
        print(f"⚠️  Config file {config_path} not found!")
        print("Run 'python3 detect_system.py' first to generate system config.")
        print("Using default CPU configuration...")
        return {
            "recommended_config": {
                "device": "cpu",
                "dtype": "float32",
                "compile": False,
                "backend": "cpu",
                "enable_tf32": False
            }
        }
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print(f"✅ Loaded system config from {config_path}")
    return config

def load_latin_tokenizer(data_dir: str = "gpt_data_latin"):
    """Load the custom Latin tokenizer and return encode/decode functions."""
    meta_path = os.path.join(data_dir, "meta.pkl")
    
    if not os.path.exists(meta_path):
        print(f"❌ Custom tokenizer not found at {meta_path}")
        print("You must run 'python3 prepare_latin.py' first to create custom tokenizer")
        print("Cannot continue without custom Latin tokenizer.")
        exit(1)
    
    # Load tokenizer metadata
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    
    tokenizer_config = meta["tokenizer_config"]
    print(f"✅ Loading custom Latin tokenizer")
    print(f"   Vocabulary size: {meta['vocab_size']}")
    print(f"   Tokenizer type: {tokenizer_config['type']}")
    
    # Load the actual tokenizer
    tokenizer = ByteLevelBPETokenizer(
        tokenizer_config["vocab_file"],
        tokenizer_config["merges_file"]
    )
    
    # Return encode/decode functions
    def encode(text):
        return tokenizer.encode(text).ids
    
    def decode(ids):
        return tokenizer.decode(ids)
    
    return encode, decode

# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Sample from trained Latin GPT model")
    parser.add_argument('--start', type=str, default="caesar ", help='Starting prompt for generation')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of samples to generate')
    parser.add_argument('--max_new_tokens', type=int, default=200, help='Number of tokens per sample')
    parser.add_argument('--temperature', type=float, default=0.7, help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=50, help='Top-k sampling parameter')
    parser.add_argument('--seed', type=int, default=None, help='Random seed (if not set, uses current time)')
    return parser.parse_args()

args = parse_args()

# -----------------------------------------------------------------------------
# Latin-specific sampling configuration
out_dir = 'out-latin'  # directory where Latin model checkpoints are saved
start = args.start  # Latin prompt to start with
num_samples = args.num_samples  # number of samples to generate
max_new_tokens = args.max_new_tokens  # number of tokens to generate per sample
temperature = args.temperature  # sampling temperature (0.6-0.8 good for regularized Latin model)
top_k = args.top_k  # retain only top_k most likely tokens
seed = args.seed if args.seed is not None else int(time.time())  # Use current time for randomness

# Load system-optimized configuration
system_config = load_system_config()
recommended = system_config["recommended_config"]

device = recommended["device"]
dtype = recommended["dtype"] 
compile = recommended["compile"]
enable_tf32 = recommended.get("enable_tf32", False)
# -----------------------------------------------------------------------------

torch.manual_seed(seed)

# Apply hardware optimizations based on detected system
if device == 'cuda':
    torch.cuda.manual_seed(seed)
    if enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device == 'mps':
    # Apple Silicon optimizations
    torch.manual_seed(seed)  # MPS uses unified memory
    # No additional seeds needed for MPS

# Set up mixed precision context based on detected capabilities
if device == 'mps':
    device_type = 'cpu'
elif device == 'cuda':
    device_type = 'cuda'
else:
    device_type = 'cpu'

ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if (device_type == 'cpu' and device == 'cpu') else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# Load the trained Latin model
ckpt_path = os.path.join(out_dir, 'ckpt.pt')
if not os.path.exists(ckpt_path):
    print(f"Error: No checkpoint found at {ckpt_path}")
    print("Make sure you have trained the model first using train_latin.py")
    exit(1)

print(f"Loading model from {ckpt_path}")
checkpoint = torch.load(ckpt_path, map_location=device)
gptconf = GPTConfig(**checkpoint['model_args'])
model = GPT(gptconf)
state_dict = checkpoint['model']

# Handle potential module prefix from compiled models
unwanted_prefix = '_orig_mod.'
for k, v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

model.load_state_dict(state_dict)
print(f"Model loaded successfully. Best validation loss: {checkpoint.get('best_val_loss', 'unknown'):.4f}")

model.eval()
model.to(device)

# Print system configuration summary
print(f"🏛️  Latin GPT Sampling Configuration:")
print(f"   Device: {device} ({dtype})")
print(f"   Compilation: {'enabled' if compile else 'disabled'}")
if device == 'cuda' and enable_tf32:
    print(f"   TF32 optimization: enabled")

if compile:
    if device == 'mps':
        print("⚠️  Model compilation disabled on MPS due to compatibility issues")
        compile = False
    else:
        print("⚡ Compiling model for faster inference...")
        model = torch.compile(model)

# Set up tokenization (using custom Latin tokenizer)
encode, decode = load_latin_tokenizer()

# Handle different start prompt formats
if start.startswith('FILE:'):
    # Load prompt from file
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()
    print(f"Loaded prompt from file: {start[:50]}...")
elif start == "":
    # Empty start - let model generate from scratch
    start = ""
    print("Generating from empty prompt...")
else:
    print(f"Starting with prompt: '{start}'")

# Encode the starting prompt
start_ids = encode(start)
if len(start_ids) == 0:
    # For empty prompts, start with a minimal tensor that won't cause MPS errors
    # Use a single token - for custom tokenizer, use the first token (usually space or similar)
    start_ids = [1]  # Use token ID 1 as fallback for empty prompts
x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

# Generate samples
print(f"\nGenerating {num_samples} samples with {max_new_tokens} tokens each:")
print(f"Temperature: {temperature}, Top-k: {top_k}")
print("=" * 80)

with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            # Use different seed for each sample to ensure variety
            sample_seed = seed + k
            torch.manual_seed(sample_seed)
            if device == 'cuda':
                torch.cuda.manual_seed(sample_seed)
            
            print(f"\n--- Sample {k+1} ---")
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            generated_text = decode(y[0].tolist())
            print(generated_text)
            print('-' * 40)

print(f"\nGenerated {num_samples} samples successfully!")

# Additional Latin-specific prompts you can try:
latin_prompts = [
    "gallia est omnis diuisa in partes tres",
    "arma uirumque cano",
    "ueni, uidi, uici",
    "in principio erat uerbum",
    "senatus populusque romanus",
    "alea iacta est",
    "marcus tullius cicero",
    "imperator caesar",
    "res publica",
    "consul romanus"
]

print("\nSuggested Latin prompts to try:")
for i, prompt in enumerate(latin_prompts, 1):
    print(f"{i:2d}. {prompt}")
print(f"\nTo use a different prompt, modify the 'start' variable in {__file__}")
print("Or use: python3 sample_latin.py --start='your prompt here'")