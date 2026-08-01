"""
Sample from a trained Latin GPT model
Uses system detection for optimal hardware configuration.
"""
import argparse
import time
from contextlib import nullcontext
import torch

# Import local model
import paths
from artifacts import (load_latin_tokenizer, load_model, load_system_config,
                       resolve_checkpoint, special_token_ids)
from model import GPTConfig, GPT

# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Sample from trained Latin GPT model")
    parser.add_argument('--start', type=str, default="caesar ", help='Starting prompt for generation')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of samples to generate')
    parser.add_argument('--max_new_tokens', type=int, default=200, help='Number of tokens per sample')
    parser.add_argument('--temperature', type=float, default=0.7, help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=50, help='Top-k sampling parameter')
    parser.add_argument('--top_p', type=float, default=None, help='Nucleus sampling threshold (e.g. 0.9); off by default')
    parser.add_argument('--min_p', type=float, default=None, help='Min-p sampling threshold (e.g. 0.05); off by default')
    parser.add_argument('--repetition_penalty', type=float, default=1.0,
                        help='Repetition penalty (1.0 = off, the default). Note: this penalizes '
                             'Latin inflectional endings and function-word BPE pieces, so it can '
                             'hurt grammaticality while cosmetically hiding repetition.')
    parser.add_argument('--stop_at_eos', action='store_true', default=True,
                        help='Stop generation at the end-of-document token (default: on)')
    parser.add_argument('--no_stop_at_eos', dest='stop_at_eos', action='store_false')
    parser.add_argument('--seed', type=int, default=None, help='Random seed (if not set, uses current time)')
    paths.add_path_args(parser, include_checkpoint=True)
    return parser.parse_args()

args = parse_args()

# -----------------------------------------------------------------------------
# Latin-specific sampling configuration
out_dir = args.out_dir  # directory where Latin model checkpoints are saved
start = args.start  # Latin prompt to start with
num_samples = args.num_samples  # number of samples to generate
max_new_tokens = args.max_new_tokens  # number of tokens to generate per sample
temperature = args.temperature  # sampling temperature (0.6-0.8 good for regularized Latin model)
top_k = args.top_k  # retain only top_k most likely tokens
repetition_penalty = args.repetition_penalty  # penalize repeated tokens
seed = args.seed if args.seed is not None else int(time.time())  # Use current time for randomness

# Load system-optimized configuration
system_config = load_system_config()
recommended = system_config["recommended_config"]

device = args.device or recommended["device"]
dtype = recommended["dtype"]
compile = recommended["compile"]
enable_tf32 = recommended.get("enable_tf32", False)
# CPU cannot run bfloat16/float16 autocast usefully here; keep it honest.
if device == 'cpu':
    dtype = 'float32'
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
device_type = device if device in ('cuda', 'mps') else 'cpu'

ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# Load the trained Latin model (defaults to ckpt_best.pt, not the rolling ckpt.pt)
try:
    ckpt_path = resolve_checkpoint(out_dir, args.checkpoint)
except FileNotFoundError as e:
    print(f"Error: {e}")
    exit(1)

model, checkpoint = load_model(ckpt_path, device, GPT, GPTConfig)

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
encode, decode, meta = load_latin_tokenizer(args.data_dir)
specials = special_token_ids(meta)
eos_id = specials["eos"]

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
    # Empty prompt: seed with the end-of-document token. Documents are separated by EOS,
    # so EOS is exactly the context that predicts "start of a fresh document". The old
    # code seeded with token id 1 (<|pad|>), which never appears in training data at all.
    start_ids = [eos_id]
    if not meta.get("eos_separated", False):
        print("⚠️  This corpus was built without EOS separators, so the model has never "
              "seen the document-start token. Empty-prompt samples will be unreliable.")
x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

# Provenance: makes any pasted sample traceable back to an exact model + decode config.
ckpt_hash = paths.file_sha1(ckpt_path, max_bytes=1 << 20)
print(f"\nGenerating {num_samples} samples with {max_new_tokens} tokens each:")
print(f"Temperature: {temperature}, Top-k: {top_k}, Top-p: {args.top_p}, Min-p: {args.min_p}")
print(f"Repetition penalty: {repetition_penalty}, Stop at EOS: {args.stop_at_eos}")
print(f"Checkpoint: {ckpt_path.name} (sha1[:12]={ckpt_hash[:12]}), seed: {seed}")
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
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k,
                               top_p=args.top_p, min_p=args.min_p,
                               repetition_penalty=repetition_penalty,
                               eos_token_id=eos_id if args.stop_at_eos else None)
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