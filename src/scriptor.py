"""
Scriptor - Interactive Latin GPT Writing Assistant
An interactive writing assistant for Latin text generation with memory and optimized text delivery
"""
import os
import json
import pickle
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

# -----------------------------------------------------------------------------
# Latin-specific configuration for interactive writing
out_dir = 'out-latin'  # directory where Latin model checkpoints are saved
max_new_tokens = 2000  # increased from 1000 for longer generations
temperature = 0.7  # reduced from 0.8 for better quality with regularized model
top_k = 50  # increased from 40 for more diverse sampling
seed_base = int(time.time())  # base seed for variety

# Load system-optimized configuration
system_config = load_system_config()
recommended = system_config["recommended_config"]

device = recommended["device"]
dtype = recommended["dtype"] 
compile = recommended["compile"]
enable_tf32 = recommended.get("enable_tf32", False)
# -----------------------------------------------------------------------------

# Apply hardware optimizations based on detected system
if device == 'cuda':
    if enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device == 'mps':
    # Apple Silicon optimizations
    pass  # MPS uses unified memory

# Set up mixed precision context based on detected capabilities
if device == 'mps':
    device_type = 'cpu'  # MPS uses CPU autocast
elif device == 'cuda':
    device_type = 'cuda'
else:
    device_type = 'cpu'

ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if (device_type == 'cpu' and device == 'cpu') else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

def load_model():
    """Load the trained Latin model"""
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
    print(f"🏛️  Latin Scriptor Configuration:")
    print(f"   Device: {device} ({dtype})")
    print(f"   Compilation: {'enabled' if compile else 'disabled'}")
    if device == 'cuda' and enable_tf32:
        print(f"   TF32 optimization: enabled")

    if compile:
        if device == 'mps':
            print("⚠️  Model compilation disabled on MPS due to compatibility issues")
        else:
            print("⚡ Compiling model for faster inference...")
            model = torch.compile(model)
    
    return model

def trim_to_whitespace(text: str) -> str:
    """Trim text to end on whitespace or paragraph break for natural stopping"""
    if not text:
        return text
    
    # Look for good stopping points in reverse order of preference
    # 1. Double newline (paragraph break) - best
    double_newline_pos = text.rfind('\n\n')
    if double_newline_pos > len(text) * 0.7:  # Only if it's in the latter part
        return text[:double_newline_pos + 2]
    
    # 2. Single newline - good
    newline_pos = text.rfind('\n')
    if newline_pos > len(text) * 0.7:
        return text[:newline_pos + 1]
    
    # 3. Sentence ending (. ! ?) followed by space - decent
    for punct in ['. ', '! ', '? ']:
        punct_pos = text.rfind(punct)
        if punct_pos > len(text) * 0.7:
            return text[:punct_pos + 1]
    
    # 4. Any whitespace - minimal acceptable
    for i in range(len(text) - 1, -1, -1):
        if text[i].isspace():
            return text[:i + 1]
    
    # If no whitespace found, return as is (shouldn't happen with proper Latin text)
    return text

def generate_text(model, prompt_text, encode_fn, decode_fn, generation_seed):
    """Generate text based on the given prompt"""
    # Set seed for this generation
    torch.manual_seed(generation_seed)
    if device == 'cuda':
        torch.cuda.manual_seed(generation_seed)
    
    # Encode the prompt
    start_ids = encode_fn(prompt_text)
    if len(start_ids) == 0:
        # Handle empty prompts - use token ID 1 as fallback
        start_ids = [1]
    x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]
    
    # Generate text
    with torch.no_grad():
        with ctx:
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            generated_text = decode_fn(y[0].tolist())
    
    return generated_text

def main():
    """Main interactive loop for Scriptor"""
    print("=" * 70)
    print("    🏛️  SCRIPTOR - Interactive Latin Writing Assistant  🏛️")
    print("=" * 70)
    print("Commands:")
    print("  '1' + Enter = Continue writing from where I left off")
    print("  '2' + Enter = Exit")
    print("  Any other text = Add your input and continue writing")
    print("=" * 70)
    
    # Load model and tokenizer
    model = load_model()
    encode, decode = load_latin_tokenizer()
    
    # Get initial prompt from user
    print("\nEnter your initial prompt to start writing:")
    initial_prompt = input("> ")
    
    if not initial_prompt.strip():
        print("Empty prompt provided. Starting with blank slate...")
        current_text = ""
    else:
        current_text = initial_prompt
    
    generation_count = 0
    
    while True:
        print("\n" + "=" * 70)
        print(f"scribens... (Generation #{generation_count + 1})")
        print("=" * 70)
        
        # Generate text based on current context (including all previous output)
        generation_seed = seed_base + generation_count
        full_text = generate_text(model, current_text, encode, decode, generation_seed)
        
        # Extract only the newly generated portion
        if current_text:
            new_text = full_text[len(current_text):]
        else:
            new_text = full_text
        
        # Trim new text to end on whitespace/paragraph break
        new_text = trim_to_whitespace(new_text)
        
        # Display the generated text
        if new_text.strip():
            print(new_text)
            
            # Update current text to include the new generation
            current_text = current_text + new_text if current_text else new_text
            generation_count += 1
        else:
            print("(No new text generated)")
        
        print("\n" + "-" * 70)
        print("quod vis deinde?")
        print("  '1' = Continue writing")
        print("  '2' = Exit")
        print("  Or type your own text to add and continue")
        
        user_input = input("> ")
        
        if user_input == '1':
            # Continue from current text
            continue
        elif user_input == '2':
            print("\nuale! (Farewell!)")
            break
        else:
            # Add user input to current text
            if current_text and not current_text.endswith(' '):
                current_text += " "
            current_text += user_input
            print(f"\nAdded your input: '{user_input}'")

if __name__ == "__main__":
    main()