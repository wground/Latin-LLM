"""
Scriptor - Interactive Latin GPT Writing Assistant
An interactive writing assistant for Latin text generation with memory and optimized text delivery
"""
import argparse
import time
from contextlib import nullcontext
import torch

# Import local model
import paths
from artifacts import (load_latin_tokenizer, load_model as _load_checkpoint,
                       load_system_config, resolve_checkpoint, special_token_ids)
from model import GPTConfig, GPT


def parse_args():
    parser = argparse.ArgumentParser(description="Interactive Latin writing assistant")
    parser.add_argument('--max_new_tokens', type=int, default=2000,
                        help='Upper bound on tokens per turn; generation stops early at EOS')
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--top_k', type=int, default=50)
    parser.add_argument('--top_p', type=float, default=None, help='Nucleus sampling threshold')
    parser.add_argument('--min_p', type=float, default=None, help='Min-p sampling threshold')
    parser.add_argument('--repetition_penalty', type=float, default=1.0,
                        help='1.0 = off (default)')
    paths.add_path_args(parser, include_checkpoint=True)
    return parser.parse_args()


args = parse_args()

# -----------------------------------------------------------------------------
# Latin-specific configuration for interactive writing
out_dir = args.out_dir  # directory where Latin model checkpoints are saved
max_new_tokens = args.max_new_tokens
temperature = args.temperature
top_k = args.top_k
repetition_penalty = args.repetition_penalty
seed_base = int(time.time())  # base seed for variety

# Load system-optimized configuration
system_config = load_system_config()
recommended = system_config["recommended_config"]

device = args.device or recommended["device"]
dtype = recommended["dtype"]
compile = recommended["compile"]
enable_tf32 = recommended.get("enable_tf32", False)
if device == 'cpu':
    dtype = 'float32'
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
device_type = device if device in ('cuda', 'mps') else 'cpu'

ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

def load_model():
    """Load the trained Latin model (defaults to ckpt_best.pt)"""
    try:
        ckpt_path = resolve_checkpoint(out_dir, args.checkpoint)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        exit(1)

    model, _ = _load_checkpoint(ckpt_path, device, GPT, GPTConfig)

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

def generate_text(model, prompt_text, encode_fn, decode_fn, generation_seed, eos_id):
    """Generate text based on the given prompt"""
    # Set seed for this generation
    torch.manual_seed(generation_seed)
    if device == 'cuda':
        torch.cuda.manual_seed(generation_seed)

    # Encode the prompt
    start_ids = encode_fn(prompt_text)
    if len(start_ids) == 0:
        # Empty prompt: seed with EOS, the token that precedes every document start.
        # (The old fallback, id 1 = <|pad|>, never occurs in training data.)
        start_ids = [eos_id]
    x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

    # Generate text. EOS stops the turn early rather than padding out to max_new_tokens.
    with torch.no_grad():
        with ctx:
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k,
                               top_p=args.top_p, min_p=args.min_p,
                               repetition_penalty=repetition_penalty,
                               eos_token_id=eos_id)
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
    encode, decode, meta = load_latin_tokenizer(args.data_dir)
    eos_id = special_token_ids(meta)["eos"]
    
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
        full_text = generate_text(model, current_text, encode, decode, generation_seed, eos_id)
        
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