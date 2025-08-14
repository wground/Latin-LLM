"""
Prepare Latin training data with custom tokenizer
Creates a ByteLevelBPE tokenizer trained on the Latin corpus for better accuracy
"""
import os
import glob
import pickle
import numpy as np
from pathlib import Path
from tokenizers import ByteLevelBPETokenizer

# Configuration
training_data_path = os.path.join(os.path.dirname(__file__), 'Training Data')
tokenizer_dir = "tokenizer_latin"
output_dir = Path("gpt_data_latin")
corpus_file = "latin_corpus_merged.txt"

# Create directories
Path(tokenizer_dir).mkdir(exist_ok=True)
output_dir.mkdir(exist_ok=True)

print("=" * 60)
print("    🏛️  Latin Data Preparation with Custom Tokenizer  🏛️")
print("=" * 60)

# Step 1: Find and merge all text files (including subdirectories)
txt_files = glob.glob(os.path.join(training_data_path, '**', '*.txt'), recursive=True)
print(f"Found {len(txt_files)} .txt files in Training Data folder and subdirectories")

if len(txt_files) == 0:
    print(f"Error: No .txt files found in {training_data_path}")
    print("Make sure your Latin texts are in the 'Training Data' folder or its subdirectories")
    exit(1)

# Merge all text files into one corpus
print("\nMerging text files...")
merged_data = ""
total_files_processed = 0

for txt_file in sorted(txt_files):
    # Show relative path from Training Data directory for better organization visibility
    rel_path = os.path.relpath(txt_file, training_data_path)
    print(f"  Processing: {rel_path}")
    
    try:
        with open(txt_file, 'r', encoding='utf-8') as f:
            file_content = f.read().strip()
            if file_content:  # Only add if file has content
                merged_data += file_content
                merged_data += "\n\n"  # Separator between files
                total_files_processed += 1
    except Exception as e:
        print(f"  ⚠️  Error reading {rel_path}: {e}")
        continue

print(f"\nSuccessfully processed {total_files_processed} files")
print(f"Total characters in corpus: {len(merged_data):,}")

if len(merged_data) == 0:
    print("Error: No readable content found in text files")
    exit(1)

# Save merged corpus
print(f"\nSaving merged corpus to {corpus_file}...")
with open(corpus_file, 'w', encoding='utf-8') as f:
    f.write(merged_data)

# Step 2: Train custom tokenizer on Latin corpus
print(f"\nTraining ByteLevelBPE tokenizer on Latin corpus...")
print("This may take a few minutes...")

# Initialize tokenizer
tokenizer = ByteLevelBPETokenizer()

# Train tokenizer on our Latin corpus
# Vocab size should be reasonable for Latin - not too large to avoid overfitting
vocab_size = 16000  # Smaller than GPT-2's 50k, appropriate for Latin
min_frequency = 2   # Minimum frequency for tokens

print(f"  Vocabulary size: {vocab_size}")
print(f"  Minimum frequency: {min_frequency}")

tokenizer.train(
    files=[corpus_file],
    vocab_size=vocab_size,
    min_frequency=min_frequency,
    special_tokens=["<|endoftext|>", "<|pad|>"]
)

# Save tokenizer
vocab_file = os.path.join(tokenizer_dir, "vocab.json")
merges_file = os.path.join(tokenizer_dir, "merges.txt")

tokenizer.save_model(tokenizer_dir)
print(f"✅ Tokenizer saved to {tokenizer_dir}/")
print(f"   - vocab.json: {vocab_size} tokens")
print(f"   - merges.txt: BPE merge rules")

# Step 3: Encode the corpus with our custom tokenizer
print(f"\nEncoding corpus with custom Latin tokenizer...")

# Split corpus into train/val first (80% train, 20% val)
split_idx = int(0.8 * len(merged_data))
train_text = merged_data[:split_idx]
val_text = merged_data[split_idx:]

print(f"  Train text: {len(train_text):,} characters")
print(f"  Val text: {len(val_text):,} characters")

# Encode train and validation sets
print("  Encoding train set...")
train_encoded = tokenizer.encode(train_text)
train_ids = np.array(train_encoded.ids, dtype=np.uint16)

print("  Encoding validation set...")
val_encoded = tokenizer.encode(val_text)
val_ids = np.array(val_encoded.ids, dtype=np.uint16)

print(f"  Train tokens: {len(train_ids):,}")
print(f"  Val tokens: {len(val_ids):,}")

# Step 4: Save binary files and metadata
print(f"\nSaving training data to {output_dir}/...")

# Save binary files
train_ids.tofile(output_dir / "train.bin")
val_ids.tofile(output_dir / "val.bin")

# Save metadata for training and sampling scripts
actual_vocab_size = tokenizer.get_vocab_size()
print(f"✅ Actual vocabulary size after training: {actual_vocab_size}")

meta = {
    "vocab_size": actual_vocab_size,
    "requested_vocab_size": vocab_size,
    "tokenizer_config": {
        "vocab_file": vocab_file,
        "merges_file": merges_file,
        "type": "ByteLevelBPE"
    },
    "data_stats": {
        "total_chars": len(merged_data),
        "train_chars": len(train_text),
        "val_chars": len(val_text),
        "train_tokens": len(train_ids),
        "val_tokens": len(val_ids),
        "files_processed": total_files_processed
    }
}

with open(output_dir / "meta.pkl", "wb") as f:
    pickle.dump(meta, f)

# Test tokenizer with some Latin text
print(f"\n🧪 Testing tokenizer on sample Latin text:")
test_text = "gallia est omnis diuisa in partes tres"
test_encoded = tokenizer.encode(test_text)
test_decoded = tokenizer.decode(test_encoded.ids)
print(f"  Original: '{test_text}'")
print(f"  Tokens: {test_encoded.ids}")
print(f"  Decoded: '{test_decoded}'")
print(f"  Token count: {len(test_encoded.ids)}")

print(f"\n✅ Latin data preparation complete!")
print(f"\nGenerated files:")
print(f"  📁 {output_dir}/train.bin - Training data ({len(train_ids):,} tokens)")
print(f"  📁 {output_dir}/val.bin - Validation data ({len(val_ids):,} tokens)")
print(f"  📁 {output_dir}/meta.pkl - Metadata and tokenizer config")
print(f"  📁 {tokenizer_dir}/ - Custom Latin tokenizer")
print(f"  📁 {corpus_file} - Merged corpus file")

print(f"\n🏛️  Ready for training with train_latin.py!")
print(f"The custom tokenizer will improve Latin text quality and reduce gibberish.")