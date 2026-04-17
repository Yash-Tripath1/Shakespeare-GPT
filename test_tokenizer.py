# test_tokenizer.py
from tokenizer import CharTokenizer

# 1. Load our clean Shakespeare data
print("Loading data from shakespeare.txt...")
try:
    with open("shakespeare.txt", "r", encoding='utf-8') as f:
        text = f.read()
    print(f"✓ Successfully loaded {len(text):,} characters.")
except FileNotFoundError:
    print("❌ ERROR: 'shakespeare.txt' not found. Make sure it's in the same folder.")
    exit()

# 2. Create a tokenizer FROM our text
#    This is the crucial step. The tokenizer learns the vocabulary
#    from the text we pass to its __init__ method.
print("\nBuilding tokenizer from the text...")
tokenizer = CharTokenizer(text)

# 3. Inspect the tokenizer
print(f"Vocabulary size: {tokenizer.vocab_size}")
print(f"First 20 characters in vocab: {''.join(tokenizer.chars[:20])}")
# This should now show a vocab size of ~100 and list the characters.

# 4. Test encoding and decoding
print("\n--- Testing encode/decode ---")
original_text = "Hello, world!"
print(f"Original: '{original_text}'")

try:
    encoded_text = tokenizer.encode(original_text)
    print(f"Encoded: {encoded_text}")

    decoded_text = tokenizer.decode(encoded_text)
    print(f"Decoded: '{decoded_text}'")

    assert original_text == decoded_text
    print("\n✓ Encode/Decode test passed!")
except KeyError as e:
    print(f"\n❌ ERROR during encoding: {e}")
    print("This means a character in 'Hello, world!' was not found in your Shakespeare text.")
    print("Is the character in your vocabulary list above? Let's check...")
    for char in original_text:
        if char not in tokenizer.chars:
            print(f"  - Character '{char}' is MISSING from the vocab!")

# 5. Save the tokenizer for later use
tokenizer_path = "tokenizer.json"
tokenizer.save(tokenizer_path)
print(f"\n✓ Tokenizer vocabulary saved to '{tokenizer_path}'")

# 6. Test loading the tokenizer
print("\n--- Testing loading ---")
loaded_tokenizer = CharTokenizer.load(tokenizer_path)
print(f"Loaded vocab size: {loaded_tokenizer.vocab_size}")
test_load = loaded_tokenizer.decode(loaded_tokenizer.encode("Test"))
assert test_load == "Test"
print("✓ Loading test passed!")