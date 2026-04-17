with open("shakespeare.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Basic stats
print(f"Total characters: {len(text):,}")
print(f"Total lines: {len(text.splitlines()):,}")

# Unique characters (this will be our vocabulary!)
unique_chars = sorted(list(set(text)))
print(f"\nUnique characters: {len(unique_chars)}")
print(f"Vocabulary: {''.join(unique_chars)}")

# Sample
print(f"\n--- First 300 characters ---")
print(text[:300])