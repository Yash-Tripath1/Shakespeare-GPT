# download_gutenberg.py
import urllib.request

url = "https://www.gutenberg.org/files/100/100-0.txt"
print("Downloading from Project Gutenberg...")
urllib.request.urlretrieve(url, "shakespeare_raw.txt")

# Clean it (Gutenberg adds legal headers/footers)
with open("shakespeare.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Find the actual content (between START and END markers)
start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK"
end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK"

start_idx = text.find(start_marker)
end_idx = text.find(end_marker)

if start_idx != -1 and end_idx != -1:
    # Get text after the start marker
    clean_text = text[start_idx:end_idx]
    # Remove the marker line itself
    clean_text = clean_text.split('\n', 1)[1]
else:
    clean_text = text  # Fallback

# Save cleaned version
with open("shakespeare.txt", "w", encoding="utf-8") as f:
    f.write(clean_text)

print(f"✓ Cleaned text saved ({len(clean_text):,} characters)")