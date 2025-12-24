import os
import re

# Set this to the root directory containing the .csv files
ROOT_DIR = "/store/real/hjchoi92/data/real/umift/WBW-iph-b5/coinft"

# Regex pattern to match 'WBW_iph_b' followed by a digit, e.g., 'WBW_iph_b0'
pattern = re.compile(r"WBW_iph_b(\d+)")

for root, dirs, files in os.walk(ROOT_DIR):
    for filename in files:
        if filename.endswith(".csv"):
            match = pattern.search(filename)
            if match:
                b_num = match.group(1)
                new_filename = filename.replace(f"WBW_iph_b{b_num}", f"WBW-iph-b{b_num}")
                old_path = os.path.join(root, filename)
                new_path = os.path.join(root, new_filename)

                print(f"Renaming: {filename} → {new_filename}")
                os.rename(old_path, new_path)

print("Done renaming.")
