new_path = "phase1/data/molecular_metadata_monomers.csv"
bak_path = "phase1/data/molecular_metadata_monomers.csv.bak"

with open(new_path, "rb") as f:
    new = f.read()
with open(bak_path, "rb") as f:
    bak = f.read()

# Strip two leading spaces from the backup header, then compare
bak_stripped = bak[2:]   # skip the two spaces

if new == bak_stripped:
    print("IDENTICAL after stripping the 2-byte header prefix.")
else:
    print("Still differs — real regression.")
    # Find the first actual difference
    for i, (a, b) in enumerate(zip(new, bak_stripped)):
        if a != b:
            print(f"First real diff at offset {i}: NEW={a!r} BAK={b!r}")
            print(f"  NEW context: ...{new[max(0,i-20):i+20]!r}...")
            print(f"  BAK context: ...{bak_stripped[max(0,i-20):i+20]!r}...")
            break