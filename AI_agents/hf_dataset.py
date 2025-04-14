from datasets import load_dataset

dataset = load_dataset("ajibawa-2023/Children-Stories-Collection")
print(dataset)
# Extract text and save to a file

for i, item in enumerate(dataset["train"]):  # Assuming the dataset has a "train" split
    if "text" in item:  # Check if the "text" field exists
        text = item["text"]
        with open(f"data/story_{i}.txt", "w", encoding="utf-8") as f:
            f.write(text)
    else:
        print("No 'text' field found in the dataset item:", item)

