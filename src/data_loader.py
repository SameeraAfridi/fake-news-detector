import os, zipfile, glob, pandas as pd

def extract_zip(zip_path, extract_to="data"):
    if os.path.exists(extract_to):
        import shutil
        shutil.rmtree(extract_to)
    os.makedirs(extract_to, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_to)
    print(f"Extracted {zip_path} -> {extract_to}")
    return extract_to

def load_dataset(data_dir="data"):
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    fake = pd.read_csv([f for f in csv_files if "Fake" in f][0])
    true = pd.read_csv([f for f in csv_files if "True" in f][0])
    fake["label"] = 0
    true["label"] = 1
    df = pd.concat([fake, true], axis=0).sample(frac=1).reset_index(drop=True)
    df["text"] = df["title"] + " " + df["text"]
    return df
