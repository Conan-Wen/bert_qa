import requests
import tqdm
import os


DATASET_URL = {
    "train": "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json",
    "dev": "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json"
}

DATASET_DIR = "./data"

DATASET_PATH = {
    "train": "./prepare_data/dataset/train-v2.0.json",
    "dev": "./prepare_data/dataset/dev-v2.0.json"
}

# Download the SQuAD dataset
def download_file():
    for dataset in DATASET_URL.keys():
        try:
            response = requests.get(DATASET_URL[dataset], stream=True)
            total_size = int(response.headers.get("content-length", 0))
            block_size = 1024 # 1 KB
            progress_bar = tqdm.tqdm(total=total_size, unit='B', unit_scale=True)
            
            os.makedirs(DATASET_DIR, exist_ok=True)
            with open(os.path.join(DATASET_DIR, DATASET_PATH[dataset]), "wb") as file:
                for data in response.iter_content(block_size):
                    progress_bar.update(len(data))
                    file.write(data)
                    
            progress_bar.close()
            
        except Exception as e:
            print(e)
            os.remove(DATASET_PATH[dataset])
            

def main():
    download_file()
    print("Done!")

if __name__ == "__main__":
    main()
    