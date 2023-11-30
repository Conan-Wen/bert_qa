import torch
import ast
from torch.utils.data import Dataset
import pandas as pd

class QuAD(Dataset):
    def __init__(self, csv_file):
        super().__init__()
        self.df = pd.read_csv(csv_file)
    
    def __getitem__(self, index):
        item = self.df.iloc[index]
        
        input_ids = ast.literal_eval(item["input_ids"])
        attention_mask = ast.literal_eval(item["attention_mask"])
        start_positions = ast.literal_eval(item["start_positions"])
        end_positions = ast.literal_eval(item["end_positions"])
        
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        start_positions = torch.tensor(start_positions)
        end_positions = torch.tensor(end_positions)
        
        return (input_ids, attention_mask), (start_positions, end_positions)
    
    def __len__(self):
        return len(self.df)

# def main():
#     quad = QuAD("./data/dev_inputs.csv")
    
#     for item in quad:
#         for key, value in item.items():
#             print(f"{key}: {value}")
#         break

# if __name__ == "__main__":
#     main()
        