# https://huggingface.co/docs/transformers/tasks/question_answering

import json
import tqdm
import csv
from transformers import AutoTokenizer
import os


class LoadSQuADQuestionAnsweringDataset():
    
    def __init__(
        self,
        tokenizer: str,
        filepath: list[str]
    ) -> None:
        """constuctor

        Args:
            tokenizer (str): the tokenizer name
        """
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.filepath = filepath
        
        self.examples = self.get_examples()
        self.inputs = []
        for example in tqdm.tqdm(self.examples):
            self.inputs.append(self.preprocess_function(example))
        
    
    def get_examples(self):
        
        data = []
        for file in self.filepath:
            with open(file, "r") as f:
                raw_data = json.loads(f.read())
                print(len(raw_data["data"]))
                data.extend(raw_data["data"])
            
        examples = []
        print(len(data))
        for i in tqdm.tqdm(range(len(data)), ncols=80):
            
            title = data[i]["title"]
            paragraphs = data[i]["paragraphs"]
            
            for j in range(len(paragraphs)):
                context = paragraphs[j]["context"]
                qas = paragraphs[j]["qas"]
                
                for k in range(len(qas)):
                    question = qas[k]["question"]
                    id = qas[k]["id"]
                    answers = qas[k]["answers"]
                    if "is_impossible" in qas[k]:
                        is_impossible = qas[k]["is_impossible"]
                    else:
                        is_impossible = False
                    answer_start = []
                    text = []
                    if is_impossible:
                        answer_start.append(-1)
                        text.append("")
                    else:
                        for answer in answers:
                            answer_start.append(answer["answer_start"])
                            text.append(answer["text"])

                    example = {
                        "answers": {
                            "answer_start": answer_start,
                            "text": text,
                        },
                        "context": context,
                        "id": id,
                        "question": question,
                        "title": title
                    }
                    examples.append(example)
        
        return examples
    
    def preprocess_function(self, example):
        questions = example["question"].strip()
        answers = example["answers"]
        context = example["context"]
        
        inputs = self.tokenizer(
            text=questions,
            text_pair=context,
            max_length=510,
            truncation="only_second",
            return_offsets_mapping=True,
            padding="max_length",
        )
        
        offset = inputs.pop("offset_mapping")
        
        start_positions = []
        end_positions = []
        
        star_char = answers["answer_start"][0]
        end_char = answers["answer_start"][0] + len(answers["text"][0])
        sequence_ids = inputs.sequence_ids(0)
        
        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1
        
        # If the answer is not fully inside the context, label it (0, 0)
        if offset[context_start][0] > end_char or offset[context_end][1] < star_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= star_char:
                idx += 1
            start_positions.append(idx - 1)
            
            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        
        return inputs
    
    def save_to_csv(self, des_path):
        with open(des_path, "w", newline="") as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(["input_ids", "attention_mask", "start_positions", "end_positions"])
            for item in self.inputs:
                csv_writer.writerow([
                    item["input_ids"],
                    item["attention_mask"],
                    item["start_positions"],
                    item["end_positions"]
                ])
        
        
def main():
    os.makedirs("./data", exist_ok=True)
    
    filepath = {
        "train": ["./prepare_data/dataset/jaquad_train.json", "./prepare_data/dataset/train-v2.0.json"],
        "dev": ["./prepare_data/dataset/jaquad_dev.json", "./prepare_data/dataset/dev-v2.0.json"],
    }
    
    for data_set, data_path in filepath.items():
        dt = LoadSQuADQuestionAnsweringDataset(tokenizer="xlm-roberta-base", filepath=data_path)
        dt.save_to_csv(f"./data/{data_set}_inputs.csv")
    
    print("Done!")

if __name__ == "__main__":
    main()
