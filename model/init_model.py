import torch
from transformers import AutoModelForQuestionAnswering

model = AutoModelForQuestionAnswering.from_pretrained("xlm-roberta-base")

torch.save(model, "./model.pth")
