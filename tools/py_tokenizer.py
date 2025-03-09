from transformers import AutoTokenizer

def get_tokenizer(model_name):
    return AutoTokenizer.from_pretrained(model_name)
