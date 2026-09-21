# SPDX-FileCopyrightText: 2024-2026 CantorAI Inc.
# SPDX-License-Identifier: Apache-2.0

from transformers import AutoTokenizer
model_name = "models/deepseek-moe-16b-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
text = "An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is"
inputs = tokenizer(text, return_tensors="pt")
input_ids = inputs["input_ids"].tolist()
attention_mask = inputs["attention_mask"].tolist()
print(inputs)
