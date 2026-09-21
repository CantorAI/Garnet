# SPDX-FileCopyrightText: 2024-2026 CantorAI Inc.
# SPDX-License-Identifier: Apache-2.0

import py_tokenizer

model_name = "models/deepseek-moe-16b-base"
tokenizer = py_tokenizer.get_tokenizer(model_name)
text = "An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is"
py_inputs = tokenizer(text, return_tensors="pt")
inputs = to_xlang(py_inputs)
print(inputs)
