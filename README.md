# Garnet

## XLang serving interface

The embedded serving API supports `Qwen3-VL-2B-Instruct` and the text-only
`Qwen3-1.7B` checkpoint:

```text
serve_model(model_root, xmodel_root, cache_root, profile_json, model_id)
serve_status_json()
infer_json(prompt, image_or_empty, max_new_tokens)
stop_serving()
```

`serve_status_json` reports `model_id` and `input_capability`. Vision inference
requires JPEG bytes or a path; text inference passes an empty image value.
The packaged XModel directories include both single-request decode graphs and
masked batched decode graphs for scheduler-driven execution.

# Model download
(git lfs install) for lfs install

git clone https://huggingface.co/deepseek-ai/deepseek-moe-16b-base


# first version uses libTorch as Tensor and Neural network lib
## build steps
- in Garnet's parent folder create a folder libTorch
- and make a subfolder for Deubg, download https://download.pytorch.org/libtorch/cu121/libtorch-win-shared-with-deps-debug-2.3.0%2Bcu121.zip
- and make anoter subfolder for release, download https://download.pytorch.org/libtorch/cu121/libtorch-win-shared-with-deps-2.3.0%2Bcu121.zip

### for libTorch, if app is debug but lib is release, will cause crash, so use this two folder

