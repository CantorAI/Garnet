# Garnet

# Model download 
(git lfs install) for lfs install

git clone https://huggingface.co/deepseek-ai/deepseek-moe-16b-base


# first version uses libTorch as Tensor and Neural network lib
## build steps
- in Garnet's parent folder create a folder libTorch
- and make a subfolder for Deubg, download https://download.pytorch.org/libtorch/cu121/libtorch-win-shared-with-deps-debug-2.3.0%2Bcu121.zip
- and make anoter subfolder for release, download https://download.pytorch.org/libtorch/cu121/libtorch-win-shared-with-deps-2.3.0%2Bcu121.zip

### for libTorch, if app is debug but lib is release, will cause crash, so use this two folder

