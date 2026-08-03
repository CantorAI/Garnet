# Qwen3-ASR and Qwen3-TTS compiled models

Garnet implements Qwen3-ASR-0.6B and the 0.6B/1.7B Qwen3-TTS 12Hz
CustomVoice checkpoints as
native XLang tensor graphs. They use TensorRT BF16, paged KV caches, external
Safetensors weights, and the same `compiled_xmodel` runtime as Qwen text/VL.
No Python or Transformers process is used during inference.

## ASR

Load `xModel/qwen3/asr_0_6b/prefill.x` with entry function
`Qwen3ASRPrefill`, frontend `qwen3_asr`, and nine input profiles:

```text
[1,T], [C,1,128,100], [C+1], [3,1,T], [1,T],
[28,P,16,8,128], [28,P,16,8,128], [P], [1]
```

`C` is the exact number of one-second audio chunks. `T` is the compiled prompt
capacity, and `P*16` must hold the prompt plus generated transcript. A forward
request accepts `audio` or `audio_path`, optional `context` and `language`, and
`max_new_tokens`.

The native frontend reads PCM16/float WAV, downmixes and resamples to 16 kHz,
computes the released 128-bin Whisper log-mel transform, and forms 100-frame
chunks. The graph implements the convolutional subsamplers, chunk-local audio
transformer, audio/text merge, 3-component MRoPE, and text decoder.

## TTS

Load `xModel/qwen3/tts_12hz_0_6b_custom_voice/talker_prefill.x` or
`xModel/qwen3/tts_12hz_1_7b_custom_voice/talker_prefill.x` with entry function
`Qwen3TTSTalkerPrefill`, frontend `qwen3_tts`, and seven profiles:

```text
[1,T,2], [3,1,T], [1,T],
[28,P,16,8,128], [28,P,16,8,128], [P], [1]
```

A request supplies `text`, `speaker`, optional `language` (default `Auto`), and
optional `max_audio_frames`. The 1.7B checkpoint also accepts a separate
`instruct` prompt; upstream Qwen explicitly disables instruction control for
0.6B. The result contains GPU `audio`, `sample_rate`
(24000), `audio_sample_count`, and duration metadata. The runtime trims the
statically compiled codec output to the generated sample count before return.

The runtime executes the talker autoregressively, predicts all 16 codec groups,
then runs the bundled `speech_tokenizer` weights through RVQ reconstruction,
the sliding-window codec transformer, ConvNeXt upsampling, and SnakeBeta
waveform decoder. `GARNET_TTS_MAX_FRAMES` selects the static codec profile
(default 256 frames, about 20.48 seconds) and must be set before model load.

Both models currently expose TensorRT BF16 profiles only. Download weights into
their original Hugging Face directory layouts; TTS requires the
`speech_tokenizer/` subdirectory included with the CustomVoice checkpoint.
