import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def require(path: Path, fragments):
    assert path.is_file(), path
    text = path.read_text(encoding="utf-8")
    for fragment in fragments:
        assert fragment in text, (path, fragment)


asr = ROOT / "xModel" / "qwen3" / "asr_0_6b"
manifest = json.loads((asr / "model.json").read_text(encoding="utf-8"))
assert manifest["task"] == "automatic-speech-recognition"
assert manifest["entrypoints"]["prefill"]["frontend"] == "qwen3_asr"
assert manifest["weights"]["source"] == "Qwen/Qwen3-ASR-0.6B"

require(asr / "audio_encoder.x", [
    "qwen3_asr_conv_subsample",
    "qwen3_asr_audio_qkv_packed",
    "qwen3_asr_audio_attention_packed",
    "thinker.audio_tower.proj2.weight",
])
require(asr / "prefill.x", [
    "qwen3_asr_merge_audio_embeddings",
    "paged_kv_prefill_write_bf16",
    "config.thinker_config.text_config",
])
require(asr / "decode.x", [
    "Qwen3ASRDecode",
    "paged_kv_decode_bf16",
])

tts = ROOT / "xModel" / "qwen3" / "tts_0_6b_custom_voice"
tts_manifest = json.loads((tts / "model.json").read_text(encoding="utf-8"))
assert tts_manifest["task"] == "text-to-speech"
assert tts_manifest["entrypoints"]["prefill"]["frontend"] == "qwen3_tts"
assert tts_manifest["weights"]["codec_subdirectory"] == "speech_tokenizer"
require(tts / "talker_common.x", [
    "qwen3_tts_aligned_prompt_embedding",
    "qwen3_tts_pack_hidden_logits",
    "qwen3_tts_decode_embedding",
    "talker.model.text_embedding.weight",
    "talker.codec_head.weight",
])
require(tts / "talker_prefill.x", [
    "Qwen3TTSTalkerPrefill",
    "paged_kv_prefill_write_bf16",
])
require(tts / "talker_decode.x", [
    "Qwen3TTSTalkerDecode",
    "paged_kv_decode_bf16",
])
require(tts / "code_predictor.x", [
    "Qwen3TTSCodePredictor",
    "qwen3_tts_dense_attention_packed",
    "num_code_groups",
])
require(tts / "codec_decode.x", [
    "Qwen3TTSCodecDecode",
    "qwen3_tts_rvq_decode",
    "qwen3_tts_causal_transconv1d",
    "qwen3_tts_snake_beta",
    "qwen3_tts_waveform",
])

print("Qwen3 ASR and TTS XLang model contracts passed")
