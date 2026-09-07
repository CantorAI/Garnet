# Model Programs

Garnet's Qwen model programs use Python source executed by XLang3. Each
`model.json` selects `.py` entrypoints; relative imports connect the shared
model helpers. The migrated `.x` copies have been removed.

The 27 programs cover text, vision-language, speech recognition, and speech
synthesis. Backend profiles remain separate from model semantics. Checkpoint
and tokenizer assets are external and are not bundled here.

`test/xlang3/models/run_production_capture.py` imports all programs and captures
40 graphs with tiny symbolic profiles. This validates capture, not pretrained
inference or complete backend operator coverage. See the production-capture
coverage document under that test directory for the verified limits.
