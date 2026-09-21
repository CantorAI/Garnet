# xModel

**xModel** is Garnet's backend-neutral model programming format and package
contract. An xModel package contains XLang3 Tensor Expression programs written
as `.py` source with Python-compatible syntax, an xModel manifest (`model.json`),
backend profiles, and contracts for external assets.

Each xModel manifest selects `.py` entrypoints; relative imports connect shared
xModel helpers. The migrated `.x` copies have been removed.

The 27 xModel programs cover text, vision-language, speech recognition, and
speech synthesis. Backend profiles remain separate from xModel semantics.
Checkpoint and tokenizer assets are external and are not bundled here.

`test/xlang3/models/run_production_capture.py` imports all programs and captures
40 graphs with tiny symbolic profiles. This validates capture, not pretrained
inference or complete backend operator coverage. See the production-capture
coverage document under that test directory for the verified limits.
