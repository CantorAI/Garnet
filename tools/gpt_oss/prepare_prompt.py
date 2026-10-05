"""CPython host utility; requires openai-harmony. No model inference here."""
import argparse
import json
from pathlib import Path
from openai_harmony import (Conversation, HarmonyEncodingName, Message, ReasoningEffort,
                            Role, SystemContent, load_harmony_encoding)

parser = argparse.ArgumentParser()
parser.add_argument('prompt')
parser.add_argument('output', type=Path)
parser.add_argument('--max-new-tokens', type=int, default=32)
args = parser.parse_args()
encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
messages = [Message.from_role_and_content(Role.SYSTEM, SystemContent.new().with_reasoning_effort(ReasoningEffort.LOW)),
            Message.from_role_and_content(Role.USER, args.prompt)]
tokens = encoding.render_conversation_for_completion(Conversation.from_messages(messages), Role.ASSISTANT)
args.output.write_text(json.dumps({'input_ids': tokens, 'max_new_tokens': args.max_new_tokens,
    'stop_token_ids': list(encoding.stop_tokens_for_assistant_actions())}))
print('Prepared Harmony input:', args.output)
