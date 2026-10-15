"""Independent CPU checkpoint reconstruction at the proposed smaller admission."""
from pathlib import Path

source=Path(__file__).with_name('audit_coalesced_body_block.py').read_text()
source=source.replace('n*k<1048576','n*k<524288')
source=source.replace('CPU admission only, no GPU speed or inference claim.', 'CPU admission at524288 values only; no GPU speed or inference claim.')
exec(compile(source,__file__,'exec'))
