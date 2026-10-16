"""Follow-up: reserve block tiles for wide-output projections; original other routes."""
from make_block_packed_prefill import replace,header,shader,context as original_context,harness as original_harness

def context(source):
 source=original_context(source)
 source=replace(source,'                const size_t nTiles=(rows+63)/64,kTiles=(columns+31)/32;','                if(rows>1024){\n                const size_t nTiles=(rows+63)/64,kTiles=(columns+31)/32;')
 needle='                if(!value->blockPackedPrefill)throw std::runtime_error("Block packed allocation failed");'
 return replace(source,needle,needle+'\n                }')

def harness(source):
 return replace(original_harness(source),'p.selected==p.weight.InterleavedBodyReady()','p.selected==(p.weight.InterleavedBodyReady()&&p.n>1024)')
