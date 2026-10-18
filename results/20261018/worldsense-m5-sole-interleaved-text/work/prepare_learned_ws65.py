from pathlib import Path
p=Path(__file__).resolve();old=p.with_name('prepare_learned_ws64.py');text=old.read_text().replace('worldsense-direct-interleaved-prefill','worldsense-sole-interleaved-text').replace('ws64.py','ws65.py').replace('WS64','WS65').replace('==5 and all','==6 and all').replace('len(manifest)==1721','len(manifest)==1722').replace("['off','primary','on','fault','restore']","['off','primary','on','fault','restore','direct64']")
# Run the recorded adaptation in this process; it compiles proof tools and freezes closure only after native proof.
exec(compile(text,str(p),'exec'))
