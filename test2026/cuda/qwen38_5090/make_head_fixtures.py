"""Tiny exact-integer head fixtures. These are not model checkpoints or benchmarks."""
import argparse, json, struct
from pathlib import Path


def save(path, head_kind):
    rows, width = 32, 64
    embed = [float(row) for row in range(rows) for _ in range(width)]
    def bf16(values):
        return b''.join(struct.pack('<f',v)[2:] for v in values)
    entries = {'model.language_model.embed_tokens.weight':('BF16',[rows,width],bf16(embed))}
    if head_kind == 'untied':
        head = [float(row*2) if col == 0 else 0. for row in range(rows) for col in range(width)]
        entries['lm_head.weight'] = ('BF16',[rows,width],bf16(head))
    elif head_kind == 'packed': entries['lm_head.weight'] = ('U8',[rows,width//2],bytes(rows*width//2))
    header, data = {}, b''
    for name, (dtype,shape,raw) in entries.items():
        header[name] = {'dtype':dtype,'shape':shape,'data_offsets':[len(data),len(data)+len(raw)]}
        data += raw
    h = json.dumps(header,separators=(',',':')).encode()
    h += b' '*((-len(h))%8)
    path.mkdir(parents=True,exist_ok=True)
    (path/'model.safetensors').write_bytes(struct.pack('<Q',len(h))+h+data)
    (path/'config.json').write_text(json.dumps({'text_config':{'tie_word_embeddings':head_kind == 'tied'}}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('output',type=Path)
    a = p.parse_args()
    for kind in ('tied','untied','packed','missing'): save(a.output/kind,kind)
