"""Private 524288-value body admission; preserve current large-weight control."""
def header(source):
    return source


def context(source):
    def replace(old,new):
        nonlocal source
        assert source.count(old)==1,old
        source=source.replace(old,new)
    replace('namespace Garnet {','namespace Garnet {bool gCoalescedBlockEnabled=false;size_t gCoalescedBlockThreads=64;')
    replace('columns>=(1048576+rows-1)/rows','columns>=(524288+rows-1)/rows')
    replace('const bool interleaved=a.rows<=2&&w->interleavedBody;',
            'const bool interleaved=a.rows<=2&&w->interleavedBody&&(gCoalescedBlockEnabled||w->rows*w->columns>=1048576);')
    return source


def harness(source):
    source=source.replace('// Isolated native component: coalesced exact12-bit blocks versus current raw BF16 decode.',
                          '// Private admission test: current coalesced large-body control versus additional smaller body routes.')
    source=source.replace('{1024,512}})', '{1024,512},{896,896},{512,1024},{257,2048}})')
    source=source.replace('n*k>=1048576','n*k>=524288')
    source=source.replace('size_t(64),size_t(128),size_t(256)','size_t(64)')
    source=source.replace('Garnet::gCoalescedBlockThreads=policy==1?64:policy==2?128:256;', 'Garnet::gCoalescedBlockThreads=64;')
    source=source.replace('expected+=policy&&p.selected;', 'expected+=p.selected&&(policy||p.n*p.k>=1048576);')
    source=source.replace('policy<=3','policy<=1').replace('p<=3','p<=1')
    source=source.replace('@"selected":@(p.selected)', '@"selected":@(p.selected),@"baseline_selected":@(p.selected&&p.n*p.k>=1048576)')
    assert 'policy<=3' not in source and 'p<=3' not in source
    assert '{257,2048}' in source and 'expected+=p.selected&&(policy||p.n*p.k>=1048576);' in source
    return source
