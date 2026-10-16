"""Generate isolated direct-row MPS context/harness, preserving the raw control."""
def context(source):
    needle='namespace Garnet {';assert source.count(needle)==1
    source=source.replace(needle,needle+' bool gDirectMPS=false;')
    needle='if((a.rows>2||w->visionPrefill)&&w->exactHalf&&w->halfScales)';assert source.count(needle)==1
    return source.replace(needle,'if((a.rows>2||w->visionPrefill||gDirectMPS)&&w->exactHalf&&w->halfScales)')
def harness(source):
    needle='#include "metal_context.h"';assert source.count(needle)==1
    source=source.replace(needle,needle+'\nnamespace Garnet {extern bool gDirectMPS;}')
    replacements=[('const size_t physical[]={batch,4,8,16}','const size_t physical[]={batch,batch}'),('input(4),output(4)','input(2),output(2)'),('p<4','p<2'),('bool admitted[]={true,true,true,true}','bool admitted[]={true,true}'),('auto run=[&](size_t policy){','auto run=[&](size_t policy){Garnet::gDirectMPS=policy!=0;'),('padded_rows_zero','no_extra_rows')]
    for old,new in replacements:assert old in source;source=source.replace(old,new)
    return source
