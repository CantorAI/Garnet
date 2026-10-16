"""Measure every originally admitted projection separately, preserving exact gates."""
from make_two_body import context,header,shader,harness as base_harness

def harness(source):
    result=base_harness(source)
    replacements=[
        ('#include <vector>','#include <vector>\n#include <string>'),
        ('static void emit(NSDictionary* v){','static NSString* currentMatrix=nil;\nstatic void emit(NSDictionary* v){if(currentMatrix){auto tagged=[v mutableCopy];tagged[@"matrix"]=currentMatrix;v=tagged;}'),
        ('struct Projection{size_t n,k;','struct Projection{std::string name;size_t n,k;'),
        ('Projection p;p.n=','Projection p;p.name=[c[@"name"] UTF8String];p.n='),
        ('    for(size_t batch:{size_t(1),size_t(2)}){','''    auto allProjections=std::move(projections);
    for(const auto& original:allProjections){
        if(!original.selected)continue;
        currentMatrix=[NSString stringWithUTF8String:original.name.c_str()];
        projections.clear();projections.push_back(original);
    for(size_t batch:{size_t(1),size_t(2)}){'''),
        ('    }return 0;','    }}return 0;')]
    for old,new in replacements:
        assert result.count(old)==1,(old,result.count(old))
        result=result.replace(old,new)
    return result
