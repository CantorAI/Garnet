#pragma once
#include <cstddef>
#include <vector>
#include <map>
#include <set>
#include <algorithm>
#include <stdexcept>
#include <utility>
namespace Garnet {
struct WorkspaceInterval {size_t slot,rows,columns,first,last;};
struct WorkspaceBinding {size_t slot,rows,columns,first,last,color;};
inline std::vector<WorkspaceBinding> ColorWorkspace(std::vector<WorkspaceInterval> intervals) {
    std::sort(intervals.begin(),intervals.end(),[](const auto&a,const auto&b){return std::make_pair(a.first,a.slot)<std::make_pair(b.first,b.slot);});
    std::map<std::pair<size_t,size_t>,std::vector<size_t>> ends;std::set<size_t> seen;std::vector<WorkspaceBinding> result;
    for(const auto&i:intervals){if(!i.rows||!i.columns||i.first>i.last||!seen.insert(i.slot).second)throw std::invalid_argument("Invalid scratch interval");
        auto& colors=ends[{i.rows,i.columns}];size_t color=0;while(color<colors.size()&&colors[color]>=i.first)++color;
        if(color==colors.size())colors.push_back(i.last);else colors[color]=i.last;
        result.push_back({i.slot,i.rows,i.columns,i.first,i.last,color});}
    return result;
}
}
