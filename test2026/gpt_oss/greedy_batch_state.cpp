// SPDX-License-Identifier: Apache-2.0
#include "../../src/model/greedy_batch_state.h"
#include <array>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <thread>

namespace {
void Need(bool condition) { if (!condition) throw std::runtime_error("state contract mismatch"); }
template<class F> void Reject(F fn) {
    bool rejected = false;
    try { fn(); } catch (const std::invalid_argument&) { rejected = true; }
    Need(rejected);
}
constexpr std::array<std::int32_t, 6> gold{2,3,5,11,2,6};
std::vector<float> Inputs(std::size_t batch, int offset) {
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    const std::array<std::array<float,4>,6> families{{
        {{3,9,5,2}}, {{4,7,4,3}}, {{nan,5,7,6}},
        {{2,11,nan,1}}, {{-0.0f,4,+0.0f,2}}, {{inf,8,inf,6}}
    }};
    std::vector<float> result;
    for (std::size_t row=0;row<batch;++row) {
        auto values=families[row%families.size()];
        values[1]+=float(offset); values[3]+=float(offset);
        result.insert(result.end(),values.begin(),values.end());
    }
    return result;
}
}
int main() {
    using Garnet::GreedyBatchState;
    std::size_t histories=0, rejected=0;
    for (std::size_t batch:{1,7,224,512}) {
        GreedyBatchState state(batch,3);
        Need(state.GetStatus().ownedBytes >= (batch*3+2*batch)*4);
        Reject([&]{state.History();}); ++rejected;
        Reject([&]{state.Current(0);}); ++rejected;
        std::vector<std::int32_t> first;
        for (int step=0;step<3;++step) {
            const auto inputs=Inputs(batch,step*32), saved=inputs;
            const auto selected=state.Consume(inputs.data(),inputs.size());
            Need(selected.size()==batch);
            Need(std::memcmp(inputs.data(),saved.data(),inputs.size()*4)==0);
            if(step<2) {
                auto current=state.Current(std::size_t(step+1));
                for(std::size_t row=0;row<batch;++row)Need(current[row]==gold[row%6]+step*32);
                if(!step)first=current;
                Reject([&]{state.Current(std::size_t(step+2));}); ++rejected;
                Reject([&]{state.Reset();}); ++rejected;
                // Two rank workers receive independent by-value snapshots.
                std::array<std::vector<std::int32_t>,2> copies;
                std::thread a([&]{copies[0]=state.Current(std::size_t(step+1));});
                std::thread b([&]{copies[1]=state.Current(std::size_t(step+1));});
                a.join();b.join();Need(copies[0]==current && copies[1]==current);
            } else {
                auto finalCurrent=state.Current(3);
                for(std::size_t row=0;row<batch;++row)
                    Need(finalCurrent[row]==gold[row%6]+64);
            }
        }
        for(std::size_t row=0;row<batch;++row)Need(first[row]==gold[row%6]);
        auto history=state.History();Need(history.size()==batch*3);
        for(std::size_t row=0;row<batch;++row)for(int step=0;step<3;++step)
            Need(history[row*3+std::size_t(step)]==gold[row%6]+step*32);
        ++histories;
        state.Reset();Need(state.GetStatus().step==0);
        for(int step=0;step<3;++step){const auto input=Inputs(batch,step*32);state.Consume(input.data(),input.size());}
        Need(state.History()==history);
        state.Release();state.Release();Need(state.GetStatus().released && state.GetStatus().ownedBytes==0);
        Reject([&]{state.Reset();});Reject([&]{state.History();});Reject([&]{state.Current(1);});
        rejected+=3;
    }
    for(float invalid:{-1.0f,.5f,16777216.0f,std::numeric_limits<float>::infinity(),std::numeric_limits<float>::quiet_NaN()}) {
        for(int column:{1,3}) {
            GreedyBatchState state(7,2);auto input=Inputs(7,0);state.Consume(input.data(),input.size());
            input[6*4+std::size_t(column)]=invalid;
            Reject([&]{state.Consume(input.data(),input.size());});++rejected;
            Need(state.GetStatus().poisoned && state.GetStatus().step==1);
            Reject([&]{state.Current(1);});Reject([&]{state.History();});rejected+=2;
            state.Reset();Need(!state.GetStatus().poisoned && state.GetStatus().step==0);
        }
    }
    for(auto shape:{std::pair<std::size_t,std::size_t>{0,3},{513,3},{1,0},{1,2049},{std::size_t(-1),3}}){
        Reject([&]{GreedyBatchState state(shape.first,shape.second);});++rejected;
    }
    for(std::size_t count:{std::size_t(0),std::size_t(3),std::size_t(5)}){
        GreedyBatchState state(1,2);auto input=Inputs(1,0);
        Reject([&]{state.Consume(input.data(),count);});++rejected;Need(state.GetStatus().poisoned && state.GetStatus().step==0);
    }
    GreedyBatchState nullState(1,1);Reject([&]{nullState.Consume(nullptr,4);});++rejected;
    GreedyBatchState boundary(1,1);const float valid[]{1,16777215,0,0};boundary.Consume(valid,4);
    Need(boundary.History()==std::vector<std::int32_t>{16777215});
    Need(boundary.Current(1)==std::vector<std::int32_t>{16777215});
    Reject([&]{boundary.Consume(valid,4);});++rejected;
    Need(boundary.GetStatus().poisoned && boundary.GetStatus().step==1);
    std::cout<<"GREEDY_BATCH_HOST_STATE_PASS histories="<<histories<<" rejected="<<rejected
        <<" exact_manual_gold_ties_nan_source_immutable_snapshot_reset_release\n";
}
