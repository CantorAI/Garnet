// SPDX-License-Identifier: Apache-2.0
#pragma once
#if !defined(GARNET_CPU_ONLY) && !defined(GARNET_METAL)
#include "xlang3/xlang3.h"
#include "safetensors_index.h"
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
namespace Garnet {
// GPU request tensors and one BF16 table shared by embedding and the tied head.
// Only token IDs and small request metadata cross the host boundary.
class CUDATensorIO {
    SafeTensorsIndex index_;
    X::Tensor table_,headInput_,headOutput_,tokenHeadInput_,tokenHeadOutput_;
    X::Tensor candidateScratch_,candidateOutput_,indexedScores_,indexedIds_,indexedValues_;
    std::unordered_map<std::string,X::Tensor> projectionWeights_;
    std::shared_ptr<void> projection_;
    int vocab_=0,width_=0;
    mutable std::mutex mutex_;
    X::Value LogitsUnlocked(X::Value input,bool reuse,bool tokens=false);
    X::Value ArgmaxUnlocked(X::Value input,int maxRows=8);
    void CloseUnlocked();
public:
    ~CUDATensorIO();
    bool Open(const std::string& checkpoint);
    void Close();
    X::Value Zeros(X::Value shape);
    X::Value StorageView(X::Value source,X::Value shape);
    X::Value FiniteF32Bytes(X::Value source);
    void Zero(X::Value tensor);
    X::Value Clone(X::Value tensor);
    void CopyInto(X::Value source,X::Value destination);
    void CopyActive(X::Value source,X::Value active,X::Value destination);
    void WriteChunkToken(X::Value source,X::Value destination,int64_t token);
    void CopyLastChunk(X::Value source,X::Value active,X::Value destination);
    void MaskChunk(X::Value hidden,X::Value active);
    void RestoreKVSuffix(X::Value cache,X::Value backup,X::Value spans);
    X::Value FromHost(X::Value values,X::Value shape);
    void Update(X::Value destination,X::Value values);
    void UpdateRequest(X::Value active,X::Value positions,X::Value slots,X::Value flags,X::Value positionValues,X::Value slotValues);
    void EmbeddingChunk(X::Value tokens,X::Value destination);
    X::Value ReadF32(const std::string& path,X::Value shape);
    X::Value Embedding(X::Value tokens);
    X::Value Slice(X::Value source,int64_t start,X::Value shape);
    void SliceInto(X::Value source,int64_t start,X::Value destination);
    X::Value VisualToken(X::Value source,int64_t token);
    X::Value Row(X::Value source,int64_t row);
    X::Value StackRows(X::Value rows);
    X::Value PackPrompt(X::Value embeddingRows,X::Value visualRows,X::Value visualStarts);
    X::Value Logits(X::Value hidden);
    X::Value Argmax(X::Value logits);
    X::Value Predict(X::Value hidden);
    X::Value PredictTokens(X::Value hidden);
    X::Value HeadCandidates(X::Value selectedRows,int64_t count,bool tokenHead);
    X::Value PredictIndexed(X::Value hidden,X::Value candidates);
    std::string WeightInfo(const std::string& name);
    X::Value Weight(const std::string& name);
    X::Value PrepareImage(const std::string& path,int pixels);
    std::string MemoryJson() const;
    int64_t TensorBytes(X::Value value);
    BEGIN_PACKAGE(CUDATensorIO)
        APISET().AddFunc<1>("open",&CUDATensorIO::Open);
        APISET().AddFunc<0>("close",&CUDATensorIO::Close);
        APISET().AddFunc<1>("zeros",&CUDATensorIO::Zeros);
        APISET().AddFunc<2>("storage_view",&CUDATensorIO::StorageView);
        APISET().AddFunc<1>("finite_f32_bytes",&CUDATensorIO::FiniteF32Bytes);
        APISET().AddFunc<1>("zero",&CUDATensorIO::Zero);
        APISET().AddFunc<1>("clone",&CUDATensorIO::Clone);
        APISET().AddFunc<2>("copy_into",&CUDATensorIO::CopyInto);
        APISET().AddFunc<3>("copy_active",&CUDATensorIO::CopyActive);
        APISET().AddFunc<3>("write_chunk_token",&CUDATensorIO::WriteChunkToken);
        APISET().AddFunc<3>("copy_last_chunk",&CUDATensorIO::CopyLastChunk);
        APISET().AddFunc<2>("mask_chunk",&CUDATensorIO::MaskChunk);
        APISET().AddFunc<3>("restore_kv_suffix",&CUDATensorIO::RestoreKVSuffix);
        APISET().AddFunc<2>("from_host",&CUDATensorIO::FromHost);
        APISET().AddFunc<2>("update",&CUDATensorIO::Update);
        APISET().AddFunc<6>("update_request",&CUDATensorIO::UpdateRequest);
        APISET().AddFunc<2>("embedding_chunk",&CUDATensorIO::EmbeddingChunk);
        APISET().AddFunc<2>("read_f32",&CUDATensorIO::ReadF32);
        APISET().AddFunc<1>("embedding",&CUDATensorIO::Embedding);
        APISET().AddFunc<3>("slice",&CUDATensorIO::Slice);
        APISET().AddFunc<3>("slice_into",&CUDATensorIO::SliceInto);
        APISET().AddFunc<2>("visual_token",&CUDATensorIO::VisualToken);
        APISET().AddFunc<2>("row",&CUDATensorIO::Row);
        APISET().AddFunc<1>("stack_rows",&CUDATensorIO::StackRows);
        APISET().AddFunc<3>("pack_prompt",&CUDATensorIO::PackPrompt);
        APISET().AddFunc<1>("logits",&CUDATensorIO::Logits);
        APISET().AddFunc<1>("argmax",&CUDATensorIO::Argmax);
        APISET().AddFunc<1>("predict",&CUDATensorIO::Predict);
        APISET().AddFunc<1>("predict_tokens",&CUDATensorIO::PredictTokens);
        APISET().AddFunc<3>("head_candidates",&CUDATensorIO::HeadCandidates);
        APISET().AddFunc<2>("predict_indexed",&CUDATensorIO::PredictIndexed);
        APISET().AddFunc<1>("weight_info",&CUDATensorIO::WeightInfo);
        APISET().AddFunc<1>("weight",&CUDATensorIO::Weight);
        APISET().AddFunc<2>("prepare_image",&CUDATensorIO::PrepareImage);
        APISET().AddFunc<0>("memory_json",&CUDATensorIO::MemoryJson);
        APISET().AddFunc<1>("tensor_bytes",&CUDATensorIO::TensorBytes);
    END_PACKAGE
};
}
#endif
