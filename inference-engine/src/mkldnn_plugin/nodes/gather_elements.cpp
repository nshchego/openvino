// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "base.hpp"

#include <string>
#include <vector>
#include "ie_parallel.hpp"
#include "jit_generator.hpp"
#include "common/cpu_memcpy.h"
#include <mkldnn_types.h>
#include <chrono>

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace mkldnn::impl;
using namespace mkldnn::impl::cpu;
using namespace mkldnn::impl::utils;

struct jitGatherElConfT {
    int32_t strideAxDst;
    int32_t dstAxDim;
    uint32_t dataSize;
};

struct jitArgsGatherEl {
    const void* src;
    void* dst;
    const int* indices;
    const int* strideAxSrc;
    const int* dstAxIdx;
    const int* strideAx1Diff;
    const int* dstShift0;
    const int* incVec;
    const int* one;
    int axStrideIt;
    size_t workAmount;
//    int* retVal;
};

struct jitUniGatherElKernel {
    void (*ker_)(const jitArgsGatherEl *);
    void operator()(const jitArgsGatherEl *args) { assert(ker_); ker_(args); }
    explicit jitUniGatherElKernel(jitGatherElConfT jpp) : ker_(nullptr), jpp(jpp) {}
    virtual ~jitUniGatherElKernel() {}

    jitGatherElConfT jpp;
};

#define GET_OFF(field) offsetof(jitArgsGatherEl, field)

template <cpu::cpu_isa_t isa>
struct jitUniGatherElKernel_32 : public jitUniGatherElKernel, public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jitUniGatherElKernel_32)

    explicit jitUniGatherElKernel_32(jitGatherElConfT jpp) : jitUniGatherElKernel(jpp), jit_generator() {
        this->preamble();

        mov(regSrc, ptr[regParams + GET_OFF(src)]);
        mov(regDst, ptr[regParams + GET_OFF(dst)]);
        mov(regIndices, ptr[regParams + GET_OFF(indices)]);
        mov(regAxStrideIt, ptr[regParams + GET_OFF(axStrideIt)]);
        mov(regWorkAmount, ptr[regParams + GET_OFF(workAmount)]);

        mov(regTmp, ptr[regParams + GET_OFF(dstAxIdx)]);
        mov(regDstAxIdx, ptr[regTmp]);
        uni_vpbroadcastd(vmmDstAxIdx, ptr[regTmp]);
        mov(regTmp, ptr[regParams + GET_OFF(strideAx1Diff)]);
        uni_vpbroadcastd(vmmStrideAx1Diff, ptr[regTmp]);
        mov(regTmp, ptr[regParams + GET_OFF(dstShift0)]);
        uni_vpbroadcastd(vmmDstShift0, ptr[regTmp]);
        mov(regTmp, ptr[regParams + GET_OFF(strideAxSrc)]);
        mov(regRetVal, ptr[regTmp]);
        uni_vpbroadcastd(vmmStrideAxSrc, ptr[regTmp]);
        mov(regTmp, ptr[regParams + GET_OFF(one)]);
        uni_vpbroadcastd(vmmOnes, ptr[regTmp]);
        mov(regTmp, ptr[regParams + GET_OFF(incVec)]);
        uni_vmovups(vmmIncVec, ptr[regTmp]);

        Xbyak::Label oLabel, endLabel, tailLabel, tailLoopLabel;

        const size_t elPerVec = vlen / jpp.dataSize;
        const int iterPerStride = jpp.strideAxDst / elPerVec;

        uni_vxorps(vmmZero, vmmZero, vmmZero);
//        uni_vpbroadcastd(vmmDstAxIdx, xmmZero);
//        uni_vpbroadcastd(vmmDstShift0, xmmZero);

        L(oLabel);
        {
            Xbyak::Label strideFinish;

            add(regAxStrideIt, elPerVec);
            cmp(regAxStrideIt, jpp.strideAxDst);
            jg(tailLabel, T_NEAR);

            for (int i = 0; i < iterPerStride; i++) {
                cmp(regWorkAmount, elPerVec);
                jl(tailLoopLabel, T_NEAR);

                uni_vmovups(vmmIndicies, ptr[regIndices]);
                uni_vpsubd(vmmSrcIdx, vmmIndicies, vmmDstAxIdx);
                uni_vpmulld(vmmSrcIdx, vmmSrcIdx, vmmStrideAxSrc);
                uni_vpaddd(vmmSrcIdx, vmmSrcIdx, vmmDstShift0);
                uni_vpaddd(vmmSrcIdx, vmmSrcIdx, vmmIncVec);

                uni_vpcmpeqd(vmmOnesBit, vmmOnesBit, vmmOnesBit);
                vpgatherdd(vmmDst, ptr[regSrc + vmmSrcIdx], vmmOnesBit);
                uni_vmovups(ptr[regDst], vmmDst);

                add(regSrc, vlen);
                add(regDst, vlen);
                add(regIndices, vlen);
                sub(regWorkAmount, elPerVec);

                add(regAxStrideIt, elPerVec);
                cmp(regAxStrideIt, jpp.strideAxDst);
                jg(tailLabel, T_NEAR);
            }
            L(tailLabel);
            sub(regAxStrideIt, elPerVec);
            L(tailLoopLabel);
            {
                cmp(regWorkAmount, 0);
                je(endLabel, T_NEAR);
                cmp(regAxStrideIt, jpp.strideAxDst);
                je(strideFinish, T_NEAR);

//                mov(regTmp, ptr[regIndices]);
//                sub(regTmp, regAxStrideIt);
//                mov(rax, regTmp);
//                mul(regRetVal);
//                vextractps(regTmp32, Xbyak::Xmm(vmmDstShift0.getIdx()), 0);
//                add(regTmp, eax);
//                mov(rdx, ptr[regSrc]);
//                mov(ptr[regDst], rdx);

                add(regSrc, jpp.dataSize);
                add(regDst, jpp.dataSize);
                add(regIndices, sizeof(int));
                sub(regWorkAmount, 1);
                add(regAxStrideIt, 1);
                jmp(tailLoopLabel, T_NEAR);
            }

            L(strideFinish);
            mov(regAxStrideIt, 0);
            uni_vpaddd(vmmDstAxIdx, vmmDstAxIdx, vmmOnes);
            inc(regDstAxIdx);
            cmp(regDstAxIdx, jpp.dstAxDim);
            jl(oLabel, T_NEAR);
            mov(regDstAxIdx, 0);
            uni_movaps(vmmDstAxIdx, vmmZero);
            uni_vpaddd(vmmDstShift0, vmmDstShift0, vmmStrideAx1Diff);

            jmp(oLabel, T_NEAR);
        }
        L(endLabel);

        this->postamble();

        ker_ = (decltype(ker_))this->getCode();
    }

    inline void uni_insertps(const Xbyak::Xmm& x1, const Xbyak::Xmm& x2, const Xbyak::Operand& op, uint8_t imm) {
        if (isa == cpu::avx512_common || isa == cpu::avx2) {
            vinsertps(x1, x2, op, imm);
        } else {
            insertps(x1, op, imm);
        }
    }

    inline void uni_movaps(const Xbyak::Xmm& x1, const Xbyak::Xmm& x2) {
        if (isa == cpu::avx512_common || isa == cpu::avx2) {
            vmovaps(x1, x2);
        } else {
            movaps(x1, x2);
        }
    }

    inline void uni_vpmuldq(const Xbyak::Xmm& x1, const Xbyak::Xmm& x2, const Xbyak::Operand& x3) {
        if (isa == cpu::avx512_common || isa == cpu::avx2) {
            vpmuldq(x1, x2, x3);
        } else {
            pmuldq(x1, x3);
        }
    }

//    void fillDstAxIdx(const Xbyak::Xmm& idx, const Xbyak::Xmm& shift0) {
//        for (int i = 0; i < 4; i++) {
//            Xbyak::Label insertLabel, incLabel;
//
//            cmp(regAxStrideIt, jpp.strideAxDst);
//            jl(insertLabel, T_NEAR);
//            mov(regAxStrideIt, 0);
//            inc(regDstAxIdx);
//
//            cmp(regDstAxIdx, jpp.dstAxDim);
//            jl(incLabel, T_NEAR);
//            mov(regDstAxIdx, 0);
//            uni_movaps(xmmDstAxIdxAux, xmmZero);
//            uni_vpaddd(xmmDstShift0Aux, xmmDstShift0Aux, xmmStrideAx1Diff);
//            jmp(insertLabel, T_NEAR);
//
//            L(incLabel);
//            uni_vpaddd(xmmDstAxIdxAux, xmmDstAxIdxAux, xmmOnes);
//
//            L(insertLabel);
//            uni_insertps(idx, xmmDstAxIdxAux, xmmDstAxIdxAux, i << 6);
//            uni_insertps(shift0, xmmDstShift0Aux, xmmDstShift0Aux, i << 6);
//            inc(regAxStrideIt);
//        }
//    }
//
//    void fillDstAxIdx(const Xbyak::Ymm& idx, const Xbyak::Ymm& shift0) {
//        for (int i = 0; i < 2; i++) {
//            fillDstAxIdx(xmmDstAxIdx, xmmDstShift0);
//            vinsertf128(idx, idx, xmmDstAxIdx, i);
//            vinsertf128(shift0, shift0, xmmDstShift0, i);
//        }
//    }
//
//    void fillDstAxIdx(const Xbyak::Zmm& idx, const Xbyak::Zmm& shift0) {
//        for (int i = 0; i < 2; i++) {
//            fillDstAxIdx(ymmDstAxIdx, ymmDstShift0);
//            vinsertf32x8(idx, idx, ymmDstAxIdx, i);
//            vinsertf32x8(shift0, shift0, ymmDstShift0, i);
//        }
//    }

private:
    using Vmm = typename conditional3<isa == cpu::sse42, Xbyak::Xmm, isa == cpu::avx2, Xbyak::Ymm, Xbyak::Zmm>::type;
    uint32_t vlen = cpu_isa_traits<isa>::vlen;

    Xbyak::Reg64 regSrc = r8;
    Xbyak::Reg64 regDst = r9;
    Xbyak::Reg64 regIndices = r10;
    Xbyak::Reg64 regAxStrideIt = r11;
    Xbyak::Reg64 regDstAxIdx = r12;
    Xbyak::Reg64 regRetVal = r13;
    Xbyak::Reg64 regWorkAmount = r14;
    Xbyak::Reg64 regTmp = r15;
    Xbyak::Reg32 regTmp32 = r15d;

    Xbyak::Reg64 regParams = abi_param1;

//    Xbyak::Xmm xmmDstAxIdx = Xbyak::Xmm(0);
//    Xbyak::Xmm xmmDstShift0 = Xbyak::Xmm(1);
//    Xbyak::Ymm ymmDstAxIdx = Xbyak::Ymm(0);
//    Xbyak::Ymm ymmDstShift0 = Xbyak::Ymm(1);
    Vmm vmmDstAxIdx = Vmm(8);
    Vmm vmmDstShift0 = Vmm(9);

//    Xbyak::Xmm xmmDstAxIdxAux = Xbyak::Xmm(2);
//    Xbyak::Xmm xmmDstShift0Aux = Xbyak::Xmm(3);
//    Xbyak::Xmm xmmStrideAx1Diff = Xbyak::Xmm(4);
//    Xbyak::Xmm xmmZero = Xbyak::Xmm(5);
//    Xbyak::Xmm xmmOnes = Xbyak::Xmm(6);

    Vmm vmmZero = Vmm(5);
    Vmm vmmOnes = Vmm(6);
    Vmm vmmIncVec = Vmm(7);
    Vmm vmmIndicies = Vmm(10);
    Vmm vmmSrcIdx = Vmm(11);
    Vmm vmmStrideAx1Diff = Vmm(12);
    Vmm vmmStrideAxSrc = Vmm(13);
    Vmm vmmOnesBit = Vmm(14);
    Vmm vmmDst = Vmm(15);
};

class GatherElementsImpl: public ExtLayerBase {
public:
    explicit GatherElementsImpl(const CNNLayer* layer) : strideAx1Diff_(0) {
        errorPrefix_ = std::string("Layer GatherElements with name '") + layer->name + "'";

        if (layer->insData.size() != 2 || layer->outData.size() != 1)
            THROW_IE_EXCEPTION << errorPrefix_ << " has invalid number of input/output edges.";

        auto inputData = layer->insData[dataIndex_].lock();
        auto indices = layer->insData[indicesIndex_].lock();
        if (!inputData || !indices)
            THROW_IE_EXCEPTION << errorPrefix_ << " has nullable inputs.";

        const auto& dataDims = inputData->getTensorDesc().getDims();
        const auto& indicesDims = indices->getTensorDesc().getDims();
        if (dataDims.size() != indicesDims.size())
            THROW_IE_EXCEPTION << errorPrefix_ << " has invalid input shapes. Inputs 'Data' and 'Indices' must have equal ranks.";

        Precision dataPrecision = inputData->getTensorDesc().getPrecision();
        if (dataPrecision.size() != sizeof(PrecisionTrait<Precision::I32>::value_type) &&
                dataPrecision.size() != sizeof(PrecisionTrait<Precision::I16>::value_type) &&
                dataPrecision.size() != sizeof(PrecisionTrait<Precision::I8>::value_type)) {
            THROW_IE_EXCEPTION << errorPrefix_ << " has unsupported 'inputData' input precision: " << dataPrecision;
        }

        Precision indicesPrecision = indices->getTensorDesc().getPrecision();
        if (indicesPrecision != Precision::I32) {
            THROW_IE_EXCEPTION << errorPrefix_ << " has unsupported 'indices' input precision: " << indicesPrecision;
        }

        auto& outputData = layer->outData[0];
        auto outputPrecision = outputData->getPrecision();
        if (!mayiuse(avx512_core)) {
            if (outputPrecision == Precision::BF16)
                outputPrecision = Precision::FP32;
        }

        dataTypeSize_ = dataPrecision.size();

        axis_ = layer->GetParamAsInt("axis", 0);
        if (axis_ >= dataDims.size())
            THROW_IE_EXCEPTION << errorPrefix_ << " has invalid axis attribute: " << axis_;
        if (axis_ < 0)
            axis_ += dataDims.size();

        strideAxDst_ = outputData->getTensorDesc().getBlockingDesc().getStrides()[axis_];
        dstAxDim_ = outputData->getTensorDesc().getDims()[axis_];
        if (axis_ > 0) {
            strideAx1Diff_ = inputData->getTensorDesc().getBlockingDesc().getStrides()[axis_ - 1] -
                    outputData->getTensorDesc().getBlockingDesc().getStrides()[axis_ - 1];
        }

        // Gather instruction is applicable just for 32 and 64 bit inputData and is not supported by SSE.
//        if (dataPrecision.size() == sizeof(PrecisionTrait<Precision::I32>::value_type) &&
//                (mayiuse(cpu::avx512_common) || mayiuse(cpu::avx2))) {
//            jitGatherElConfT jpp;
//            jpp.strideAxDst = strideAxDst_;
//            jpp.dstAxDim = dstAxDim_;
//            jpp.dataSize = dataTypeSize_;
//            if (mayiuse(cpu::avx512_common)) {
//                kernel32_.reset(new jitUniGatherElKernel_32<cpu::avx512_common>(jpp));
//                incVec_.resize(16, 0);
//            } else if (mayiuse(cpu::avx2)) {
//                kernel32_.reset(new jitUniGatherElKernel_32<cpu::avx2>(jpp));
//                incVec_.resize(8, 0);
//            }
//            for (int j = 1; j < incVec_.size(); j++)
//                incVec_[j] = incVec_[j - 1] + dataTypeSize_;
//        }

        LayerConfig config;
        DataConfig dataConfig, indicesConfig, outConfig;
        dataConfig.desc = TensorDesc(dataPrecision, dataDims,
            inputData->getTensorDesc().getLayoutByDims(dataDims));
        config.inConfs.push_back(dataConfig);
        indicesConfig.desc = TensorDesc(Precision::I32, indicesDims,
            indices->getTensorDesc().getLayoutByDims(indicesDims));
        config.inConfs.push_back(indicesConfig);

        const auto& outDims = layer->outData[0]->getTensorDesc().getDims();
        outConfig.desc = TensorDesc(outputPrecision, outDims,
                layer->outData[0]->getTensorDesc().getLayoutByDims(outDims));
        config.outConfs.push_back(outConfig);

        config.dynBatchSupport = false;

        confs.push_back(config);
    }

    StatusCode execute(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs, ResponseDesc *resp) noexcept override {
        if (kernel32_) {
            return vectorizedExecution(inputs, outputs, resp);
        } else {
            switch (dataTypeSize_) {
                case sizeof(PrecisionTrait<Precision::I32>::value_type):
                    return directExecution<PrecisionTrait<Precision::I32>::value_type>(inputs, outputs, resp);
                case sizeof(PrecisionTrait<Precision::I16>::value_type):
                    return directExecution<PrecisionTrait<Precision::I16>::value_type>(inputs, outputs, resp);
                case sizeof(PrecisionTrait<Precision::I8>::value_type):
                    return directExecution<PrecisionTrait<Precision::I8>::value_type>(inputs, outputs, resp);
                default:
                    std::string errMsg = errorPrefix_ + " has inputData input with unsupported precision: " +
                        inputs[dataIndex_]->getTensorDesc().getPrecision().name();
                    errMsg.copy(resp->msg, sizeof(resp->msg) - 1);
                    return GENERAL_ERROR;
            }
        }
    }

protected:
    template <typename dataType>
    StatusCode directExecution(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs, ResponseDesc *resp) noexcept {
//static double du1 = 0.0;
//static int c1 = 0;
//auto start = std::chrono::steady_clock::now();
        const dataType* srcData = inputs[dataIndex_]->cbuffer().as<const dataType*>() +
            inputs[dataIndex_]->getTensorDesc().getBlockingDesc().getOffsetPadding();
        const int* indices = inputs[indicesIndex_]->cbuffer().as<const int*>() +
            inputs[indicesIndex_]->getTensorDesc().getBlockingDesc().getOffsetPadding();
        dataType* dstData = outputs[0]->buffer().as<dataType*>() +
            outputs[0]->getTensorDesc().getBlockingDesc().getOffsetPadding();

//    std::string probsStr = "srcData:\n";
//for (int i = 0; i < inputs[dataIndex_]->size(); i++) {
//    if (i % 10 == 0)
//        probsStr += "(" + std::to_string(i) + "); ";
//    probsStr += std::to_string(srcData[i]) + "; ";
//}
//probsStr += "\nIndices:\n";
//for (int i = 0; i < inputs[indicesIndex_]->size(); i++) {
//    if (i % 10 == 0)
//        probsStr += "(" + std::to_string(i) + "); ";
//    probsStr += std::to_string(indices[i]) + "; ";
//}
//printf("%s\n", probsStr.c_str());

        const size_t outSize = outputs[0]->size();
//        {
////probsStr += "\nSeq Out:\n";
//            size_t axStrideIt = 0lu;
//            size_t dstAxIdx = 0lu;
//            int dstShift0 = 0;
//            for (size_t o = 0; o < outSize; o++, axStrideIt++) {
//                if (axStrideIt == strideAxDst_) {
//                    axStrideIt = 0lu;
//                    dstAxIdx++;
//                    if (dstAxIdx == dstAxDim_) {
//                        dstAxIdx = 0lu;
//                        dstShift0 += strideAx1Diff_;
//                    }
//                }
////if (o % 10 == 0)
////    probsStr += "(" + std::to_string(o) + "); ";
////probsStr += std::to_string(srcData[o + dstShift0 + (indices[o] - dstAxIdx) * strideAxDst_]) + "; ";
//                dstData[o] = srcData[o + dstShift0 + (indices[o] - dstAxIdx) * strideAxDst_];
//            }
//        }

        auto threadBody = [&](const int ithr, const int nthr) {
            size_t start(0lu), end(0lu);
            splitter(outSize, nthr, ithr, start, end);
            if (start >= end)
                return;

            int axStrideIt = start % strideAxDst_;
            int dstAxIdx = (start / strideAxDst_) % dstAxDim_;
            int dstShift0 = (start / strideAxDst_ / dstAxDim_) * strideAx1Diff_;
////printf("ithr: %d; start: %lu; end: %lu; axStrideIt: %d\n", ithr, start, end, axStrideIt);
//int retVal = 1;
            for (size_t o = start; o < end; o++, axStrideIt++) {
                if (axStrideIt == strideAxDst_) {
                    axStrideIt = 0lu;
                    dstAxIdx++;
                    if (dstAxIdx == dstAxDim_) {
                        dstAxIdx = 0lu;
                        dstShift0 += strideAx1Diff_;
                    }
                }
                dstData[o] = srcData[o + dstShift0 + (indices[o] - dstAxIdx) * strideAxDst_];
            }
        };
        parallel_nt(0, threadBody);

//probsStr += "\nOutput:\n";
//for (int i = 0; i < outputs[0]->size(); i++) {
//    if (i % 10 ==0)
//        probsStr += "(" + std::to_string(i) + "); ";
//    probsStr += std::to_string(dstData[i]) + "; ";
//}
//printf("%s\n", probsStr.c_str());
//auto end = std::chrono::steady_clock::now();
//c1++;
//du1 += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
//if (c1 % 100 == 0) {
//    printf("DU1: %f\n", du1 / c1);
//}
        return OK;
    }

    StatusCode vectorizedExecution(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs, ResponseDesc *resp) noexcept {
//static double du1 = 0.0;
//static int c1 = 0;
//auto start = std::chrono::steady_clock::now();
        const int* srcData = inputs[dataIndex_]->cbuffer().as<const int*>() +
            inputs[dataIndex_]->getTensorDesc().getBlockingDesc().getOffsetPadding();
        const int* indices = inputs[indicesIndex_]->cbuffer().as<const int*>() +
            inputs[indicesIndex_]->getTensorDesc().getBlockingDesc().getOffsetPadding();
        int* dstData = outputs[0]->buffer().as<int*>() +
            outputs[0]->getTensorDesc().getBlockingDesc().getOffsetPadding();

        const size_t outSize = outputs[0]->size();

//    std::string probsStr = "srcData:\n";
//for (int i = 0; i < inputs[dataIndex_]->size(); i++) {
//    if (i % 10 == 0)
//        probsStr += "(" + std::to_string(i) + "); ";
//    probsStr += std::to_string(srcData[i]) + "; ";
//}
//probsStr += "\nIndices:\n";
//for (int i = 0; i < inputs[indicesIndex_]->size(); i++) {
//    if (i % 10 == 0)
//        probsStr += "(" + std::to_string(i) + "); ";
//    probsStr += std::to_string(indices[i]) + "; ";
//}
//printf("%s\n", probsStr.c_str());
//
//        {
//probsStr += "\nSeq Out:\n";
//            size_t axStrideIt = 0lu;
//            size_t dstAxIdx = 0lu;
//            int dstShift0 = 0;
//            for (size_t o = 0; o < outSize; o++, axStrideIt++) {
//                if (axStrideIt == strideAxDst_) {
//                    axStrideIt = 0lu;
//                    dstAxIdx++;
//                    if (dstAxIdx == dstAxDim_) {
//                        dstAxIdx = 0lu;
//                        dstShift0 += strideAx1Diff_;
//                    }
//                }
//if (o % 10 == 0)
//    probsStr += "(" + std::to_string(o) + "); ";
//probsStr += std::to_string(srcData[o + dstShift0 + (indices[o] - dstAxIdx) * strideAxDst_]) + "; ";
////                dstData[o] = srcData[o + dstShift0 + (indices[o] - dstAxIdx) * strideAxDst_];
//            }
//        }

        auto threadBody = [&](const int ithr, const int nthr) {
            size_t start(0lu), end(0lu);
            splitter(outSize, nthr, ithr, start, end);
            if (start >= end)
                return;

            int axStrideIt = start % strideAxDst_;
            int dstAxIdx = (start / strideAxDst_) % dstAxDim_;
            int dstShift0 = (start / strideAxDst_ / dstAxDim_) * strideAx1Diff_;
//printf("ithr: %d; start: %lu; end: %lu; axStrideIt: %d; dstAxIdx: %d; dstShift0: %d\n", ithr, start, end, axStrideIt, dstAxIdx, dstShift0);
//int retVal = 1;

            const int strideAxDst = strideAxDst_ * dataTypeSize_;
            const int dstShift0B = dstShift0 * dataTypeSize_;
            const int strideAx1DiffB = strideAx1Diff_ * dataTypeSize_;
            const int one = 1;
            auto arg = jitArgsGatherEl();
            arg.src = srcData + start;
            arg.dst = dstData + start;
            arg.indices = indices + start;
            arg.axStrideIt = axStrideIt;
            arg.strideAxSrc = &strideAxDst;
            arg.dstAxIdx = &dstAxIdx;
            arg.strideAx1Diff = &strideAx1DiffB;
            arg.dstShift0 = &dstShift0B;
            arg.incVec = incVec_.data();
            arg.one = &one;
            arg.workAmount = end - start;
//                arg.retVal = &retVal;
            (*kernel32_)(&arg);
//printf("ithr: %d finished\n", ithr);
//printf("retVal: %d\n", retVal);
        };
        parallel_nt(0, threadBody);

//probsStr += "\nOutput:\n";
//for (int i = 0; i < outputs[0]->size(); i++) {
//    if (i % 10 ==0)
//        probsStr += "(" + std::to_string(i) + "); ";
//    probsStr += std::to_string(dstData[i]) + "; ";
//}
//printf("%s\n", probsStr.c_str());
//auto end = std::chrono::steady_clock::now();
//c1++;
//du1 += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
//if (c1 % 100 == 0) {
//    printf("DU1: %f\n", du1 / c1);
//}
        return OK;
    }

    const size_t dataIndex_ = 0;
    const size_t indicesIndex_ = 1;

    size_t axis_;
    size_t dataTypeSize_;
    int strideAxDst_;
    int dstAxDim_;
    int strideAx1Diff_;
    std::vector<int> incVec_;
    std::shared_ptr<jitUniGatherElKernel> kernel32_;
    std::string errorPrefix_;
};

REG_FACTORY_FOR(GatherElementsImpl, GatherElements);
}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
