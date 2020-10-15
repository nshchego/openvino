// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "base.hpp"

#include <string>
#include <vector>
#include "ie_parallel.hpp"

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

class GatherNDImpl: public ExtLayerBase {
public:
    explicit GatherNDImpl(const CNNLayer* layer) {
        _errorPrefix = std::string("Layer GatherND with name '") + layer->name + "'";

        if (layer->insData.size() != 2 || layer->outData.empty())
            THROW_IE_EXCEPTION << _errorPrefix << " has invalid number of input/output edges.";

        auto data = layer->insData[_dataIndex].lock();
        auto indices = layer->insData[_indicesIndex].lock();
        if (!data || !indices)
            THROW_IE_EXCEPTION << _errorPrefix << " has nullable inputs.";
        Precision dataPrecision = data->getTensorDesc().getPrecision();
        if (dataPrecision == Precision::BF16)
            dataPrecision = Precision::FP32;
        if (dataPrecision != Precision::FP32 && dataPrecision != Precision::I32 &&
                dataPrecision != Precision::U8 && dataPrecision != Precision::I8)
            THROW_IE_EXCEPTION << _errorPrefix << " has unsupported data precision or input data and output have different precisions."
                "\nData precision: " << dataPrecision.name() << "; Out precision: " << layer->outData[0]->getTensorDesc().getPrecision().name();

        Precision indicesPrecision = indices->getTensorDesc().getPrecision();
        if (indicesPrecision != Precision::I32)
            THROW_IE_EXCEPTION << _errorPrefix << " has unsupported indices precision: " << indicesPrecision.name();

        const auto& dataDims = data->getTensorDesc().getDims();
        const auto& indicesDims = indices->getTensorDesc().getDims();

        _batchDims = layer->GetParamAsInt("batch_dims", 0);
        if (_batchDims >= dataDims.size())
            THROW_IE_EXCEPTION << _errorPrefix << " has invalid batch_dims attribute: " << std::to_string(_batchDims);

        _batchNum = 1lu;
        for (size_t i = 0; i < _batchDims; i++) {
            _batchNum *= indicesDims[i];
        }

        _sliceRank = indicesDims[indicesDims.size() - 1];
        _dataRank = dataDims.size() - _batchDims;
        if (_sliceRank > _dataRank)
            THROW_IE_EXCEPTION << _errorPrefix << " has invalid inputs shapes.";

        _blockSize = 1lu;
        for (size_t i = _sliceRank + _batchDims; i < dataDims.size(); i++) {
            _blockSize *= dataDims[i];
        }
        _batchStep = 1;
        for (size_t i = _batchDims; i < dataDims.size(); i++) {
            _batchStep *= dataDims[i];
        }
        _srcMultipliers.resize(dataDims.size(), 1);
        for (int i = dataDims.size() - 2; i >= 0; i--)
            _srcMultipliers[i] = _srcMultipliers[i + 1] * dataDims[i + 1];

        LayerConfig config;
        DataConfig dataConfig, indicesConfig, outConfig;
        dataConfig.desc = TensorDesc(dataPrecision, dataDims,
            data->getTensorDesc().getLayoutByDims(dataDims));
        config.inConfs.push_back(dataConfig);
        indicesConfig.desc = TensorDesc(indicesPrecision, indicesDims,
            indices->getTensorDesc().getLayoutByDims(indicesDims));
        config.inConfs.push_back(indicesConfig);

        const auto& outDims = layer->outData[0]->getTensorDesc().getDims();
        outConfig.desc = TensorDesc(dataPrecision, outDims,
                layer->outData[0]->getTensorDesc().getLayoutByDims(outDims));
        config.outConfs.push_back(outConfig);
        config.dynBatchSupport = false;

        confs.push_back(config);
    }

    StatusCode execute(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs, ResponseDesc *resp) noexcept override {
        switch (inputs[_dataIndex]->getTensorDesc().getPrecision().size()) {
            case sizeof(PrecisionTrait<Precision::I32>::value_type):
                process<PrecisionTrait<Precision::I32>::value_type>(inputs, outputs, resp);
                break;
            case sizeof(PrecisionTrait<Precision::I8>::value_type):
                process<PrecisionTrait<Precision::I8>::value_type>(inputs, outputs, resp);
                break;
            default:
                std::string errMsg = _errorPrefix + " has data input with unsupported precision: " +
                    inputs[_dataIndex]->getTensorDesc().getPrecision().name();
                errMsg.copy(resp->msg, sizeof(resp->msg) - 1);
                return GENERAL_ERROR;
        }

        return OK;
    }

protected:
    template <typename dataType>
    void process(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs, ResponseDesc *resp) noexcept {
        const dataType* data = inputs[_dataIndex]->cbuffer().as<const dataType*>() +
            inputs[_dataIndex]->getTensorDesc().getBlockingDesc().getOffsetPadding();
        const int* indices = inputs[_indicesIndex]->cbuffer().as<const int*>() +
            inputs[_indicesIndex]->getTensorDesc().getBlockingDesc().getOffsetPadding();
        dataType* dstData = outputs[0]->buffer().as<dataType*>() +
            outputs[0]->getTensorDesc().getBlockingDesc().getOffsetPadding();

        const auto& dataDims = inputs[_dataIndex]->getTensorDesc().getDims();

        const dataType* shiftedData = data;
        const int* shiftedIndices = indices;
        dataType* shiftedDstData = dstData;
        const size_t* srcMultipliers = _srcMultipliers.data() + _batchDims;

        if (_blockSize > 1) {
            // blocked
            const size_t dataStep = sizeof(dataType) * _blockSize;
            const size_t cycles = outputs[0]->byteSize() / (dataStep * _batchNum);
            const size_t CS = cycles * _sliceRank;
            const size_t CB = cycles * _blockSize;
            const size_t workAmount = _batchNum * cycles;

            auto threadBody = [&](const int ithr, const int nthr) {
                size_t start(0lu), end(0lu);
                splitter(workAmount, nthr, ithr, start, end);
                if (start >= end)
                    return;
                size_t bStart = start / cycles;
                size_t cStart = start % cycles;
                size_t workCounter = start;

                const dataType* shiftedData = data + bStart * _batchStep;
                const int* shiftedIndices = indices + bStart * CS + cStart * _sliceRank;
                dataType* shiftedDstData = dstData + bStart * CB + cStart * _blockSize;

                for (size_t b = bStart; b < _batchNum; b++) {
                    for (size_t j = cStart; j < cycles; j++) {
                        size_t dataIdx = 0lu;
                        for (size_t i = 0; i < _sliceRank ; i++)
                            dataIdx += srcMultipliers[i] * shiftedIndices[i];
                        memcpy(shiftedDstData, &(shiftedData[dataIdx]), dataStep);
                        shiftedDstData += _blockSize;
                        shiftedIndices += _sliceRank;
                        if (++workCounter == end) {
                            return;
                        }
                    }
                    shiftedData += _batchStep;
                }
            };

            parallel_nt(0, threadBody);
        } else {
            // elementwise
            const size_t cycles = outputs[0]->byteSize() / (sizeof(dataType) * _batchNum);
            const size_t CS = cycles * _sliceRank;
            const size_t CB = cycles * _blockSize;
            const size_t workAmount = _batchNum * cycles;

            auto threadBody = [&](const int ithr, const int nthr) {
                size_t start(0lu), end(0lu);
                splitter(workAmount, nthr, ithr, start, end);
                if (start >= end)
                    return;
                size_t bStart = start / cycles;
                size_t cStart = start % cycles;
                size_t workCounter = start;

                const dataType* shiftedData = data + bStart * _batchStep;
                const int* shiftedIndices = indices + bStart * CS + cStart * _sliceRank;
                dataType* shiftedDstData = dstData + bStart * CB + cStart * _blockSize;

                for (size_t b = bStart; b < _batchNum; b++) {
                    for (size_t j = cStart; j < cycles; j++) {
                        size_t dataIdx = 0lu;
                        for (size_t i = 0lu; i < _sliceRank; i++)
                            dataIdx += srcMultipliers[i] * shiftedIndices[i];
                        shiftedDstData[0] = shiftedData[dataIdx];
                        shiftedDstData++;
                        shiftedIndices += _sliceRank;
                        if (++workCounter == end) {
                            return;
                        }
                    }
                    shiftedData += _batchStep;
                }
            };

            parallel_nt(0, threadBody);
        }
    }

    size_t _dataRank;
    size_t _sliceRank;
    size_t _blockSize;
    size_t _batchDims;
    size_t _batchNum;
    size_t _batchStep;
    const size_t _dataIndex = 0;
    const size_t _indicesIndex = 1;
    std::vector<size_t> _srcMultipliers;
    std::string _errorPrefix;
};


REG_FACTORY_FOR(GatherNDImpl, GatherND);
}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
