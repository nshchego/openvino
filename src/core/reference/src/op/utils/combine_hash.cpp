// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// The CRC computation is used for x86
// The calculations were taken from the article "Fast CRC Computation for Generic Polynomials Using PCLMULQDQ Instruction - Intel (December, 2009)"

#include "openvino/core/visibility.hpp"
#include "openvino/reference/utils/combine_hash.hpp"

#if defined(OPENVINO_ARCH_X86) || defined(OPENVINO_ARCH_X86_64)
#    include "openvino/reference/utils/registers_pool.hpp"
#endif // OPENVINO_ARCH_X86 || OPENVINO_ARCH_X86_64

#include <cstring>
#include <iterator>
#include <iostream>
#include <type_traits>
#include <typeinfo>
#include <inttypes.h>

namespace ov {
namespace runtime {

#if defined(OPENVINO_ARCH_X86) || defined(OPENVINO_ARCH_X86_64)
namespace jit {

#define GET_OFF(field) offsetof(CombineHashCallArgs, field)
#define getReg64() RegistersPool::Reg<Xbyak::Reg64>(registersPool)
#define getVmm()   RegistersPool::Reg<Vmm>(registersPool)
#define getXmm()   RegistersPool::Reg<Xbyak::Xmm>(registersPool)

struct CombineHashCompileParams {
};

struct CombineHashCallArgs {
    const void* src_ptr;
    void* dst_ptr;
    uint64_t work_amount = 0lu;
};

typedef void (*fn_t)(const CombineHashCallArgs*);

template <cpu_isa_t isa>
class CombineHash : public Generator {
public:
    explicit CombineHash(const CombineHashCompileParams& jcp) :
            m_jcp(jcp) {
        if (isa == avx512_core) {
            vlen = zmm_len;
        } else if (isa == avx2) {
            vlen = ymm_len;
        } else {
            OPENVINO_THROW("Unsupported isa: ", isa);
        }
        if (!mayiuse(cpu_isa_t::pclmulqdq)) {
            OPENVINO_THROW("The current CPU does not support pclmulqdq instruction, which is required for the CRC algorithm.");
        }

        generate();
    }

    void generate() {
        this->preamble();
        registersPool = RegistersPool::create(isa, {rax, rcx, rsp, rdi, k0});

        r64_src = getReg64();
        r64_dst = getReg64();
        r64_work_amount = getReg64();

        mov(r64_src, ptr[r64_params + GET_OFF(src_ptr)]);
        mov(r64_dst, ptr[r64_params + GET_OFF(dst_ptr)]);
        mov(r64_work_amount, ptr[r64_params + GET_OFF(work_amount)]);

        initVectors();
        if (mayiuse(cpu_isa_t::vpclmulqdq)) {
            bulkFold();
        } else {
            bulkFold_128();
        }
        restFold();
        tailFold();

        vpextrq(ptr[r64_dst], Xbyak::Xmm(v_dst.getIdx()), 0x0);

        registersPool.reset();
        this->postamble();
    }

    static fn_t get() {
        static const CombineHashCompileParams params;
        static CombineHash<isa> kernel(params);

        return (fn_t)kernel.getCode();
    }

private:
    static constexpr uint64_t CHUNK_SIZE = 32;
    // static constexpr uint64_t P64 = 0x42F0E1EBA9EA3693;
    // static const uint64_t K12;
    static const uint64_t INIT_CRC;
    static const uint64_t CONST_K[20];

    // using Vmm = typename std::conditional<true, Xbyak::Zmm, Xbyak::Ymm>::type;
    // using Vmm = typename std::conditional<true, Xbyak::Zmm, typename std::conditional<isa == sse41, Xbyak::Xmm, Xbyak::Ymm>::type>::type;
    using Vmm = typename std::conditional<isa == avx512_core, Xbyak::Zmm, typename std::conditional<isa == sse42, Xbyak::Xmm, Xbyak::Ymm>::type>::type;
    size_t vlen = xmm_len;

    CombineHashCompileParams m_jcp;
    RegistersPool::Ptr registersPool;

    RegistersPool::Reg<Xbyak::Reg64> r64_src;
    RegistersPool::Reg<Xbyak::Reg64> r64_dst;
    RegistersPool::Reg<Xbyak::Reg64> r64_work_amount;

    const Xbyak::Reg64 r64_params = abi_param1;

    // Vector registers
    RegistersPool::Reg<Vmm> v_dst;
    RegistersPool::Reg<Vmm> v_k_12;
    RegistersPool::Reg<Vmm> v_k_34;
    RegistersPool::Reg<Vmm> v_k_56;
    RegistersPool::Reg<Vmm> v_k_16_17;
    RegistersPool::Reg<Vmm> v_shuf_mask;

    // static const uint8_t shuf_mask[]

    void initVectors();

    void bulkFold_128();
    // void bulkFold_128() {
    //     Xbyak::Label l_fold_loop, l_end;
    //     cmp(r64_work_amount, xmm_len);
    //     jl(l_end, T_NEAR);

    //     auto xmm_dst = Xbyak::Xmm(v_dst.getIdx());
    //     auto xmm_k_12 = Xbyak::Xmm(v_k_12.getIdx());
    //     auto xmm_shuf_mask = Xbyak::Xmm(v_shuf_mask.getIdx());
    //     auto xmm_src = getXmm();
    //     auto xmm_aux = getXmm();

    //     // prefetchnta(ptr[r64_src]); // TODO compare perf
    //     vmovups(xmm_src, ptr[r64_src]);
    //     vpshufb(xmm_src, xmm_src, xmm_shuf_mask); // Endianness swap
    //     vxorps(xmm_dst, xmm_dst, xmm_src);

    //     // Bulk fold
    //     L(l_fold_loop); {
    //         add(r64_src, xmm_len);
    //         sub(r64_work_amount, xmm_len);
    //         cmp(r64_work_amount, xmm_len);
    //         jl(l_end, T_NEAR);

    //         vmovups(xmm_src, ptr[r64_src]);
    //         vpshufb(xmm_src, xmm_src, xmm_shuf_mask); // Endianness swap

    //         vpclmulqdq(xmm_aux, xmm_dst, xmm_k_12, 0b00000000);
    //         vpclmulqdq(xmm_dst, xmm_dst, xmm_k_12, 0b00010001);
    //         vxorps(xmm_src, xmm_src, xmm_dst);
    //         vxorps(xmm_dst, xmm_src, xmm_aux);

    //         jmp(l_fold_loop, T_NEAR);
    //     }

    //     L(l_end);
    // }

    void bulkFold() {
        Xbyak::Label l_fold_loop, l_fold_128, l_end;
        cmp(r64_work_amount, vlen);
        jl(l_end, T_NEAR);

        auto v_src = getVmm();
        auto v_aux = getVmm();

        vmovups(v_src, ptr[r64_src]);
        vpshufb(v_src, v_src, v_shuf_mask); // Endianness swap
        vxorps(v_dst, v_dst, v_src);

        // Bulk fold
        L(l_fold_loop); {
            add(r64_src, vlen);
            sub(r64_work_amount, vlen);
            cmp(r64_work_amount, vlen);
            jl(l_fold_128, T_NEAR);

            vmovups(v_src, ptr[r64_src]);
            vpshufb(v_src, v_src, v_shuf_mask); // Endianness swap

            vpclmulqdq(v_aux, v_dst, v_k_12, 0b00000000);
            vpclmulqdq(v_dst, v_dst, v_k_12, 0b00010001);
            vxorps(v_src, v_src, v_dst);
            vxorps(v_dst, v_src, v_aux);

            jmp(l_fold_loop, T_NEAR);
        }

        // Fold Vmm to 128
        L(l_fold_128); {
            auto ymm_dst = Xbyak::Ymm(v_dst.getIdx());
            if (isa == avx512_core) {
                // 4 chunks, 4*128 -> 2*128
                auto zmm_dst = Xbyak::Zmm(v_dst.getIdx());
                auto ymm_src = Xbyak::Ymm(v_src.getIdx());
                auto ymm_aux = Xbyak::Ymm(v_aux.getIdx());
                auto ymm_k_12 = Xbyak::Ymm(v_k_12.getIdx());
                vextractf64x4(ymm_src, zmm_dst, 0x1);

                vpclmulqdq(ymm_aux, ymm_dst, ymm_k_12, 0b00000000);
                vpclmulqdq(ymm_dst, ymm_dst, ymm_k_12, 0b00010001);
                vxorps(ymm_src, ymm_src, ymm_dst);
                vxorps(ymm_dst, ymm_src, ymm_aux);
            }

            // 2 chunks, 2*128 -> 128
            auto xmm_dst = Xbyak::Xmm(v_dst.getIdx());
            auto xmm_src = Xbyak::Xmm(v_src.getIdx());
            auto xmm_aux = Xbyak::Xmm(v_aux.getIdx());
            auto xmm_k_12 = Xbyak::Xmm(v_k_12.getIdx());
            vextractf128(xmm_src, ymm_dst, 0x1);

            vpclmulqdq(xmm_aux, xmm_dst, xmm_k_12, 0b00000000);
            vpclmulqdq(xmm_dst, xmm_dst, xmm_k_12, 0b00010001);
            vxorps(xmm_src, xmm_src, xmm_dst);
            vxorps(xmm_dst, xmm_src, xmm_aux);
        }

        L(l_end);
    }

    // CHUNK_SIZE <= Fold < VMM
    void restFold() {
        Xbyak::Label l_end;
        cmp(r64_work_amount, CHUNK_SIZE);
        jl(l_end, T_NEAR);

        L(l_end);
    }

    // Fold < CHUNK_SIZE
    void tailFold() {
        Xbyak::Label l_end;
        cmp(r64_work_amount, 0);
        jle(l_end, T_NEAR);

        L(l_end);
    }
};

template <cpu_isa_t isa>
void CombineHash<isa>::initVectors() {
    v_dst = getVmm();
    // v_k_12 = getVmm();
    v_k_34 = getVmm();
    v_shuf_mask = getVmm();
    auto r64_aux = getReg64();

    // static const uint64_t ff_const = 0xffffffffffffffff;
    // mov(r64_aux, reinterpret_cast<uintptr_t>(&ff_const));
    // vpbroadcastq(v_dst, ptr[r64_aux]);

    // mov(r64_aux, reinterpret_cast<uintptr_t>(&K12));
    // vpbroadcastq(v_k_12, ptr[r64_aux]);
    mov(r64_aux, reinterpret_cast<uintptr_t>(CONST_K + 2));
    vbroadcasti64x2(v_k_16_17, ptr[r64_aux]);

    static const uint8_t shuf_mask[] = { 0b00000000, 0b00000001, 0b00000010, 0b00000011, 0b00000100, 0b00000101, 0b00000110, 0b00000111,
                                         0b00001000, 0b00001001, 0b00001010, 0b00001011, 0b00001100, 0b00001101, 0b00001110, 0b00001111,
                                         0b00000000, 0b00000001, 0b00000010, 0b00000011, 0b00000100, 0b00000101, 0b00000110, 0b00000111,
                                         0b00001000, 0b00001001, 0b00001010, 0b00001011, 0b00001100, 0b00001101, 0b00001110, 0b00001111 };
    mov(r64_aux, reinterpret_cast<uintptr_t>(shuf_mask));
    vmovups(v_shuf_mask, ptr[r64_aux]);
}

template <>
void CombineHash<avx512_core>::bulkFold_128() {
    Xbyak::Label l_end;
    cmp(r64_work_amount, 4 * zmm_len);
    jl(l_end, T_NEAR);

    if (mayiuse(cpu_isa_t::vpclmulqdq)) {
    } else {
        Xbyak::Label l_fold_loop;

        auto r64_aux = getReg64();
        // auto xmm_dst = Xbyak::Xmm(v_dst.getIdx());
        auto xmm_k_16_17 = Xbyak::Xmm(v_k_16_17.getIdx());
        // auto xmm_shuf_mask = Xbyak::Xmm(v_shuf_mask.getIdx());
        auto v_src_0 = getVmm();
        auto v_src_1 = getVmm();
        auto v_dst_0 = getVmm();
        auto v_dst_1 = getVmm();
        auto v_dst_2 = getVmm();
        auto v_dst_3 = getVmm();
        auto v_dst_4 = getVmm();
        auto v_dst_5 = getVmm();
        auto v_dst_6 = getVmm();
        auto v_dst_7 = getVmm();
        auto v_aux_0 = getVmm();
        // auto v_aux_1 = getVmm();
        // auto v_aux_2 = getVmm();
        // auto v_aux_3 = getVmm();

        // auto ymm_src_0 = Xbyak::Ymm(v_src_0.getIdx());
        // auto ymm_src_1 = Xbyak::Ymm(v_src_1.getIdx());
        // auto ymm_dst = Xbyak::Ymm(v_dst.getIdx());
        // auto ymm_aux = Xbyak::Ymm(v_aux.getIdx());

        auto xmm_src_0 = Xbyak::Xmm(v_src_0.getIdx());
        // auto xmm_src_1 = Xbyak::Xmm(v_src_1.getIdx());
        auto xmm_dst_0 = Xbyak::Xmm(v_dst_0.getIdx());
        auto xmm_dst_1 = Xbyak::Xmm(v_dst_1.getIdx());
        auto xmm_dst_2 = Xbyak::Xmm(v_dst_2.getIdx());
        auto xmm_dst_3 = Xbyak::Xmm(v_dst_3.getIdx());
        auto xmm_dst_4 = Xbyak::Xmm(v_dst_4.getIdx());
        auto xmm_dst_5 = Xbyak::Xmm(v_dst_5.getIdx());
        auto xmm_dst_6 = Xbyak::Xmm(v_dst_6.getIdx());
        auto xmm_dst_7 = Xbyak::Xmm(v_dst_7.getIdx());
        auto xmm_aux_0 = Xbyak::Xmm(v_aux_0.getIdx());
        // auto xmm_aux_1 = Xbyak::Xmm(v_aux_1.getIdx());
        // auto xmm_aux_2 = Xbyak::Xmm(v_aux_2.getIdx());
        // auto xmm_aux_3 = Xbyak::Xmm(v_aux_3.getIdx());

        vmovdqu64(v_dst_0, ptr[r64_src]);
        vpshufb(v_dst_0, v_dst_0, v_shuf_mask); // Swap bytes
        vextracti64x2(xmm_dst_1, v_dst_0, 0x1);
        vextracti64x2(xmm_dst_2, v_dst_0, 0x2);
        vextracti64x2(xmm_dst_3, v_dst_0, 0x3);

        add(r64_src, zmm_len);
        vmovdqu64(v_dst_4, ptr[r64_src]);
        vpshufb(v_dst_4, v_dst_4, v_shuf_mask); // Swap bytes
        vextracti64x2(xmm_dst_5, v_dst_0, 0x1);
        vextracti64x2(xmm_dst_6, v_dst_0, 0x2);
        vextracti64x2(xmm_dst_7, v_dst_0, 0x3);

        // prefetchnta(ptr[r64_src + zmm_len]); // TODO compare perf
        // vmovdqu(xmm_dst_0, ptr[r64_src + 0]);
        // vmovdqu(xmm_dst_1, ptr[r64_src + 16]);
        // vmovdqu(xmm_dst_2, ptr[r64_src + 32]);
        // vmovdqu(xmm_dst_3, ptr[r64_src + 48]);
        // vmovdqu(xmm_dst_4, ptr[r64_src + 64]);
        // vmovdqu(xmm_dst_5, ptr[r64_src + 80]);
        // vmovdqu(xmm_dst_6, ptr[r64_src + 96]);
        // vmovdqu(xmm_dst_7, ptr[r64_src + 128]);

        // vpslldq(xmm_dst, xmm_dst, 0x8); TODO modify INIT_CRC instead
        mov(r64_aux, reinterpret_cast<uintptr_t>(&INIT_CRC));
        vpxorq(xmm_dst_0, xmm_dst_0, ptr_b[r64_aux]);

        // Bulk fold
        L(l_fold_loop); {
            add(r64_src, zmm_len);
            sub(r64_work_amount, 2 * zmm_len);
            // cmp(r64_work_amount, 2 * zmm_len); // TODO check
            jl(l_end, T_NEAR);

            // vmovdqu64(v_src, ptr[r64_src]);
            // vpshufb(v_src, v_src, v_shuf_mask); // Endianness swap

            // vpclmulqdq(xmm_aux, xmm_dst, xmm_k_16_17, 0b00000000);
            // vpclmulqdq(xmm_dst, xmm_dst, xmm_k_16_17, 0b00010001);
            // vperm2i128(ymm_dst, ymm_dst, ymm_dst, 0x1);
            // vperm2i128(ymm_aux, ymm_aux, ymm_aux, 0x1);
            // vpclmulqdq(xmm_aux, xmm_dst, xmm_k_16_17, 0b00000000);
            // vpclmulqdq(xmm_dst, xmm_dst, xmm_k_16_17, 0b00010001);
            // vperm2i128(ymm_dst, ymm_dst, ymm_dst, 0x1);
            // vperm2i128(ymm_aux, ymm_aux, ymm_aux, 0x1);
            // vpxorq(ymm_aux, ymm_aux, ymm_src);
            // vpxorq(ymm_dst, ymm_dst, ymm_aux);

            // 0
            vmovdqu64(v_src_0, ptr[r64_src]);
            vpshufb(v_src_0, v_src_0, v_shuf_mask); // Swap bytes

            vpclmulqdq(xmm_aux_0, xmm_dst_0, xmm_k_16_17, 0b00000000);
            vpclmulqdq(xmm_dst_0, xmm_dst_0, xmm_k_16_17, 0b00010001);
            vpxorq(xmm_aux_0, xmm_aux_0, xmm_src_0);
            vpxorq(xmm_dst_0, xmm_dst_0, xmm_aux_0);
            // 1
            vextracti64x2(xmm_src_0, v_src_0, 0x1);
            vpclmulqdq(xmm_aux_0, xmm_dst_1, xmm_k_16_17, 0b00000000);
            vpclmulqdq(xmm_dst_1, xmm_dst_1, xmm_k_16_17, 0b00010001);
            vpxorq(xmm_aux_0, xmm_aux_0, xmm_src_0);
            vpxorq(xmm_dst_1, xmm_dst_1, xmm_aux_0);
            // 2
            vextracti64x2(xmm_src_0, v_src_0, 0x2);
            vpclmulqdq(xmm_aux_0, xmm_dst_2, xmm_k_16_17, 0b00000000);
            vpclmulqdq(xmm_dst_2, xmm_dst_2, xmm_k_16_17, 0b00010001);
            vpxorq(xmm_aux_0, xmm_aux_0, xmm_src_0);
            vpxorq(xmm_dst_2, xmm_dst_2, xmm_aux_0);
            // 3
            vextracti64x2(xmm_src_0, v_src_0, 0x3);
            vpclmulqdq(xmm_aux_0, xmm_dst_3, xmm_k_16_17, 0b00000000);
            vpclmulqdq(xmm_dst_3, xmm_dst_3, xmm_k_16_17, 0b00010001);
            vpxorq(xmm_aux_0, xmm_aux_0, xmm_src_0);
            vpxorq(xmm_dst_3, xmm_dst_3, xmm_aux_0);

            // 4
            add(r64_src, zmm_len);
            vmovdqu64(v_src_0, ptr[r64_src]);
            vpshufb(v_src_0, v_src_0, v_shuf_mask); // Swap bytes

            vpclmulqdq(xmm_aux_0, xmm_dst_4, xmm_k_16_17, 0b00000000);
            vpclmulqdq(xmm_dst_4, xmm_dst_4, xmm_k_16_17, 0b00010001);
            vpxorq(xmm_aux_0, xmm_aux_0, xmm_src_0);
            vpxorq(xmm_dst_4, xmm_dst_4, xmm_aux_0);
            // 5
            vextracti64x2(xmm_src_0, v_src_0, 0x1);
            vpclmulqdq(xmm_aux_0, xmm_dst_5, xmm_k_16_17, 0b00000000);
            vpclmulqdq(xmm_dst_5, xmm_dst_5, xmm_k_16_17, 0b00010001);
            vpxorq(xmm_aux_0, xmm_aux_0, xmm_src_0);
            vpxorq(xmm_dst_5, xmm_dst_5, xmm_aux_0);
            // 6
            vextracti64x2(xmm_src_0, v_src_0, 0x2);
            vpclmulqdq(xmm_aux_0, xmm_dst_6, xmm_k_16_17, 0b00000000);
            vpclmulqdq(xmm_dst_6, xmm_dst_6, xmm_k_16_17, 0b00010001);
            vpxorq(xmm_aux_0, xmm_aux_0, xmm_src_0);
            vpxorq(xmm_dst_6, xmm_dst_6, xmm_aux_0);
            // 7
            vextracti64x2(xmm_src_0, v_src_0, 0x3);
            vpclmulqdq(xmm_aux_0, xmm_dst_7, xmm_k_16_17, 0b00000000);
            vpclmulqdq(xmm_dst_7, xmm_dst_7, xmm_k_16_17, 0b00010001);
            vpxorq(xmm_aux_0, xmm_aux_0, xmm_src_0);
            vpxorq(xmm_dst_7, xmm_dst_7, xmm_aux_0);

            jmp(l_fold_loop, T_NEAR);
        }

        mov(r64_aux, reinterpret_cast<uintptr_t>(CONST_K + 8));
        vpclmulqdq(xmm_aux_0, xmm_dst_0, ptr[r64_aux], 0b00000000);
        vpclmulqdq(xmm_dst_0, xmm_dst_0, ptr[r64_aux], 0b00010001);
        vpxorq(xmm_dst_7, xmm_dst_7, xmm_aux_0);
        vpxorq(xmm_dst_7, xmm_dst_7, xmm_dst_0);

        mov(r64_aux, reinterpret_cast<uintptr_t>(CONST_K + 10));
        vpclmulqdq(xmm_aux_0, xmm_dst_1, ptr[r64_aux], 0b00000000);
        vpclmulqdq(xmm_dst_1, xmm_dst_1, ptr[r64_aux], 0b00010001);
        vpxorq(xmm_dst_7, xmm_dst_7, xmm_aux_0);
        vpxorq(xmm_dst_7, xmm_dst_7, xmm_dst_1);

        mov(r64_aux, reinterpret_cast<uintptr_t>(CONST_K + 12));
        vpclmulqdq(xmm_aux_0, xmm_dst_2, ptr[r64_aux], 0b00000000);
        vpclmulqdq(xmm_dst_2, xmm_dst_2, ptr[r64_aux], 0b00010001);
        vpxorq(xmm_dst_7, xmm_dst_7, xmm_aux_0);
        vpxorq(xmm_dst_7, xmm_dst_7, xmm_dst_2);

        mov(r64_aux, reinterpret_cast<uintptr_t>(CONST_K + 14));
        vpclmulqdq(xmm_aux_0, xmm_dst_3, ptr[r64_aux], 0b00000000);
        vpclmulqdq(xmm_dst_3, xmm_dst_3, ptr[r64_aux], 0b00010001);
        vpxorq(xmm_dst_7, xmm_dst_7, xmm_aux_0);
        vpxorq(xmm_dst_7, xmm_dst_7, xmm_dst_3);

        mov(r64_aux, reinterpret_cast<uintptr_t>(CONST_K + 16));
        vpclmulqdq(xmm_aux_0, xmm_dst_4, ptr[r64_aux], 0b00000000);
        vpclmulqdq(xmm_dst_4, xmm_dst_4, ptr[r64_aux], 0b00010001);
        vpxorq(xmm_dst_7, xmm_dst_7, xmm_aux_0);
        vpxorq(xmm_dst_7, xmm_dst_7, xmm_dst_4);

        mov(r64_aux, reinterpret_cast<uintptr_t>(CONST_K + 18));
        vpclmulqdq(xmm_aux_0, xmm_dst_5, ptr[r64_aux], 0b00000000);
        vpclmulqdq(xmm_dst_5, xmm_dst_5, ptr[r64_aux], 0b00010001);
        vpxorq(xmm_dst_7, xmm_dst_7, xmm_aux_0);
        vpxorq(xmm_dst_7, xmm_dst_7, xmm_dst_5);

        mov(r64_aux, reinterpret_cast<uintptr_t>(CONST_K));
        vpclmulqdq(xmm_aux_0, xmm_dst_6, ptr[r64_aux], 0b00000000);
        vpclmulqdq(xmm_dst_6, xmm_dst_6, ptr[r64_aux], 0b00010001);
        vpxorq(xmm_dst_7, xmm_dst_7, xmm_aux_0);
        vpxorq(xmm_dst_7, xmm_dst_7, xmm_dst_6);

        L(l_end);
    }
}

template <>
void CombineHash<avx2>::bulkFold_128() {
    Xbyak::Label l_fold_loop, l_end;
    cmp(r64_work_amount, xmm_len);
    jl(l_end, T_NEAR);

    auto xmm_dst = Xbyak::Xmm(v_dst.getIdx());
    auto xmm_k_12 = Xbyak::Xmm(v_k_12.getIdx());
    auto xmm_shuf_mask = Xbyak::Xmm(v_shuf_mask.getIdx());
    auto xmm_src = getXmm();
    auto xmm_aux = getXmm();

    // prefetchnta(ptr[r64_src]); // TODO compare perf
    vmovups(xmm_src, ptr[r64_src]);
    vpshufb(xmm_src, xmm_src, xmm_shuf_mask); // Endianness swap
    vxorps(xmm_dst, xmm_dst, xmm_src);

    // Bulk fold
    L(l_fold_loop); {
        add(r64_src, xmm_len);
        sub(r64_work_amount, xmm_len);
        cmp(r64_work_amount, xmm_len);
        jl(l_end, T_NEAR);

        vmovups(xmm_src, ptr[r64_src]);
        vpshufb(xmm_src, xmm_src, xmm_shuf_mask); // Endianness swap

        vpclmulqdq(xmm_aux, xmm_dst, xmm_k_12, 0b00000000);
        vpclmulqdq(xmm_dst, xmm_dst, xmm_k_12, 0b00010001);
        vxorps(xmm_src, xmm_src, xmm_dst);
        vxorps(xmm_dst, xmm_src, xmm_aux);

        jmp(l_fold_loop, T_NEAR);
    }

    L(l_end);
}

// template <cpu_isa_t isa>
// const uint64_t CombineHash<isa>::K12 = 0x7B4BC8789D65B2A5;

template <cpu_isa_t isa>
const uint64_t CombineHash<isa>::INIT_CRC = 0xffffffffffffffff;

// Auxiliary fn to obtain K constant multipliers.
// uint64_t get_k_value(int t, uint64_t poly = 0x42F0E1EBA9EA3693) {
//     uint64_t res = poly, mask = 0x8000000000000000;
//     do {
//         if (res & mask) {
//             res = (res << 1) ^ poly;
//         } else {
//             res = (res << 1);
//         }
//     } while (--t);
//     std::cout << std::hex << "K64: " << res << std::endl;
//     return res;
// }

template <cpu_isa_t isa>
const uint64_t CombineHash<isa>::CONST_K[20] = { 0x05f5c3c7eb52fab6, 0x4eb938a7d257740e,
                                                 0x05cf79dea9ac37d6, 0x001067e571d7d5c2,
                                                 0x05f5c3c7eb52fab6, 0x0000000000000000,
                                                 0x578d29d06cc4f872, 0x42f0e1eba9ea3693,
                                                 0xe464f4df5fb60ac1, 0xb649c5b35a759cf2,
                                                 0X9af04e1eff82d0dd, 0x6e82e609297f8fe8,
                                                 0x097c516e98bd2e73, 0x0b76477b31e22e7b,
                                                 0x5f6843ca540df020, 0xddf4b6981205b83f,
                                                 0x54819d8713758b2c, 0x4a6b90073eb0af5a,
                                                 0x571bee0a227ef92b, 0x44bef2a201b5200c };

}   // namespace jit
#endif // OPENVINO_ARCH_X86 || OPENVINO_ARCH_X86_64

size_t combine_hash(const void* src, size_t size) {
// std::cout << "combine_hash size: " << size << std::endl;
#if defined(OPENVINO_ARCH_X86) || defined(OPENVINO_ARCH_X86_64)
    jit::fn_t kernel;
// if (jit::Generator::mayiuse(jit::vpclmulqdq)) {
//     std::cout << "CPU supports vpclmulqdq\n";
// }
    if (jit::Generator::mayiuse(jit::avx512_core)) {
        kernel = jit::CombineHash<jit::avx512_core>::get();
    } else if (jit::Generator::mayiuse(jit::avx2)) {
        kernel = jit::CombineHash<jit::avx2>::get();
    }

    if (kernel) {
        size_t res = 0lu;
        jit::CombineHashCallArgs args;
        args.src_ptr = src;
        args.dst_ptr = &res;
        args.work_amount = size;
        kernel(&args);
// std::cout << "combine_hash kernel: " << res << std::endl;
        return res;
    }
#endif // OPENVINO_ARCH_X86 || OPENVINO_ARCH_X86_64

    constexpr auto cel_size = sizeof(size_t);
    auto seed = static_cast<size_t>(size);
    const auto data = static_cast<const size_t*>(src);
    const auto d_end = std::next(data, size / cel_size);
    // The constant value used as a magic number has been
    // traditionally used e.g. in boost library's hash_combine.
    // It happens to be derived from the golden ratio.
    for (auto d = data; d != d_end; ++d) {
        seed ^= *d + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    size_t last_bytes{0};
    std::memcpy(&last_bytes, d_end, size % cel_size);
    seed ^= last_bytes + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    return seed;
}

}   // namespace runtime
}   // namespace ov
