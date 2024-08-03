// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// The CRC computation is used for x86
// The calculations were taken from the article "Fast CRC Computation for Generic Polynomials Using PCLMULQDQ Instruction - Intel (December, 2009)"

#include "openvino/core/visibility.hpp"
#include "openvino/core/parallel.hpp"
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
    uint64_t make_64_fold = 0lu;
    void* tmp_ptr; // TODO: remomve
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
        if (mayiuse(cpu_isa_t::vpclmulqdq)) {
            is_vpclmulqdq = true;
        }

        generate();
    }

    void generate() {
        this->preamble();
        registersPool = RegistersPool::create(isa, {rax, rcx, rsp, rdi, k0});

        r64_src = getReg64();
        r64_dst = getReg64();
        r64_work_amount  = getReg64();
        r64_make_64_fold = getReg64();
        r64_tmp = getReg64();

        mov(r64_src, ptr[r64_params + GET_OFF(src_ptr)]);
        mov(r64_dst, ptr[r64_params + GET_OFF(dst_ptr)]);
        mov(r64_work_amount, ptr[r64_params + GET_OFF(work_amount)]);
        mov(r64_make_64_fold, ptr[r64_params + GET_OFF(make_64_fold)]);
        mov(r64_tmp, ptr[r64_params + GET_OFF(tmp_ptr)]);

        initVectors();
        bulkFold(v_dst);
        restFold(v_dst);
        tailFold(v_dst);

        registersPool.reset();
        this->postamble();
    }

    static fn_t get() {
        static const CombineHashCompileParams params;
        static CombineHash<isa> kernel(params);

        return (fn_t)kernel.getCode();
    }

    void fillRestWorkMask(const Xbyak::Opmask& k_dst_mask,
                          const Xbyak::Reg64& r64_work_rest) {
        Xbyak::Label l_mv_mask;
        auto rOnes = getReg64();

        mov(rOnes, 0xFFFFFFFFFFFFFFFF);
        cmp(r64_work_rest, 0x3f);
        jg(l_mv_mask);

        shlx(rOnes, rOnes, r64_work_rest);
        not_(rOnes);

        L(l_mv_mask);
        kmovq(k_dst_mask, rOnes);
    }

    void partialLoad(const Xbyak::Xmm&     xmm_dst,
                     const Xbyak::Address& src_addr,
                     const Xbyak::Reg64&   r64_load_num) {
        Xbyak::Label l_partial, l_end;

        cmp(r64_load_num, xmm_len);
        jl(l_partial, T_NEAR);
        vmovdqu(xmm_dst, ptr[src_addr.getRegExp()]);
        jmp(l_end, T_NEAR);

        L(l_partial); {
            size_t offset = xmm_len;

            for (size_t j = 0lu; j < xmm_len - 1; j++) {
                pinsrb(xmm_dst, ptr[src_addr.getRegExp() + offset], j);
                cmp(r64_load_num, ++offset);
                jle(l_end, T_NEAR);
            }
        }

        L(l_end);
    }

    void partialLoad(const Xbyak::Ymm&     ymm_dst,
                     const Xbyak::Address& src_addr,
                     const Xbyak::Reg64&   r64_load_num) {
        Xbyak::Label l_xmm, l_partial, l_end;
        auto xmm_dst = Xbyak::Xmm(ymm_dst.getIdx());

        cmp(r64_load_num, ymm_len);
        jl(l_xmm, T_NEAR);
        vmovdqu(ymm_dst, ptr[src_addr.getRegExp()]);
        jmp(l_end, T_NEAR);

        L(l_xmm);
        // vpxorq(ymm_dst, ymm_dst, ymm_dst);
        cmp(r64_load_num, xmm_len);
        jl(l_partial, T_NEAR);
        vmovdqu(xmm_dst, ptr[src_addr.getRegExp()]);
        je(l_end, T_NEAR);

        {
            Xbyak::Label l_rest_loop, l_perm;
            size_t offset = xmm_len;

            vperm2f128(ymm_dst, ymm_dst, ymm_dst, 0x1);
            for (size_t j = 0lu; j < xmm_len - 1; j++) {
                pinsrb(xmm_dst, ptr[src_addr.getRegExp() + offset], j);
                cmp(r64_load_num, ++offset);
                jle(l_perm, T_NEAR);
            }
            L(l_perm);
            vperm2f128(ymm_dst, ymm_dst, ymm_dst, 0x1);
        }
        jmp(l_end, T_NEAR);

        L(l_partial); {
            size_t offset = xmm_len;

            for (size_t j = 0lu; j < xmm_len - 1; j++) {
                pinsrb(xmm_dst, ptr[src_addr.getRegExp() + offset], j);
                cmp(r64_load_num, ++offset);
                jle(l_end, T_NEAR);
            }
        }

        L(l_end);
    }

private:
    static constexpr uint64_t CHUNK_SIZE = 32;
    // static constexpr uint64_t P64 = 0x42F0E1EBA9EA3693;
    // static const uint64_t K12;
    static const uint64_t CRC_VAL;
    static const uint64_t CONST_K[54];
    static const uint8_t SHUF_MASK[16];

    using Vmm = typename std::conditional<isa == avx512_core, Xbyak::Zmm, Xbyak::Ymm>::type;
    size_t vlen = xmm_len;
    bool is_vpclmulqdq = false;

    CombineHashCompileParams m_jcp;
    RegistersPool::Ptr registersPool;

    RegistersPool::Reg<Xbyak::Reg64> r64_src;
    RegistersPool::Reg<Xbyak::Reg64> r64_dst;
    RegistersPool::Reg<Xbyak::Reg64> r64_work_amount;
    RegistersPool::Reg<Xbyak::Reg64> r64_make_64_fold;
    RegistersPool::Reg<Xbyak::Reg64> r64_tmp;

    const Xbyak::Reg64 r64_params = abi_param1;

    // Vector registers
    RegistersPool::Reg<Vmm> v_dst;
    RegistersPool::Reg<Vmm> v_k_1_2;
    // RegistersPool::Reg<Vmm> v_k_56;
    RegistersPool::Reg<Vmm> v_k_4_5;
    RegistersPool::Reg<Vmm> v_k_8_9;
    RegistersPool::Reg<Vmm> v_k_16_17;
    RegistersPool::Reg<Vmm> v_shuf_mask;

    size_t getVlen() {
        return vlen;
    }

    void initVectors();

    void bulkFold(const Vmm& v_dst);

    void restFold(const Vmm& v_dst) {
        Xbyak::Label l_fold_loop, l_end;
        cmp(r64_work_amount, xmm_len);
        jl(l_end, T_NEAR);

        // auto r64_aux = getReg64();
        auto xmm_shuf_mask = Xbyak::Xmm(v_shuf_mask.getIdx());
        auto xmm_k_1_2 = Xbyak::Xmm(v_k_1_2.getIdx());
        auto xmm_src = getXmm();
        auto xmm_dst = Xbyak::Xmm(v_dst.getIdx());
        auto xmm_aux = getXmm();

        L(l_fold_loop); {
            vmovdqu64(xmm_src, ptr[r64_src]);
            vpshufb(xmm_src, xmm_src, xmm_shuf_mask);

            vpclmulqdq(xmm_aux, xmm_dst, xmm_k_1_2, 0b00000000);
            vpclmulqdq(xmm_dst, xmm_dst, xmm_k_1_2, 0b00010001);
            vpxorq(xmm_dst, xmm_dst, xmm_aux);
            vpxorq(xmm_dst, xmm_dst, xmm_src);

            add(r64_src, xmm_len);
            sub(r64_work_amount, xmm_len);
            cmp(r64_work_amount, xmm_len);
            jge(l_fold_loop, T_NEAR);
        }

        L(l_end);
    }

    void tailFold(const Vmm& v_dst);
};

template <>
void CombineHash<avx512_core>::initVectors() {
    auto r64_aux = getReg64();

    v_k_1_2 = getVmm();
    mov(r64_aux, reinterpret_cast<uintptr_t>(CONST_K));
    vbroadcasti64x2(v_k_1_2, ptr[r64_aux]);
    v_k_8_9 = getVmm();
    mov(r64_aux, reinterpret_cast<uintptr_t>(CONST_K + 14));
    vbroadcasti64x2(v_k_8_9, ptr[r64_aux]);

    v_shuf_mask = getVmm();
    mov(r64_aux, reinterpret_cast<uintptr_t>(SHUF_MASK));
    vbroadcasti64x2(v_shuf_mask, ptr[r64_aux]);

    v_dst = getVmm();
    auto xmm_dst = Xbyak::Xmm(v_dst.getIdx());
    auto xmm_shuf_mask = Xbyak::Xmm(v_shuf_mask.getIdx());
    auto xmm_aux = getXmm();
    auto k_rest_mask = RegistersPool::Reg<Xbyak::Opmask>(registersPool);
    // Initial CRC
    mov(r64_aux, CRC_VAL);
    vpxorq(v_dst, v_dst, v_dst);
    vpinsrq(xmm_dst, xmm_dst, r64_aux, 0x1);
    // First xor with source
    fillRestWorkMask(k_rest_mask, r64_work_amount);
    vmovdqu8(Xbyak::Xmm(xmm_aux.getIdx()) | k_rest_mask, ptr[r64_src]);
    vpshufb(xmm_aux, xmm_aux, xmm_shuf_mask);
    vpxorq(xmm_dst, xmm_dst, xmm_aux);
// vmovdqu64(ptr[r64_tmp], xmm_dst);
    sub(r64_work_amount, xmm_len);
    add(r64_src, xmm_len);
}

template <cpu_isa_t isa>
void CombineHash<isa>::initVectors() {
    auto r64_aux = getReg64();

    v_k_1_2 = getVmm();
    mov(r64_aux, reinterpret_cast<uintptr_t>(CONST_K));
    vbroadcasti128(v_k_1_2, ptr[r64_aux]);
    v_k_8_9 = getVmm();
    mov(r64_aux, reinterpret_cast<uintptr_t>(CONST_K + 14));
    vbroadcasti128(v_k_8_9, ptr[r64_aux]);

    v_shuf_mask = getVmm();
    mov(r64_aux, reinterpret_cast<uintptr_t>(SHUF_MASK));
    vbroadcasti128(v_shuf_mask, ptr[r64_aux]);

    v_dst = getVmm();
    auto xmm_dst = Xbyak::Xmm(v_dst.getIdx());
    auto xmm_shuf_mask = Xbyak::Xmm(v_shuf_mask.getIdx());
    auto xmm_aux = getXmm();
    auto k_rest_mask = RegistersPool::Reg<Xbyak::Opmask>(registersPool);
    // Initial CRC
    mov(r64_aux, CRC_VAL);
    vpxorq(v_dst, v_dst, v_dst);
    vpinsrq(xmm_dst, xmm_dst, r64_aux, 0x1);
    // First xor with source
    partialLoad(xmm_aux, ptr[r64_src], r64_work_amount);
    vpshufb(xmm_aux, xmm_aux, xmm_shuf_mask);
    vpxorq(xmm_dst, xmm_dst, xmm_aux);
    sub(r64_work_amount, xmm_len);
}

template <>
void CombineHash<avx512_core>::bulkFold(const Vmm& v_dst) {
    Xbyak::Label l_fold_loop, l_end;
    cmp(r64_work_amount, zmm_len + 3 * xmm_len);
    jl(l_end, T_NEAR);

    auto r64_aux = getReg64();

    auto v_src_0 = getVmm();
    auto v_dst_0 = getVmm();
    auto v_dst_1 = getVmm();
    auto v_dst_2 = getVmm();
    auto& v_dst_3 = v_dst;
    auto v_aux_0 = getVmm();

    auto xmm_k_8_9 = Xbyak::Xmm(v_k_8_9.getIdx());
    auto xmm_k_1_2 = Xbyak::Xmm(v_k_1_2.getIdx());
    auto xmm_src_0 = Xbyak::Xmm(v_src_0.getIdx());
    auto xmm_src_1 = getXmm();
    auto xmm_dst_0 = Xbyak::Xmm(v_dst_0.getIdx());
    auto xmm_dst_1 = Xbyak::Xmm(v_dst_1.getIdx());
    auto xmm_dst_2 = Xbyak::Xmm(v_dst_2.getIdx());
    auto xmm_dst_3 = Xbyak::Xmm(v_dst_3.getIdx());
    auto xmm_aux_0 = Xbyak::Xmm(v_aux_0.getIdx());

    vmovdqu64(xmm_dst_0, xmm_dst_3);

    if (!is_vpclmulqdq) {
        // prefetchnta(ptr[r64_src]);
        vmovdqu64(xmm_dst_1, ptr[r64_src + 0 * xmm_len]);
        vmovdqu64(xmm_dst_2, ptr[r64_src + 1 * xmm_len]);
        vmovdqu64(xmm_dst_3, ptr[r64_src + 2 * xmm_len]);
    }

    add(r64_src, 3 * xmm_len);
    sub(r64_work_amount, zmm_len + 3 * xmm_len);

    L(l_fold_loop); {
        vmovdqu64(v_src_0, ptr[r64_src]);
        vpshufb(v_src_0, v_src_0, v_shuf_mask);

        if (is_vpclmulqdq) {
            vpclmulqdq(v_aux_0, v_dst_0, v_k_8_9, 0b00000000);
            vpclmulqdq(v_dst_0, v_dst_0, v_k_8_9, 0b00010001);
            vpxorq(v_aux_0, v_aux_0, v_src_0);
            vpxorq(v_dst_0, v_dst_0, v_aux_0);
        } else {
            // 0
            vpclmulqdq(xmm_aux_0, xmm_dst_0, xmm_k_8_9, 0b00000000);
            vpclmulqdq(xmm_dst_0, xmm_dst_0, xmm_k_8_9, 0b00010001);
            vpxorq(xmm_aux_0, xmm_aux_0, xmm_src_0);
            vpxorq(xmm_dst_0, xmm_dst_0, xmm_aux_0);
            // 1
            vextracti64x2(xmm_src_1, v_src_0, 0x1);
            vpclmulqdq(xmm_aux_0, xmm_dst_1, xmm_k_8_9, 0b00000000);
            vpclmulqdq(xmm_dst_1, xmm_dst_1, xmm_k_8_9, 0b00010001);
            vpxorq(xmm_aux_0, xmm_aux_0, xmm_src_1);
            vpxorq(xmm_dst_1, xmm_dst_1, xmm_aux_0);
            // 2
            vextracti64x2(xmm_src_1, v_src_0, 0x2);
            vpclmulqdq(xmm_aux_0, xmm_dst_2, xmm_k_8_9, 0b00000000);
            vpclmulqdq(xmm_dst_2, xmm_dst_2, xmm_k_8_9, 0b00010001);
            vpxorq(xmm_aux_0, xmm_aux_0, xmm_src_1);
            vpxorq(xmm_dst_2, xmm_dst_2, xmm_aux_0);
            // 3
            vextracti64x2(xmm_src_1, v_src_0, 0x3);
            vpclmulqdq(xmm_aux_0, xmm_dst_3, xmm_k_8_9, 0b00000000);
            vpclmulqdq(xmm_dst_3, xmm_dst_3, xmm_k_8_9, 0b00010001);
            vpxorq(xmm_aux_0, xmm_aux_0, xmm_src_1);
            vpxorq(xmm_dst_3, xmm_dst_3, xmm_aux_0);
        }

        add(r64_src, zmm_len);
        sub(r64_work_amount, zmm_len);
        jge(l_fold_loop, T_NEAR);
    }
    add(r64_work_amount, zmm_len);

    if (is_vpclmulqdq) {
        auto ymm_dst_0 = Xbyak::Ymm(v_dst_0.getIdx());
        auto ymm_dst_1 = Xbyak::Ymm(v_dst_1.getIdx());
        auto ymm_aux_0 = Xbyak::Ymm(v_aux_0.getIdx());

        vextracti64x4(ymm_dst_1, v_dst_0, 0x1);
        mov(r64_aux, reinterpret_cast<uintptr_t>(CONST_K + 18));
        vpclmulqdq(ymm_aux_0, ymm_dst_0, ptr[r64_aux], 0b00000000);
        vpclmulqdq(ymm_dst_0, ymm_dst_0, ptr[r64_aux], 0b00010001);
        vpxorq(ymm_dst_1, ymm_dst_1, ymm_aux_0);
        vpxorq(ymm_dst_0, ymm_dst_0, ymm_dst_1);

        vextracti64x2(xmm_dst_3, ymm_dst_0, 0x1);
        vpclmulqdq(xmm_aux_0, xmm_dst_0, xmm_k_1_2, 0b00000000);
        vpclmulqdq(xmm_dst_0, xmm_dst_0, xmm_k_1_2, 0b00010001);
        vpxorq(xmm_dst_3, xmm_dst_3, xmm_aux_0);
        vpxorq(xmm_dst_3, xmm_dst_3, xmm_dst_0);
    } else {
        mov(r64_aux, reinterpret_cast<uintptr_t>(CONST_K + 16));
        vpclmulqdq(xmm_aux_0, xmm_dst_0, ptr[r64_aux], 0b00000000);
        vpclmulqdq(xmm_dst_0, xmm_dst_0, ptr[r64_aux], 0b00010001);
        vpxorq(xmm_dst_3, xmm_dst_3, xmm_aux_0);
        vpxorq(xmm_dst_3, xmm_dst_3, xmm_dst_0);

        mov(r64_aux, reinterpret_cast<uintptr_t>(CONST_K + 18));
        vpclmulqdq(xmm_aux_0, xmm_dst_1, ptr[r64_aux], 0b00000000);
        vpclmulqdq(xmm_dst_1, xmm_dst_1, ptr[r64_aux], 0b00010001);
        vpxorq(xmm_dst_3, xmm_dst_3, xmm_aux_0);
        vpxorq(xmm_dst_3, xmm_dst_3, xmm_dst_1);

        vpclmulqdq(xmm_aux_0, xmm_dst_2, xmm_k_1_2, 0b00000000);
        vpclmulqdq(xmm_dst_2, xmm_dst_2, xmm_k_1_2, 0b00010001);
        vpxorq(xmm_dst_3, xmm_dst_3, xmm_aux_0);
        vpxorq(xmm_dst_3, xmm_dst_3, xmm_dst_2);
    }

    L(l_end);
}

template <>
void CombineHash<avx2>::bulkFold(const Vmm& v_dst) {
    Xbyak::Label l_fold_loop, l_end;
    cmp(r64_work_amount, 2 * vlen - xmm_len);
    jl(l_end, T_NEAR);

    auto r64_aux = getReg64();

    auto v_src_0 = getVmm();
    auto v_dst_0 = getVmm();
    auto v_dst_1 = getVmm();
    auto v_dst_2 = getVmm();
    auto& v_dst_3 = v_dst;
    auto v_aux_0 = getVmm();

    auto xmm_k_4_5 = Xbyak::Xmm(v_k_4_5.getIdx());
    auto xmm_k_1_2 = Xbyak::Xmm(v_k_1_2.getIdx());
    auto xmm_src_0 = Xbyak::Xmm(v_src_0.getIdx());
    auto xmm_src_1 = getXmm();
    auto xmm_dst_0 = Xbyak::Xmm(v_dst_0.getIdx());
    auto xmm_dst_1 = Xbyak::Xmm(v_dst_1.getIdx());
    auto xmm_dst_2 = Xbyak::Xmm(v_dst_2.getIdx());
    auto xmm_dst_3 = Xbyak::Xmm(v_dst_3.getIdx());
    auto xmm_aux_0 = Xbyak::Xmm(v_aux_0.getIdx());

    if (!is_vpclmulqdq) {
        vmovdqu64(xmm_dst_1, ptr[r64_src + 0 * xmm_len]);
    }

    add(r64_src, vlen - xmm_len);
    sub(r64_work_amount, 2 * vlen - xmm_len);

    L(l_fold_loop); {
        vmovdqu64(v_src_0, ptr[r64_src]);
        vpshufb(v_src_0, v_src_0, v_shuf_mask);

        if (is_vpclmulqdq) {
            vpclmulqdq(v_aux_0, v_dst_0, v_k_4_5, 0b00000000);
            vpclmulqdq(v_dst_0, v_dst_0, v_k_4_5, 0b00010001);
            vpxorq(v_aux_0, v_aux_0, v_src_0);
            vpxorq(v_dst_0, v_dst_0, v_aux_0);
        } else {
            // 0
            vpclmulqdq(xmm_aux_0, xmm_dst_0, xmm_k_4_5, 0b00000000);
            vpclmulqdq(xmm_dst_0, xmm_dst_0, xmm_k_4_5, 0b00010001);
            vpxorq(xmm_aux_0, xmm_aux_0, xmm_src_0);
            vpxorq(xmm_dst_0, xmm_dst_0, xmm_aux_0);
            // 1
            vextracti128(xmm_src_1, v_src_0, 0x1);
            vpclmulqdq(xmm_aux_0, xmm_dst_1, xmm_k_4_5, 0b00000000);
            vpclmulqdq(xmm_dst_1, xmm_dst_1, xmm_k_4_5, 0b00010001);
            vpxorq(xmm_aux_0, xmm_aux_0, xmm_src_1);
            vpxorq(xmm_dst_1, xmm_dst_1, xmm_aux_0);
        }

        add(r64_src, vlen);
        sub(r64_work_amount, vlen);
        jge(l_fold_loop, T_NEAR);
    }
    add(r64_work_amount, vlen);

    if (is_vpclmulqdq) {
        auto ymm_dst_0 = Xbyak::Ymm(v_dst_0.getIdx());
        auto ymm_dst_1 = Xbyak::Ymm(v_dst_1.getIdx());
        auto ymm_aux_0 = Xbyak::Ymm(v_aux_0.getIdx());

        vextracti128(xmm_dst_3, ymm_dst_0, 0x1);
        vpclmulqdq(xmm_aux_0, xmm_dst_0, xmm_k_1_2, 0b00000000);
        vpclmulqdq(xmm_dst_0, xmm_dst_0, xmm_k_1_2, 0b00010001);
        vpxorq(xmm_dst_3, xmm_dst_3, xmm_aux_0);
        vpxorq(xmm_dst_3, xmm_dst_3, xmm_dst_0);
    } else {
        vpclmulqdq(xmm_aux_0, xmm_dst_2, xmm_k_1_2, 0b00000000);
        vpclmulqdq(xmm_dst_2, xmm_dst_2, xmm_k_1_2, 0b00010001);
        vpxorq(xmm_dst_3, xmm_dst_3, xmm_aux_0);
        vpxorq(xmm_dst_3, xmm_dst_3, xmm_dst_2);
    }

    L(l_end);
}


template <>
void CombineHash<avx512_core>::tailFold(const Vmm& v_dst) {
    Xbyak::Label l_fold_to_64, l_save_128, l_end;
    cmp(r64_work_amount, 0);
    jle(l_fold_to_64, T_NEAR);

    auto r64_aux = getReg64();
    auto xmm_shuf_mask = Xbyak::Xmm(v_shuf_mask.getIdx());
    auto xmm_k_1_2 = Xbyak::Xmm(v_k_1_2.getIdx()); // TODO calc a new table for bytes
    auto xmm_src = getXmm();
    auto xmm_dst = Xbyak::Xmm(v_dst.getIdx());
    auto xmm_aux = getXmm();
    auto xmm_aux_1 = getXmm();
    auto xmm_aux_2 = getXmm();
    auto k_rest_mask = RegistersPool::Reg<Xbyak::Opmask>(registersPool);

    fillRestWorkMask(k_rest_mask, r64_work_amount);

    vpxorq(xmm_src, xmm_src, xmm_src);
    vmovdqu8(Xbyak::Xmm(xmm_src.getIdx()) | k_rest_mask, ptr[r64_src]);
    vpshufb(xmm_src, xmm_src, xmm_shuf_mask);

    vpclmulqdq(xmm_aux, xmm_dst, xmm_k_1_2, 0b00000000);
    vpclmulqdq(xmm_dst, xmm_dst, xmm_k_1_2, 0b00010001);
    vpxorq(xmm_aux, xmm_aux, xmm_src);
    vpxorq(xmm_dst, xmm_dst, xmm_aux);

    L(l_fold_to_64);
    cmp(r64_make_64_fold, 0);
    je(l_save_128, T_NEAR);

    mov(r64_aux, reinterpret_cast<uintptr_t>(CONST_K + 4));
    vpclmulqdq(xmm_aux, xmm_dst, ptr[r64_aux], 0b00000001);
    vpslldq(xmm_dst, xmm_dst, 0x8);
    vpxorq(xmm_dst, xmm_dst, xmm_aux);

    mov(r64_aux, reinterpret_cast<uintptr_t>(CONST_K + 6));
    vmovdqu64(xmm_aux_2, ptr[r64_aux]);
    vpclmulqdq(xmm_aux, xmm_dst, xmm_aux_2, 0b00000001);
    mov(r64_aux, 0x0);
    vpinsrq(xmm_aux_1, xmm_dst, r64_aux, 0x0);
    vpxorq(xmm_aux, xmm_aux, xmm_aux_1);
    vpinsrq(xmm_aux_1, xmm_aux, r64_aux, 0x0);
    vpclmulqdq(xmm_aux, xmm_aux, xmm_aux_2, 0b00010001);
    vpxorq(xmm_aux, xmm_aux, xmm_aux_1);
    vpxorq(xmm_dst, xmm_dst, xmm_aux);

    vpextrq(ptr[r64_dst], xmm_dst, 0x0);
    jmp(l_end, T_NEAR);


    L(l_save_128);
    vmovdqu64(ptr[r64_dst], xmm_dst);

    L(l_end);
}

template <>
void CombineHash<avx2>::tailFold(const Vmm& v_dst) {
}

// template <cpu_isa_t isa>
// const uint64_t CombineHash<isa>::K12 = 0x7B4BC8789D65B2A5;

template <cpu_isa_t isa>
const uint64_t CombineHash<isa>::CRC_VAL = 0xffffffffffffffff;

// Auxiliary fn to obtain K constant multipliers.
// uint32_t get_k_value(int t, uint32_t poly = 0x04C11DB7) {
// uint32_t get_k_value(int t, uint32_t poly = 0xD663B05D) {
// uint32_t get_k_value(int t, uint32_t poly = 0x741B8CD7) {
    // uint32_t res = poly, mask = 0x80000000;
// uint64_t get_k_value(int t, uint64_t poly = 0xD663B05D) {
uint64_t get_k_value(int t, uint64_t poly = 0x42F0E1EBA9EA3693) {
    uint64_t res = poly, mask = 0x8000000000000000;
    do {
        // std::cout << std::dec << "t: " << t << std::endl;
        if (res & mask) {
            res = (res << 1) ^ poly;
        } else {
            res = (res << 1);
        }
    } while (--t);
    // std::cout << std::hex << "K64: " << res << std::endl;
    return res;
}

uint64_t xt_mod_P_neg(int t, uint64_t poly = 0x42F0E1EBA9EA3693) {
    uint64_t v = 1lu;
// uint32_t xt_mod_P_neg(int t, uint32_t poly = 0xD663B05D) {
    // uint32_t v = 1;
    int bits = 32;
    do {
        if (v & 1)
            v = (v >> 1) ^ (1u << (bits - 1) | poly >> 1);
        else
            v = (v >> 1);
    } while (--t);
    return v;
}

// uint64_t get_k_value_reflect(int t, uint64_t poly = 0xD663B05D) {
uint64_t get_k_value_reflect(int t, uint64_t poly = 0xEB31D82E) {
// uint64_t get_k_value_reflect(int t, uint64_t poly = 0xEDB88320) {
// uint64_t get_k_value_reflect(int t, uint64_t poly = 0x42F0E1EBA9EA3693) {
    uint64_t v = poly;
    do {
        if (v & 1) {
            v = (v >> 1) ^ poly;
        } else {
            v = (v >> 1);
        }
    } while (--t);
    return v;

    // uint32_t table[16] = {0};
	// uint32_t p = poly, poly_32 = poly;
    // int bits = 32;
	// int i,j;
	// table[0] = 0;
	// table[1] = p;
	// for (i=1;(1<<i)<16;i++)
	// {
	// 	if (p&1)
	// 		p = (p>>1) ^ poly_32;
	// 	else
	// 		p = (p>>1);
	// 	table[(1<<i)] = p;
	// 	for(j=1; j<(1<<i); j++) {
	// 		table[(1<<i)+j] = p ^ table[j];
	// 	}
	// }
	// printf("POLY=0x%0*X\n", bits/4, poly_32);
	// for(i=0;i<16;i++){
	// 	int ri;
	// 	ri = ( i&0x3)<<2 | ( i&0xC)>>2;
	// 	ri = (ri&0x5)<<1 | (ri&0xA)>>1;
	// 	printf("0x%0*X, ", bits/4, table[ri]);
	// 	if ((i&0x7)==0x7) printf("\n");
	// }
	// printf("\n");
    // return 0;
}

uint64_t barrett_calc(uint64_t poly = 0x42F0E1EBA9EA3693, int bits = 64) {
    int n = bits;
    uint64_t r = poly;
    uint64_t v = 0lu;
    while (--n) {
        if (r & (1ULL << n)) {
            r ^= poly >> (bits-n);
            v |= 1ULL << n;
        }
    }
    if (r) v|=1;// для деления с остатком округляем в большую сторону
    return v;
}

template <cpu_isa_t isa>
const uint64_t CombineHash<isa>::CONST_K[54] = { 0x05f5c3c7eb52fab6, 0x4eb938a7d257740e,  // x^(64*1), x^(64*2) U
                                                 0x05cf79dea9ac37d6, 0x001067e571d7d5c2,  // x^(64*15), x^(64*16)
                                                 0x05f5c3c7eb52fab6, 0x0000000000000000,  // x^(64*1), x^(64*0)
                                                 0x578d29d06cc4f872, 0x42f0e1eba9ea3693,  // x^(64*), x^(64*)
                                                 0xe464f4df5fb60ac1, 0xb649c5b35a759cf2,  // x^(64*13), x^(64*14)
                                                 0X9af04e1eff82d0dd, 0x6e82e609297f8fe8,  // x^(64*11), x^(64*12)
                                                 0x097c516e98bd2e73, 0x0b76477b31e22e7b,  // x^(64*9), x^(64*10)
                                                 0x5f6843ca540df020, 0xddf4b6981205b83f,  // x^(64*7), x^(64*8) U
                                                 0x54819d8713758b2c, 0x4a6b90073eb0af5a,  // x^(64*5), x^(64*6) U
                                                 0x571bee0a227ef92b, 0x44bef2a201b5200c,  // x^(64*3), x^(64*4) U
                                                 0x05f5c3c7eb52fab6, 0x4eb938a7d257740e,  // TODO the last one repeats the first. Modify?

                                                 0x34e4dc94ed8d963f, 0x00000001, // x^{95}, x^{31} 
                                                 0xe59c4cf90ce5976b, 0x94e66b9518f59db3, // x^{-25}, x^{-89}
                                                 0x9B1BE78B, 0xd84c42383503cf94, // x^{-17}, x^{-81}
                                                 0xC790B954, 0xfe004045f5526c4c, // x^{-9},  x^{-73}
                                                 0xD663B05D, 0x05f5c3c7eb52fab6, // x^{-1},  x^{-65}
                                                 0x01000000, 0x6e4d3e593561ee88, // x^{7},   x^{-57}
                                                 0x00010000, 0xc7cc909df556430c, // x^{15},  x^{-49}
                                                 0x00000100, 0xf40847980ddd6874, // x^{23},  x^{-41}
                                                 0x00000001, 0x770a6888f4a2ef70, // x^{31},  x^{-33}
                                                 0x0b854997ba2f81e7, 0xe59c4cf90ce5976b, // x^{39},  x^{-25}
                                                 0x1bb7156710bcf7af, 0x9B1BE78B, // x^{47}, x^{-17}
                                                 0x34b9c2e2a60fcb9f, 0xC790B954, // x^{55}, x^{-9}
                                                 0x30b9c22b9927dbca, 0xD663B05D, // x^{63}, x^{-1}
                                                 0x195cbfc1cd2a901a, 0x01000000, // x^{71}, x^{7}
                                                 0x347c386a3c2863a3, 0x00010000, // x^{79}, x^{15}
                                                 0x045982c73b0b613c, 0x00000100  // x^{87}, x^{23}
                                                };

template <cpu_isa_t isa>
const uint8_t CombineHash<isa>::SHUF_MASK[] = { 0b00001111, 0b00001110, 0b00001101, 0b00001100, 0b00001011, 0b00001010, 0b00001001, 0b00001000,
                                                0b00000111, 0b00000110, 0b00000101, 0b00000100, 0b00000011, 0b00000010, 0b00000001, 0b00000000 };

}   // namespace jit
#endif // OPENVINO_ARCH_X86 || OPENVINO_ARCH_X86_64

size_t combine_hash(const void* src, size_t size) {
// static uint64_t counter = 0;
// static uint64_t sum = 0;
// counter++;
// auto t1 = std::chrono::high_resolution_clock::now();
// std::cout << "barrett_calc 64: " << std::hex << jit::barrett_calc()
//           << "; barrett_calc 32: " << std::hex << jit::barrett_calc(0xD663B05D, 32)
//           << "; barrett_calc 32: " << std::hex << jit::barrett_calc(0x04C11DB7, 32)
//           << "; barrett_calc 16: " << std::hex << jit::barrett_calc(0x8408, 16)
//           << "; barrett_calc 16: " << std::hex << jit::barrett_calc(0xA001, 16)
//           << "; barrett_calc 16m: " << std::hex << jit::barrett_calc(0x4003, 16)
//           << "; barrett_calc 16b: " << std::hex << jit::barrett_calc(0x0811, 16) << std::endl;
// if (size == 0)
    // std::cout << "[CORE] combine_hash size: " << size << std::endl;
#if defined(OPENVINO_ARCH_X86) || defined(OPENVINO_ARCH_X86_64)
    jit::fn_t kernel;
// if (jit::Generator::mayiuse(jit::vpclmulqdq)) {
//     std::cout << "CPU supports vpclmulqdq\n";
// }

// std::cout << std::hex
//           << "\nx^7: " << jit::get_k_value_reflect(7)
//           << "\nx^15: " << jit::get_k_value_reflect(15)
//           << "\nx^23: " << jit::get_k_value_reflect(23)
        //   << "\nx^64*3: " << jit::get_k_value_reflect(64*3)
        //   << "\nx^64*4: " << jit::get_k_value_reflect(64*4)
        //   << "\nx^64*5: " << jit::get_k_value_reflect(64*5)
        //   << "\nx^64*6: " << jit::get_k_value_reflect(64*6)
        //   << "\nx^64*7: " << jit::get_k_value_reflect(64*7)
        //   << "\nx^64*8: " << jit::get_k_value_reflect(64*8)
        //   << "\nx^64*9: " << jit::get_k_value_reflect(64*9)
        //   << "\nx^64*10: " << jit::get_k_value_reflect(64*10)
        //   << "\nx^64*11: " << jit::get_k_value_reflect(64*11)
        //   << "\nx^64*12: " << jit::get_k_value_reflect(64*12)
        //   << "\nx^64*13: " << jit::get_k_value_reflect(64*13)
        //   << "\nx^64*14: " << jit::get_k_value_reflect(64*14)
        //   << "\nx^64*15: " << jit::get_k_value_reflect(64*15)
        //   << "\nx^64*16: " << jit::get_k_value_reflect(64*16)
        //   << "\nx^64*17: " << jit::get_k_value_reflect(64*17)
        //   << "\nx^64*18: " << jit::get_k_value_reflect(64*18)
        //   << "\nx^64*19: " << jit::get_k_value_reflect(64*19)
        //   << "\nx^64*20: " << jit::get_k_value_reflect(64*20)
        //   << std::endl;
// std::cout << std::hex
//           << "\nx^-128: "  << jit::get_k_value(-128)
//           << "\nx^-96: "  << jit::get_k_value(-96)
//           << "\nx^-80: "  << jit::get_k_value(-80)
//           << "\nx^-64: "  << jit::get_k_value(-64)
//           << "\nx^-56: "  << jit::get_k_value(-56)
//           << "\nx^-48: "  << jit::get_k_value(-48)
//           << "\nx^-40: "  << jit::get_k_value(-40)
//           << "\nx^-32: "  << jit::get_k_value(-32)
//           << "\nx^-24: "  << jit::get_k_value(-24)
//           << "\nx^-16: "  << jit::get_k_value(-16)
//           << "\nx^-8: "  << jit::get_k_value(-8)
//           << "\nx^-1: "  << jit::get_k_value(-1)
//           << "\nx^0: "  << jit::get_k_value(0)
//           << "\nx^1: "  << jit::get_k_value(1)
//           << "\nx^2: "  << jit::get_k_value(2)
//           << "\nx^3: "  << jit::get_k_value(3)
//           << "\nx^4: "  << jit::get_k_value(4)
//           << "\nx^5: "  << jit::get_k_value(5)
//           << "\nx^6: "  << jit::get_k_value(6)
//           << "\nx^7: "  << jit::get_k_value(7)
//           << "\nx^8: "  << jit::get_k_value(8)
//           << "\nx^15: " << jit::get_k_value(15)
//           << "\nx^16: " << jit::get_k_value(16)
//           << "\nx^23: " << jit::get_k_value(23)
//           << "\nx^24: " << jit::get_k_value(24)
//           << "\nx^32: " << jit::get_k_value(32)
//           << "\nx^40: " << jit::get_k_value(40)
//           << "\nx^48: " << jit::get_k_value(48)
//           << "\nx^56: " << jit::get_k_value(56)
//           << "\nx^64: " << jit::get_k_value(64)
//           << "\nx^80: " << jit::get_k_value(80)
//           << "\nx^88: " << jit::get_k_value(88)
//           << "\nx^96: "  << jit::get_k_value(96)
//           << "\nx^104: " << jit::get_k_value(104)
//           << "\nx^128: " << jit::get_k_value(128)
//           << "\nx^144: " << jit::get_k_value(144)
//           << "\nx^159: " << jit::get_k_value(159)
//           << "\nx^160: " << jit::get_k_value(160)
//           << "\nx^192: " << jit::get_k_value(192)
        //   << "\nx^64*0: " << jit::get_k_value(64*0)
        //   << "\nx^64*1: " << jit::get_k_value(64*1)
        //   << "\nx^64*2: " << jit::get_k_value(64*2)
        //   << "\nx^64*3: " << jit::get_k_value(64*3)
        //   << "\nx^64*4: " << jit::get_k_value(64*4)
        //   << "\nx^64*5: " << jit::get_k_value(64*5)
        //   << "\nx^64*6: " << jit::get_k_value(64*6)
        //   << "\nx^64*7: " << jit::get_k_value(64*7)
        //   << "\nx^64*8: " << jit::get_k_value(64*8)
        //   << "\nx^64*9: " << jit::get_k_value(64*9)
        //   << "\nx^64*10: " << jit::get_k_value(64*10)
        //   << "\nx^64*11: " << jit::get_k_value(64*11)
        //   << "\nx^64*12: " << jit::get_k_value(64*12)
        //   << "\nx^64*13: " << jit::get_k_value(64*13)
        //   << "\nx^64*14: " << jit::get_k_value(64*14)
        //   << "\nx^64*15: " << jit::get_k_value(64*15)
        //   << "\nx^64*16: " << jit::get_k_value(64*16)
        //   << "\nx^64*17: " << jit::get_k_value(64*17)
        //   << "\nx^64*18: " << jit::get_k_value(64*18)
        //   << "\nx^64*19: " << jit::get_k_value(64*19)
        //   << "\nx^64*20: " << jit::get_k_value(64*20)
        //   << std::endl;
    // std::cout << std::hex;
    // for (int i = 8; i < 65; i += 8) {
    //     // std::cout << std::dec << "\nx^" << i << ": " << std::hex << jit::get_k_value(i);
    //     std::cout << std::dec << "\nx^" << i << ": " << std::hex << jit::xt_mod_P_neg(i);
    // }
    // std::cout << std::endl;
    // std::cout << std::dec << "\nx^" << 128 << ": " << std::hex << jit::xt_mod_P_neg(128);
// std::cout << std::hex
//           << "\nx^-128: "  << jit::xt_mod_P_neg(-128)
//           << "\nx^-96: "  << jit::xt_mod_P_neg(-96)
//           << "\nx^-80: "  << jit::xt_mod_P_neg(-80)
//           << "\nx^-64: "  << jit::xt_mod_P_neg(-64)
//           << "\nx^-56: "  << jit::xt_mod_P_neg(-56)
//           << "\nx^-48: "  << jit::xt_mod_P_neg(-48)
//           << "\nx^-40: "  << jit::xt_mod_P_neg(-40)
//           << "\nx^-32: "  << jit::xt_mod_P_neg(-32)
//           << "\nx^-24: "  << jit::xt_mod_P_neg(-24)
//           << "\nx^-16: "  << jit::xt_mod_P_neg(-16)
//           << "\nx^-8: "  << jit::xt_mod_P_neg(-8)
//           << "\nx^-1: "  << jit::xt_mod_P_neg(-1)
//           << "\nx^0: "  << jit::xt_mod_P_neg(0)
//           << "\nx^1: "  << jit::xt_mod_P_neg(1)
//           << "\nx^2: "  << jit::xt_mod_P_neg(2)
//           << "\nx^3: "  << jit::xt_mod_P_neg(3)
//           << "\nx^4: "  << jit::xt_mod_P_neg(4)
//           << "\nx^5: "  << jit::xt_mod_P_neg(5)
//           << "\nx^6: "  << jit::xt_mod_P_neg(6)
//           << "\nx^7: "  << jit::xt_mod_P_neg(7)
//           << "\nx^8: "  << jit::xt_mod_P_neg(8)
//           << "\nx^15: " << jit::xt_mod_P_neg(15)
//           << "\nx^16: " << jit::xt_mod_P_neg(16)
//           << "\nx^23: " << jit::xt_mod_P_neg(23)
//           << "\nx^24: " << jit::xt_mod_P_neg(24)
//           << "\nx^32: " << jit::xt_mod_P_neg(32)
//           << "\nx^40: " << jit::xt_mod_P_neg(40)
//           << "\nx^48: " << jit::xt_mod_P_neg(48)
//           << "\nx^56: " << jit::xt_mod_P_neg(56)
//           << "\nx^64: " << jit::xt_mod_P_neg(64)
//           << "\nx^80: " << jit::xt_mod_P_neg(80)
//           << "\nx^88: " << jit::xt_mod_P_neg(88)
//           << "\nx^96: " << jit::xt_mod_P_neg(96)
//           << "\nx^104: " << jit::xt_mod_P_neg(104)
//           << "\nx^144: " << jit::xt_mod_P_neg(144)
//           << "\nx^159: " << jit::xt_mod_P_neg(159)
//           << "\nx^160: " << jit::xt_mod_P_neg(160)
//           << std::endl;
    if (jit::Generator::mayiuse(jit::avx512_core)) {
        kernel = jit::CombineHash<jit::avx512_core>::get();
    } else if (jit::Generator::mayiuse(jit::avx2)) {
        kernel = jit::CombineHash<jit::avx2>::get();
    }

    if (kernel) {
        size_t res = 0lu;

        static const size_t block_size = 2lu * jit::Generator::zmm_len;
        // There is no sense to perform parallel execution if there are less than 2 blocks.
        if (size >= 2lu * block_size) {
            const auto nthr = parallel_get_max_threads() / 2; // TODO: WA for Hyper Threading
            std::vector<uint64_t> intermediate(nthr * 2); // xmm_len * nthr
            const uint64_t blocks = size / block_size;
            const uint64_t el_per_thread = block_size * ((blocks + nthr - 1) / nthr);

// std::vector<uint64_t> tmp_vec(nthr * 2);
// std::vector<uint64_t> tmp_vec_2(nthr * 2);

// if (!(counter == 39)) {
// if (!(counter == 39 || counter == 84)) {
            parallel_nt(nthr, [&](const int ithr, const int nthr) {
                uint64_t start = ithr * el_per_thread;
                if (start >= size) {
                    return;
                }
                uint64_t work_amount = (el_per_thread + start > size) ? size - start : el_per_thread;

                size_t res = 0lu;
                jit::CombineHashCallArgs args;

                args.src_ptr = reinterpret_cast<const uint8_t *>(src) + start;
                args.dst_ptr = &intermediate[ithr * 2];
                args.work_amount = work_amount;
                args.make_64_fold = 0lu;
// args.tmp_ptr = &(tmp_vec_2[ithr * 2]);
// if ((counter == 39 || counter == 84) && ithr == 1)
//     printf("    [%d] start: %lu, work_amount: %lu\n", ithr, start, work_amount);
                kernel(&args);
            });
// } else {
//     for (int ithr = 0; ithr < nthr; ithr++) {
//         uint64_t start = ithr * el_per_thread;
//         if (start >= size) {
//             continue;
//         }
//         uint64_t work_amount = (el_per_thread + start > size) ? size - start : el_per_thread;

//         size_t res = 0lu;
//         jit::CombineHashCallArgs args;

//         args.src_ptr = reinterpret_cast<const uint8_t *>(src) + start;
//         args.dst_ptr = &(intermediate[ithr * 2]);
//         args.work_amount = work_amount;
//         args.make_64_fold = 0lu;
// args.tmp_ptr = &(tmp_vec[ithr * 2]);
//         kernel(&args);
//     }
// }

// if (counter == 39) {
// // if (counter == 39 || counter == 84) {
//     std::cout << "Combine hash " << counter << " Hash: " ;
//     for (int i = 0; i < intermediate.size(); i++) {
//         std::cout << intermediate[i] << "; ";
//     }
//     std::cout << std::endl << "    tmp vals: ";
//     for (int i = 0; i < tmp_vec.size(); i++) {
//         std::cout << tmp_vec[i] << "; ";
//     }
//     std::cout << std::endl;

//     auto data = reinterpret_cast<const uint8_t *>(src);// + 131072;
//     for (int i = 0; i < 131072; i++) {
//         std::cout << static_cast<uint32_t>(data[i]) << std::endl;
//     }
// }

            jit::CombineHashCallArgs args;
            args.src_ptr = intermediate.data();
            args.dst_ptr = &res;
            args.work_amount = ((size + el_per_thread - 1) / el_per_thread) * jit::Generator::xmm_len;
            args.make_64_fold = 1lu;
// args.tmp_ptr = tmp_vec_2.data();
// if (size == 2359296)
//     printf("    [single] work_amount: %lu\n", args.work_amount);
            kernel(&args);
        } else {
// std::vector<uint64_t> tmp_vec(2);

            jit::CombineHashCallArgs args;
            args.src_ptr = src;
            args.dst_ptr = &res;
            args.work_amount = size;
            args.make_64_fold = 1lu;
// args.tmp_ptr = tmp_vec.data();
// if (size > 16 && size < 32)
//     std::cout << "combine_hash size: " << size << std::endl;
            kernel(&args);
        }
// static uint64_t counter = 0lu;
// counter++;
// // // if (counter < 200) {
// if (size == 4) {
//     std::cout << "combine_hash(" << counter << ") kernel res: " << res << "; size: " << size << std::endl;
//     // if (res == 0 || size == 8) {
//         auto src_u8 = reinterpret_cast<const uint8_t *>(src);
//         for (int i = 0; i < size; i++) {
//             std::cout << int(src_u8[i]) << "; ";
//         }
//         std::cout << std::endl;
//     // }
// }

// auto t2 = std::chrono::high_resolution_clock::now();
// auto ms_int = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1);
// sum += ms_int.count();
// if (counter >= 582)
// // if (counter == 1173 || counter == 582)
//     std::cout << "combine_hash sum: " << sum << "; count: " << counter << "; avg_time: " << sum / counter << " nanoseconds" << std::endl;
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
// static uint64_t counter = 0lu;
// if (counter++ < 100)
//     std::cout << "combine_hash ref res: " << seed << "; size: " << size << std::endl;
    return seed;
}

}   // namespace runtime
}   // namespace ov
