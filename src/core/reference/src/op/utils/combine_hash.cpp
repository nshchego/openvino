// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// The CRC computation is used for x86.
// The calculations were taken from the article
// "Fast CRC Computation for Generic Polynomials Using PCLMULQDQ Instruction - Intel (December, 2009)".

#include "openvino/core/visibility.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/reference/utils/combine_hash.hpp"

#if defined(OPENVINO_ARCH_X86) || defined(OPENVINO_ARCH_X86_64)
#    include "openvino/reference/utils/registers_pool.hpp"
#endif // OPENVINO_ARCH_X86 || OPENVINO_ARCH_X86_64

#include <cstring>

namespace ov {
namespace runtime {

#if defined(OPENVINO_ARCH_X86) || defined(OPENVINO_ARCH_X86_64)
namespace jit {

#define GET_OFF(field) offsetof(CombineHashCallArgs, field)
#define getReg64() RegistersPool::Reg<Xbyak::Reg64>(registersPool)
#define getVmm()   RegistersPool::Reg<Vmm>(registersPool)
#define getXmm()   RegistersPool::Reg<Xbyak::Xmm>(registersPool)

constexpr static size_t KERNELS_NUM = 4lu;

enum KernelType {
    SINGLE_THREAD = 0,
    FIRST_THREAD,
    N_THREAD,
    FINAL_FOLD,
    CUSTOM_4B
};

struct CombineHashCompileParams {
    KernelType type = SINGLE_THREAD;
};

struct CombineHashCallArgs {
    const void* src_ptr    = nullptr;
    void* dst_ptr          = nullptr;
    void* intermediate_ptr = nullptr;
    uint64_t work_amount   = 0lu;
    uint64_t threads_num   = 1lu;
void* tmp_ptr; // TODO: remomve
};

typedef void (*fn_t)(const CombineHashCallArgs*);

template <cpu_isa_t isa>
class CombineHash : public Generator {
public:
    explicit CombineHash(const CombineHashCompileParams& jcp) :
            m_jcp(jcp) {
        if (isa == avx512_core) {
printf("[CPU][CombineHash] avx512_core\n");
            vlen = zmm_len;
        } else if (isa == avx2) {
printf("[CPU][CombineHash] avx2\n");
            vlen = ymm_len;
        } else {
            OPENVINO_THROW("Unsupported isa: ", isa);
        }
        if (!mayiuse(cpu_isa_t::pclmulqdq)) {
            OPENVINO_THROW("The current CPU does not support pclmulqdq instruction, which is required for the CRC algorithm.");
        }
        if (mayiuse(cpu_isa_t::vpclmulqdq)) {
printf("[CPU][CombineHash] supports vpclmulqdq\n");
            is_vpclmulqdq = true;
        }

        generate();
    }

    void generate() {
        registersPool = RegistersPool::create(isa, {rax, rcx, rsp, rdi, k0});

        r64_src = getReg64();
        r64_dst = getReg64();
        r64_work_amount  = getReg64();
        // r64_aux = getReg64();
r64_tmp = getReg64();
        // v_dst       = getVmm();
        // v_k_1_2     = getVmm();
        // v_shuf_mask = getVmm();

        this->preamble();

        mov(r64_src, ptr[r64_params + GET_OFF(src_ptr)]);
        mov(r64_dst, ptr[r64_params + GET_OFF(dst_ptr)]);
        mov(r64_work_amount, ptr[r64_params + GET_OFF(work_amount)]);
mov(r64_tmp, ptr[r64_params + GET_OFF(tmp_ptr)]);

        initVectors();
        if (m_jcp.type == SINGLE_THREAD || m_jcp.type == FIRST_THREAD || m_jcp.type == N_THREAD) {
        // if (m_jcp.type == SINGLE_THREAD) {
            bulkFold(v_dst);
        }
        afterBulkFold(v_dst);
        if (m_jcp.type == SINGLE_THREAD || m_jcp.type == FINAL_FOLD) {
            restFold(v_dst);
        }
        if (m_jcp.type == SINGLE_THREAD || m_jcp.type == FINAL_FOLD) {
            tailFold(v_dst);
        }

        this->postamble();
        registersPool.reset();
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
            for (size_t j = 0lu; j < xmm_len - 1; j++) {
                cmp(r64_load_num, j);
                jle(l_end, T_NEAR);
                pinsrb(xmm_dst, ptr[src_addr.getRegExp() + j], j);
            }
        }

        L(l_end);
    }

    // void partialLoad(const Xbyak::Ymm&     ymm_dst,
    //                  const Xbyak::Address& src_addr,
    //                  const Xbyak::Reg64&   r64_load_num) {
    //     Xbyak::Label l_xmm, l_partial, l_end;
    //     auto xmm_dst = Xbyak::Xmm(ymm_dst.getIdx());

    //     cmp(r64_load_num, ymm_len);
    //     jl(l_xmm, T_NEAR);
    //     vmovdqu(ymm_dst, ptr[src_addr.getRegExp()]);
    //     jmp(l_end, T_NEAR);

    //     L(l_xmm);
    //     uni_vpxorq(ymm_dst, ymm_dst, ymm_dst);
    //     cmp(r64_load_num, xmm_len);
    //     jl(l_partial, T_NEAR);
    //     vmovdqu(xmm_dst, ptr[src_addr.getRegExp()]);
    //     je(l_end, T_NEAR);

    //     {
    //         Xbyak::Label l_rest_loop, l_perm;
    //         size_t offset = xmm_len;

    //         vperm2f128(ymm_dst, ymm_dst, ymm_dst, 0x1);
    //         for (size_t j = 0lu; j < xmm_len - 1; j++) {
    //             pinsrb(xmm_dst, ptr[src_addr.getRegExp() + offset], j);
    //             cmp(r64_load_num, ++offset);
    //             jle(l_perm, T_NEAR);
    //         }
    //         L(l_perm);
    //         vperm2f128(ymm_dst, ymm_dst, ymm_dst, 0x1);
    //     }
    //     jmp(l_end, T_NEAR);

    //     L(l_partial); {
    //         size_t offset = xmm_len;

    //         for (size_t j = 0lu; j < xmm_len - 1; j++) {
    //             pinsrb(xmm_dst, ptr[src_addr.getRegExp() + offset], j);
    //             cmp(r64_load_num, ++offset);
    //             jle(l_end, T_NEAR);
    //         }
    //     }

    //     L(l_end);
    // }

private:
    static constexpr uint64_t CHUNK_SIZE = 32;
    static const uint64_t CRC_VAL;
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
    // RegistersPool::Reg<Xbyak::Reg64> r64_bulk_step;
    // RegistersPool::Reg<Xbyak::Reg64> r64_aux;
RegistersPool::Reg<Xbyak::Reg64> r64_tmp;

    const Xbyak::Reg64 r64_params = abi_param1;

    // Vector registers
    RegistersPool::Reg<Vmm> v_dst;
    RegistersPool::Reg<Vmm> v_k_1_2;
    RegistersPool::Reg<Vmm> v_shuf_mask;

    size_t getVlen() {
        return vlen;
    }

    void initVectors();

    void bulkFold(const Vmm& v_dst);

    void afterBulkFold(const Vmm& v_dst);

    void restFold(const Vmm& v_dst);

    void tailFold(const Vmm& v_dst);

    void uni_vpxorq(const Xbyak::Xmm& v_dst, const Xbyak::Xmm& v_src_0, const Xbyak::Xmm& v_src_1);
};

template <>
void CombineHash<avx512_core>::uni_vpxorq(const Xbyak::Xmm& v_dst, const Xbyak::Xmm& v_src_0, const Xbyak::Xmm& v_src_1) {
    vpxorq(v_dst, v_src_0, v_src_1);
}

template <cpu_isa_t isa>
void CombineHash<isa>::uni_vpxorq(const Xbyak::Xmm& v_dst, const Xbyak::Xmm& v_src_0, const Xbyak::Xmm& v_src_1) {
    vpxor(v_dst, v_src_0, v_src_1);
}

// P(x) = 0x42F0E1EBA9EA3693
static const uint64_t K_1_2[]   = { 0x05f5c3c7eb52fab6, 0x4eb938a7d257740e };  // x^(64*1),  x^(64*2)
static const uint64_t K_3_4[]   = { 0x571bee0a227ef92b, 0x44bef2a201b5200c };  // x^(64*3),  x^(64*4)
static const uint64_t K_5_6[]   = { 0x54819d8713758b2c, 0x4a6b90073eb0af5a };  // x^(64*5),  x^(64*6)
static const uint64_t K_7_8[]   = { 0x5f6843ca540df020, 0xddf4b6981205b83f };  // x^(64*7),  x^(64*8)
static const uint64_t K_9_10[]  = { 0x097c516e98bd2e73, 0x0b76477b31e22e7b };  // x^(64*9),  x^(64*10)
static const uint64_t K_11_12[] = { 0x9af04e1eff82d0dd, 0x6e82e609297f8fe8 };  // x^(64*11), x^(64*12)
static const uint64_t K_13_14[] = { 0xe464f4df5fb60ac1, 0xb649c5b35a759cf2 };  // x^(64*13), x^(64*14)
static const uint64_t K_15_16[] = { 0x05cf79dea9ac37d6, 0x001067e571d7d5c2 };  // x^(64*15), x^(64*16)
static const uint64_t K_1_0[]   = { 0x05f5c3c7eb52fab6, 0x0000000000000000 };  // x^(64*1),  x^(64*1) mod P(x)
static const uint64_t K_P_P[]   = { 0x578d29d06cc4f872, 0x42f0e1eba9ea3693 };  // floor(x^128/P(x)) - x^64, P(x) - x^64

template <>
void CombineHash<avx512_core>::initVectors() {
    v_dst = getVmm();
    auto r64_aux = getReg64();

    v_k_1_2 = getVmm();
    mov(r64_aux, reinterpret_cast<uintptr_t>(K_1_2));
    vbroadcasti64x2(v_k_1_2, ptr[r64_aux]);

    v_shuf_mask = getVmm();
    mov(r64_aux, reinterpret_cast<uintptr_t>(SHUF_MASK));
    vbroadcasti64x2(v_shuf_mask, ptr[r64_aux]);

    if (m_jcp.type == SINGLE_THREAD || m_jcp.type == FIRST_THREAD) {
        auto xmm_dst = Xbyak::Xmm(v_dst.getIdx());
        auto xmm_aux = getXmm();
        auto k_rest_mask = RegistersPool::Reg<Xbyak::Opmask>(registersPool);

        // Initial CRC
        mov(r64_aux, CRC_VAL);
        vpinsrq(xmm_aux, xmm_aux, r64_work_amount, 0x0);
        vpinsrq(xmm_aux, xmm_aux, r64_aux, 0x1);
        // Initial xor with source
        fillRestWorkMask(k_rest_mask, r64_work_amount);
        vmovdqu8(Vmm(v_dst.getIdx()) | k_rest_mask | T_z, ptr[r64_src]);
        vpshufb(v_dst, v_dst, v_shuf_mask);
        pxor(xmm_dst, xmm_aux); // The SSE version is used to avoid zeroing out the rest of the Vmm.
        add(r64_src, xmm_len);
    } else if (m_jcp.type == N_THREAD) {
        vmovdqu64(v_dst, ptr[r64_src]);
        vpshufb(v_dst, v_dst, v_shuf_mask);
    }
// vmovdqu64(ptr[r64_tmp], xmm_dst);
    if (m_jcp.type == SINGLE_THREAD || m_jcp.type == FIRST_THREAD || m_jcp.type == N_THREAD) {
        sub(r64_work_amount, xmm_len);
    }
    // add(r64_src, vlen);
}

template <cpu_isa_t isa>
void CombineHash<isa>::initVectors() {
    auto r64_aux = getReg64();

    v_k_1_2 = getVmm();
    mov(r64_aux, reinterpret_cast<uintptr_t>(K_1_2));
    vbroadcasti128(v_k_1_2, ptr[r64_aux]);

    v_shuf_mask = getVmm();
    mov(r64_aux, reinterpret_cast<uintptr_t>(SHUF_MASK));
    vbroadcasti128(v_shuf_mask, ptr[r64_aux]);

    v_dst = getVmm();
    auto xmm_dst = Xbyak::Xmm(v_dst.getIdx());
    auto xmm_shuf_mask = Xbyak::Xmm(v_shuf_mask.getIdx());
    auto xmm_aux = getXmm();
    // Initial CRC
    mov(r64_aux, CRC_VAL);
    uni_vpxorq(v_dst, v_dst, v_dst);
    vpinsrq(xmm_dst, xmm_dst, r64_work_amount, 0x0);
    vpinsrq(xmm_dst, xmm_dst, r64_aux, 0x1);
    // First xor with source
    uni_vpxorq(xmm_aux, xmm_aux, xmm_aux);
    partialLoad(xmm_aux, ptr[r64_src], r64_work_amount);
    vpshufb(xmm_aux, xmm_aux, xmm_shuf_mask);
    uni_vpxorq(xmm_dst, xmm_dst, xmm_aux);
// vmovdqu(ptr[r64_tmp], xmm_dst);
    sub(r64_work_amount, xmm_len);
    add(r64_src, xmm_len);
}

template <>
void CombineHash<avx512_core>::bulkFold(const Vmm& v_dst) {
    Xbyak::Label l_fold_loop, l_end;
    cmp(r64_work_amount, 2 * vlen - xmm_len);
    // cmp(r64_work_amount, 2 * vlen);
    jl(l_end, T_NEAR);

    auto r64_aux = getReg64();

    auto v_src_0 = getVmm();
    auto v_dst_0 = getVmm();
    auto v_dst_1 = getVmm();
    auto v_dst_2 = getVmm();
    auto& v_dst_3 = v_dst;
    auto v_aux_0 = getVmm();
    auto v_k_loop = getVmm();

    auto xmm_k_loop = Xbyak::Xmm(v_k_loop.getIdx());
    auto xmm_k_1_2 = Xbyak::Xmm(v_k_1_2.getIdx());
    auto xmm_src_0 = Xbyak::Xmm(v_src_0.getIdx());
    auto xmm_src_1 = getXmm();
    auto xmm_dst_0 = Xbyak::Xmm(v_dst_0.getIdx());
    auto xmm_dst_1 = Xbyak::Xmm(v_dst_1.getIdx());
    auto xmm_dst_2 = Xbyak::Xmm(v_dst_2.getIdx());
    auto xmm_dst_3 = Xbyak::Xmm(v_dst_3.getIdx());
    auto xmm_aux_0 = Xbyak::Xmm(v_aux_0.getIdx());
    auto xmm_shuf_mask = Xbyak::Xmm(v_shuf_mask.getIdx());

    RegistersPool::Reg<Xbyak::Reg64> r64_bulk_step;
    if (m_jcp.type == FIRST_THREAD || m_jcp.type == N_THREAD) {
        r64_bulk_step = getReg64();
        mov(r64_bulk_step, ptr[r64_params + GET_OFF(threads_num)]);
        // if (is_vpclmulqdq) {
            // sal(r64_bulk_step, 6); // *vlen
        // }
    }

    if (m_jcp.type == SINGLE_THREAD) {
        mov(r64_aux, reinterpret_cast<uintptr_t>(K_7_8));
        vbroadcasti64x2(v_k_loop, ptr[r64_aux]);
    } else {
        mov(r64_aux, reinterpret_cast<uintptr_t>(K_15_16));
        vbroadcasti64x2(v_k_loop, ptr[r64_aux]);
        // ...
    }

    vmovdqu64(v_dst_0, v_dst_3);

    if (!is_vpclmulqdq) {
        vextracti64x2(xmm_dst_1, v_dst, 0x1);
        vextracti64x2(xmm_dst_2, v_dst, 0x2);
        vextracti64x2(xmm_dst_3, v_dst, 0x3);
    }

// auto r64_counter = getReg64();
// mov(r64_counter, vlen);

    if (m_jcp.type == FIRST_THREAD || m_jcp.type == N_THREAD) {
        add(r64_src, r64_bulk_step);
    } else {
        add(r64_src, vlen - xmm_len);
    }
    prefetcht0(ptr[r64_src + 1024]);
    sub(r64_work_amount, 2 * vlen - xmm_len);
    // sub(r64_work_amount, 2 * vlen);

    L(l_fold_loop); {
        vmovdqu64(v_src_0, ptr[r64_src]);
        vpshufb(v_src_0, v_src_0, v_shuf_mask);

        if (m_jcp.type == FIRST_THREAD || m_jcp.type == N_THREAD) {
            add(r64_src, r64_bulk_step);
        } else {
            add(r64_src, vlen);
        }
        prefetcht0(ptr[r64_src + 1024]);

        if (is_vpclmulqdq) {
            vpclmulqdq(v_aux_0, v_dst_0, v_k_loop, 0b00000000);
            vpclmulqdq(v_dst_0, v_dst_0, v_k_loop, 0b00010001);
            uni_vpxorq(v_aux_0, v_aux_0, v_src_0);
            uni_vpxorq(v_dst_0, v_dst_0, v_aux_0);
        } else {
            // prefetchnta(ptr[r64_src + 3 * xmm_len]);
            // prefetchnta(ptr[r64_src + 64]);

            // vmovdqu64(xmm_src_0, ptr[r64_src]);
            // vpshufb(xmm_src_0, xmm_src_0, xmm_shuf_mask);
            // 0
            vpclmulqdq(xmm_aux_0, xmm_dst_0, xmm_k_loop, 0b00000000);
            vpclmulqdq(xmm_dst_0, xmm_dst_0, xmm_k_loop, 0b00010001);
            uni_vpxorq(xmm_aux_0, xmm_aux_0, xmm_src_0);
            uni_vpxorq(xmm_dst_0, xmm_dst_0, xmm_aux_0);

            // 1
            vextracti64x2(xmm_src_1, v_src_0, 0x1);
            // vmovdqu64(xmm_src_1, ptr[r64_src + 1 * xmm_len]);
            // vpshufb(xmm_src_1, xmm_src_1, xmm_shuf_mask);

            vpclmulqdq(xmm_aux_0, xmm_dst_1, xmm_k_loop, 0b00000000);
            vpclmulqdq(xmm_dst_1, xmm_dst_1, xmm_k_loop, 0b00010001);
            uni_vpxorq(xmm_aux_0, xmm_aux_0, xmm_src_1);
            uni_vpxorq(xmm_dst_1, xmm_dst_1, xmm_aux_0);

            // 2
            vextracti64x2(xmm_src_1, v_src_0, 0x2);
            // vmovdqu64(xmm_src_1, ptr[r64_src + 2 * xmm_len]);
            // vpshufb(xmm_src_1, xmm_src_1, xmm_shuf_mask);

            vpclmulqdq(xmm_aux_0, xmm_dst_2, xmm_k_loop, 0b00000000);
            vpclmulqdq(xmm_dst_2, xmm_dst_2, xmm_k_loop, 0b00010001);
            uni_vpxorq(xmm_aux_0, xmm_aux_0, xmm_src_1);
            uni_vpxorq(xmm_dst_2, xmm_dst_2, xmm_aux_0);

            // 3
            vextracti64x2(xmm_src_1, v_src_0, 0x3);
            // vmovdqu64(xmm_src_1, ptr[r64_src + 3 * xmm_len]);
            // vpshufb(xmm_src_1, xmm_src_1, xmm_shuf_mask);

            vpclmulqdq(xmm_aux_0, xmm_dst_3, xmm_k_loop, 0b00000000);
            vpclmulqdq(xmm_dst_3, xmm_dst_3, xmm_k_loop, 0b00010001);
            uni_vpxorq(xmm_aux_0, xmm_aux_0, xmm_src_1);
            uni_vpxorq(xmm_dst_3, xmm_dst_3, xmm_aux_0);
        }

// add(r64_counter, vlen);
        sub(r64_work_amount, vlen);
        jge(l_fold_loop, T_NEAR);
    }
    add(r64_work_amount, vlen);
// mov(ptr[r64_tmp], r64_counter);
// if (m_jcp.type == FIRST_THREAD || m_jcp.type == N_THREAD) {
//     mov(ptr[r64_tmp], r64_bulk_step);
// }

// vmovdqu64(ptr[r64_tmp], xmm_dst_3);

    if (m_jcp.type == SINGLE_THREAD) {
        if (is_vpclmulqdq) {
            auto ymm_dst_0 = Xbyak::Ymm(v_dst_0.getIdx());
            auto ymm_dst_1 = Xbyak::Ymm(v_dst_1.getIdx());
            auto ymm_aux_0 = Xbyak::Ymm(v_aux_0.getIdx());

            vextracti64x4(ymm_dst_1, v_dst_0, 0x1);
            mov(r64_aux, reinterpret_cast<uintptr_t>(K_3_4));
            vpclmulqdq(ymm_aux_0, ymm_dst_0, ptr[r64_aux], 0b00000000);
            vpclmulqdq(ymm_dst_0, ymm_dst_0, ptr[r64_aux], 0b00010001);
            uni_vpxorq(ymm_dst_1, ymm_dst_1, ymm_aux_0);
            uni_vpxorq(ymm_dst_0, ymm_dst_0, ymm_dst_1);

            vextracti64x2(xmm_dst_3, ymm_dst_0, 0x1);
            vpclmulqdq(xmm_aux_0, xmm_dst_0, xmm_k_1_2, 0b00000000);
            vpclmulqdq(xmm_dst_0, xmm_dst_0, xmm_k_1_2, 0b00010001);
            uni_vpxorq(xmm_dst_3, xmm_dst_3, xmm_aux_0);
            uni_vpxorq(xmm_dst_3, xmm_dst_3, xmm_dst_0);
        } else {
            mov(r64_aux, reinterpret_cast<uintptr_t>(K_5_6));
            vpclmulqdq(xmm_aux_0, xmm_dst_0, ptr[r64_aux], 0b00000000);
            vpclmulqdq(xmm_dst_0, xmm_dst_0, ptr[r64_aux], 0b00010001);
            uni_vpxorq(xmm_dst_3, xmm_dst_3, xmm_aux_0);
            uni_vpxorq(xmm_dst_3, xmm_dst_3, xmm_dst_0);

            mov(r64_aux, reinterpret_cast<uintptr_t>(K_3_4));
            vpclmulqdq(xmm_aux_0, xmm_dst_1, ptr[r64_aux], 0b00000000);
            vpclmulqdq(xmm_dst_1, xmm_dst_1, ptr[r64_aux], 0b00010001);
            uni_vpxorq(xmm_dst_3, xmm_dst_3, xmm_aux_0);
            uni_vpxorq(xmm_dst_3, xmm_dst_3, xmm_dst_1);

            vpclmulqdq(xmm_aux_0, xmm_dst_2, xmm_k_1_2, 0b00000000);
            vpclmulqdq(xmm_dst_2, xmm_dst_2, xmm_k_1_2, 0b00010001);
            uni_vpxorq(xmm_dst_3, xmm_dst_3, xmm_aux_0);
            uni_vpxorq(xmm_dst_3, xmm_dst_3, xmm_dst_2);
        }
    } else {
        if (is_vpclmulqdq) {
            vmovdqu64(ptr[r64_dst], v_dst_0);
        } else {
            vmovdqu64(ptr[r64_dst + xmm_len * 0lu], xmm_dst_0);
            vmovdqu64(ptr[r64_dst + xmm_len * 1lu], xmm_dst_1);
            vmovdqu64(ptr[r64_dst + xmm_len * 2lu], xmm_dst_2);
            vmovdqu64(ptr[r64_dst + xmm_len * 3lu], xmm_dst_3);
        }
    }

    L(l_end);
}

template <cpu_isa_t isa>
void CombineHash<isa>::bulkFold(const Vmm& v_dst) {
    Xbyak::Label l_fold_loop, l_end;
    cmp(r64_work_amount, 2 * vlen - xmm_len);
    jl(l_end, T_NEAR);

    auto r64_aux = getReg64();

    auto v_src_0 = getVmm();
    auto v_dst_0 = getVmm();
    // auto v_dst_1 = getVmm();
    // auto v_dst_2 = getVmm();
    auto& v_dst_1 = v_dst;
    auto v_aux_0 = getVmm();
    auto v_k_loop = getVmm();

    auto xmm_k_loop = Xbyak::Xmm(v_k_loop.getIdx());
    auto xmm_k_1_2 = Xbyak::Xmm(v_k_1_2.getIdx());
    auto xmm_src_0 = Xbyak::Xmm(v_src_0.getIdx());
    auto xmm_src_1 = getXmm();
    auto xmm_dst_0 = Xbyak::Xmm(v_dst_0.getIdx());
    auto xmm_dst_1 = Xbyak::Xmm(v_dst_1.getIdx());
    // auto xmm_dst_2 = Xbyak::Xmm(v_dst_2.getIdx());
    // auto xmm_dst_3 = Xbyak::Xmm(v_dst_3.getIdx());
    auto xmm_aux_0 = Xbyak::Xmm(v_aux_0.getIdx());

    mov(r64_aux, reinterpret_cast<uintptr_t>(K_3_4));
    vbroadcasti128(v_k_loop, ptr[r64_aux]);

    vmovdqu(v_dst_0, v_dst_1);

    if (is_vpclmulqdq) {
        vmovdqu(xmm_aux_0, ptr[r64_src]);
        vpshufb(xmm_aux_0, xmm_aux_0, Xbyak::Xmm(v_shuf_mask.getIdx()));
        vinserti128(v_dst_0, v_dst_0, xmm_aux_0, 0x1);
        // vmovdqu(v_dst_1, ptr[r64_src + xmm_len]);
        // vpshufb(v_dst_1, v_dst_1, v_shuf_mask);
        // add(r64_src, vlen + xmm_len);
    } else {
        vmovdqu(xmm_dst_1, ptr[r64_src]);
        vpshufb(xmm_dst_1, xmm_dst_1, Xbyak::Xmm(v_shuf_mask.getIdx()));
    }

    add(r64_src, xmm_len);
    sub(r64_work_amount, 2 * vlen - xmm_len);  // Check

    L(l_fold_loop); {
        vmovdqu(v_src_0, ptr[r64_src]);
        vpshufb(v_src_0, v_src_0, v_shuf_mask);
        add(r64_src, vlen);

        if (is_vpclmulqdq) {
            // vpclmulqdq(v_aux_0, v_dst_0, v_k_loop, 0b00000000);
            // vpclmulqdq(v_dst_0, v_dst_0, v_k_loop, 0b00010001);
            // uni_vpxorq(v_aux_0, v_aux_0, v_src_0);
            // uni_vpxorq(v_dst_0, v_dst_0, v_aux_0);
            // 0

            vpclmulqdq(v_aux_0, v_dst_0, v_k_loop, 0b00000000);
            vpclmulqdq(v_dst_0, v_dst_0, v_k_loop, 0b00010001);
            uni_vpxorq(v_aux_0, v_aux_0, v_src_0);
            uni_vpxorq(v_dst_0, v_dst_0, v_aux_0);

            // // 1
            // vmovdqu(v_src_0, ptr[r64_src]);
            // vpshufb(v_src_0, v_src_0, v_shuf_mask);
            // add(r64_src, vlen);

            // vpclmulqdq(v_aux_0, v_dst_1, v_k_loop, 0b00000000);
            // vpclmulqdq(v_dst_1, v_dst_1, v_k_loop, 0b00010001);
            // uni_vpxorq(v_aux_0, v_aux_0, v_src_1);
            // uni_vpxorq(v_dst_1, v_dst_1, v_aux_0);

            // sub(r64_work_amount, vlen * 2lu);
        } else {
            // 0
            vpclmulqdq(xmm_aux_0, xmm_dst_0, xmm_k_loop, 0b00000000);
            vpclmulqdq(xmm_dst_0, xmm_dst_0, xmm_k_loop, 0b00010001);
            uni_vpxorq(xmm_aux_0, xmm_aux_0, xmm_src_0);
            uni_vpxorq(xmm_dst_0, xmm_dst_0, xmm_aux_0);
            // 1
            vextracti128(xmm_src_1, v_src_0, 0x1);
            vpclmulqdq(xmm_aux_0, xmm_dst_1, xmm_k_loop, 0b00000000);
            vpclmulqdq(xmm_dst_1, xmm_dst_1, xmm_k_loop, 0b00010001);
            uni_vpxorq(xmm_aux_0, xmm_aux_0, xmm_src_1);
            uni_vpxorq(xmm_dst_1, xmm_dst_1, xmm_aux_0);

            // add(r64_src, vlen);
            // sub(r64_work_amount, vlen);
        }

        sub(r64_work_amount, vlen);
        jge(l_fold_loop, T_NEAR);
    }
    add(r64_work_amount, vlen);

// vmovdqu(ptr[r64_tmp], xmm_dst_0);

    if (is_vpclmulqdq) {
        vextracti128(xmm_dst_1, v_dst_0, 0x1);
        vpclmulqdq(xmm_aux_0, xmm_dst_0, xmm_k_1_2, 0b00000000);
        vpclmulqdq(xmm_dst_0, xmm_dst_0, xmm_k_1_2, 0b00010001);
        uni_vpxorq(xmm_dst_1, xmm_dst_1, xmm_aux_0);
        uni_vpxorq(xmm_dst_1, xmm_dst_1, xmm_dst_0);
    } else {
        // vpclmulqdq(xmm_aux_0, xmm_dst_2, xmm_k_1_2, 0b00000000);
        // vpclmulqdq(xmm_dst_2, xmm_dst_2, xmm_k_1_2, 0b00010001);
        // uni_vpxorq(xmm_dst_3, xmm_dst_3, xmm_aux_0);
        // uni_vpxorq(xmm_dst_3, xmm_dst_3, xmm_dst_2);
    }

    L(l_end);
}

template <>
void CombineHash<avx512_core>::afterBulkFold(const Vmm& v_dst) {
    if (m_jcp.type != FINAL_FOLD) {
        return;
    }
    if (is_vpclmulqdq) {
        // auto ymm_dst_0 = Xbyak::Ymm(v_dst_0.getIdx());
        // auto ymm_dst_1 = Xbyak::Ymm(v_dst_1.getIdx());
        // auto ymm_aux_0 = Xbyak::Ymm(v_aux_0.getIdx());

        // vextracti64x4(ymm_dst_1, v_dst_0, 0x1);
        // mov(r64_aux, reinterpret_cast<uintptr_t>(K_3_4));
        // vpclmulqdq(ymm_aux_0, ymm_dst_0, ptr[r64_aux], 0b00000000);
        // vpclmulqdq(ymm_dst_0, ymm_dst_0, ptr[r64_aux], 0b00010001);
        // uni_vpxorq(ymm_dst_1, ymm_dst_1, ymm_aux_0);
        // uni_vpxorq(ymm_dst_0, ymm_dst_0, ymm_dst_1);

        // vextracti64x2(xmm_dst_3, ymm_dst_0, 0x1);
        // vpclmulqdq(xmm_aux_0, xmm_dst_0, xmm_k_1_2, 0b00000000);
        // vpclmulqdq(xmm_dst_0, xmm_dst_0, xmm_k_1_2, 0b00010001);
        // uni_vpxorq(xmm_dst_3, xmm_dst_3, xmm_aux_0);
        // uni_vpxorq(xmm_dst_3, xmm_dst_3, xmm_dst_0);
    } else {
        auto r64_intm = getReg64();
        auto r64_aux  = getReg64();
        mov(r64_intm, ptr[r64_params + GET_OFF(intermediate_ptr)]);
        prefetcht0(ptr[r64_intm + 1024]);

        auto xmm_src_0 = getXmm();
        auto xmm_src_last = Xbyak::Xmm(v_dst.getIdx());
        auto xmm_aux_0 = getXmm();
        auto xmm_k_1_2 = Xbyak::Xmm(v_k_1_2.getIdx());

        vmovdqu64(xmm_src_last, ptr[r64_intm + xmm_len * 7]);

        vmovdqu64(xmm_src_0, ptr[r64_intm]);
        mov(r64_aux, reinterpret_cast<uintptr_t>(K_13_14));
        vpclmulqdq(xmm_aux_0, xmm_src_0, ptr[r64_aux], 0b00000000);
        vpclmulqdq(xmm_src_0, xmm_src_0, ptr[r64_aux], 0b00010001);
        uni_vpxorq(xmm_src_last, xmm_src_last, xmm_aux_0);
        uni_vpxorq(xmm_src_last, xmm_src_last, xmm_src_0);

        vmovdqu64(xmm_src_0, ptr[r64_intm + xmm_len]);
        mov(r64_aux, reinterpret_cast<uintptr_t>(K_11_12));
        vpclmulqdq(xmm_aux_0, xmm_src_0, ptr[r64_aux], 0b00000000);
        vpclmulqdq(xmm_src_0, xmm_src_0, ptr[r64_aux], 0b00010001);
        uni_vpxorq(xmm_src_last, xmm_src_last, xmm_aux_0);
        uni_vpxorq(xmm_src_last, xmm_src_last, xmm_src_0);

        vmovdqu64(xmm_src_0, ptr[r64_intm + xmm_len * 2lu]);
        mov(r64_aux, reinterpret_cast<uintptr_t>(K_9_10));
        vpclmulqdq(xmm_aux_0, xmm_src_0, ptr[r64_aux], 0b00000000);
        vpclmulqdq(xmm_src_0, xmm_src_0, ptr[r64_aux], 0b00010001);
        uni_vpxorq(xmm_src_last, xmm_src_last, xmm_aux_0);
        uni_vpxorq(xmm_src_last, xmm_src_last, xmm_src_0);

        vmovdqu64(xmm_src_0, ptr[r64_intm + xmm_len * 3lu]);
        mov(r64_aux, reinterpret_cast<uintptr_t>(K_7_8));
        vpclmulqdq(xmm_aux_0, xmm_src_0, ptr[r64_aux], 0b00000000);
        vpclmulqdq(xmm_src_0, xmm_src_0, ptr[r64_aux], 0b00010001);
        uni_vpxorq(xmm_src_last, xmm_src_last, xmm_aux_0);
        uni_vpxorq(xmm_src_last, xmm_src_last, xmm_src_0);

        vmovdqu64(xmm_src_0, ptr[r64_intm + xmm_len * 4lu]);
        mov(r64_aux, reinterpret_cast<uintptr_t>(K_5_6));
        vpclmulqdq(xmm_aux_0, xmm_src_0, ptr[r64_aux], 0b00000000);
        vpclmulqdq(xmm_src_0, xmm_src_0, ptr[r64_aux], 0b00010001);
        uni_vpxorq(xmm_src_last, xmm_src_last, xmm_aux_0);
        uni_vpxorq(xmm_src_last, xmm_src_last, xmm_src_0);

        vmovdqu64(xmm_src_0, ptr[r64_intm + xmm_len * 5lu]);
        mov(r64_aux, reinterpret_cast<uintptr_t>(K_3_4));
        vpclmulqdq(xmm_aux_0, xmm_src_0, ptr[r64_aux], 0b00000000);
        vpclmulqdq(xmm_src_0, xmm_src_0, ptr[r64_aux], 0b00010001);
        uni_vpxorq(xmm_src_last, xmm_src_last, xmm_aux_0);
        uni_vpxorq(xmm_src_last, xmm_src_last, xmm_src_0);

        vmovdqu64(xmm_src_0, ptr[r64_intm + xmm_len * 6lu]);
        vpclmulqdq(xmm_aux_0, xmm_src_0, xmm_k_1_2, 0b00000000);
        vpclmulqdq(xmm_src_0, xmm_src_0, xmm_k_1_2, 0b00010001);
        uni_vpxorq(xmm_src_last, xmm_src_last, xmm_aux_0);
        uni_vpxorq(xmm_src_last, xmm_src_last, xmm_src_0);
    }
}

template <cpu_isa_t isa>
void CombineHash<isa>::afterBulkFold(const Vmm& v_dst) {
    if (m_jcp.type != FINAL_FOLD) {
        return;
    }
}

template <>
void CombineHash<avx512_core>::restFold(const Vmm& v_dst) {
    Xbyak::Label l_fold_loop, l_end;
    cmp(r64_work_amount, xmm_len);
    jl(l_end, T_NEAR);

    auto xmm_shuf_mask = Xbyak::Xmm(v_shuf_mask.getIdx());
    auto xmm_k_1_2 = Xbyak::Xmm(v_k_1_2.getIdx());
    auto xmm_src = getXmm();
    auto xmm_dst = Xbyak::Xmm(v_dst.getIdx());
    auto xmm_aux = getXmm();

    L(l_fold_loop); {
        vmovdqu64(xmm_src, ptr[r64_src]); // TODO: compare assembled code with vmovdqu
        vpshufb(xmm_src, xmm_src, xmm_shuf_mask);

        vpclmulqdq(xmm_aux, xmm_dst, xmm_k_1_2, 0b00000000);
        vpclmulqdq(xmm_dst, xmm_dst, xmm_k_1_2, 0b00010001);
        uni_vpxorq(xmm_dst, xmm_dst, xmm_aux);
        uni_vpxorq(xmm_dst, xmm_dst, xmm_src);

        add(r64_src, xmm_len);
        sub(r64_work_amount, xmm_len);
        cmp(r64_work_amount, xmm_len);
        jge(l_fold_loop, T_NEAR);
    }

    L(l_end);
}

template <cpu_isa_t isa>
void CombineHash<isa>::restFold(const Vmm& v_dst) {
    Xbyak::Label l_fold_loop, l_end;
    cmp(r64_work_amount, xmm_len);
    jl(l_end, T_NEAR);

    auto xmm_shuf_mask = Xbyak::Xmm(v_shuf_mask.getIdx());
    auto xmm_k_1_2 = Xbyak::Xmm(v_k_1_2.getIdx());
    auto xmm_src = getXmm();
    auto xmm_dst = Xbyak::Xmm(v_dst.getIdx());
    auto xmm_aux = getXmm();

    L(l_fold_loop); {
        vmovdqu(xmm_src, ptr[r64_src]);
        vpshufb(xmm_src, xmm_src, xmm_shuf_mask);

        vpclmulqdq(xmm_aux, xmm_dst, xmm_k_1_2, 0b00000000);
        vpclmulqdq(xmm_dst, xmm_dst, xmm_k_1_2, 0b00010001);
        uni_vpxorq(xmm_dst, xmm_dst, xmm_aux);
        uni_vpxorq(xmm_dst, xmm_dst, xmm_src);

        add(r64_src, xmm_len);
        sub(r64_work_amount, xmm_len);
        cmp(r64_work_amount, xmm_len);
        jge(l_fold_loop, T_NEAR);
    }

    L(l_end);
}

template <>
void CombineHash<avx512_core>::tailFold(const Vmm& v_dst) {
// vmovdqu64(ptr[r64_tmp], Xbyak::Xmm(v_dst.getIdx()));
    Xbyak::Label l_fold_to_64;
    cmp(r64_work_amount, 0);
    jle(l_fold_to_64, T_NEAR);

    auto r64_aux = getReg64();
    auto xmm_shuf_mask = Xbyak::Xmm(v_shuf_mask.getIdx());
    auto xmm_k_1_2 = Xbyak::Xmm(v_k_1_2.getIdx());
    auto xmm_src = getXmm();
    auto xmm_dst = Xbyak::Xmm(v_dst.getIdx());
    auto xmm_aux = getXmm();
    auto xmm_aux_1 = getXmm();
    auto xmm_aux_2 = getXmm();
    auto k_rest_mask = RegistersPool::Reg<Xbyak::Opmask>(registersPool);

    fillRestWorkMask(k_rest_mask, r64_work_amount);
    vmovdqu8(Xbyak::Xmm(xmm_src.getIdx()) | k_rest_mask | T_z, ptr[r64_src]);
    vpshufb(xmm_src, xmm_src, xmm_shuf_mask);

    vpclmulqdq(xmm_aux, xmm_dst, xmm_k_1_2, 0b00000000);
    vpclmulqdq(xmm_dst, xmm_dst, xmm_k_1_2, 0b00010001);
    uni_vpxorq(xmm_aux, xmm_aux, xmm_src);
    uni_vpxorq(xmm_dst, xmm_dst, xmm_aux);

    L(l_fold_to_64);

    mov(r64_aux, reinterpret_cast<uintptr_t>(K_1_0));
    vpclmulqdq(xmm_aux, xmm_dst, ptr[r64_aux], 0b00000001);
    vpslldq(xmm_dst, xmm_dst, 0x8);
    uni_vpxorq(xmm_dst, xmm_dst, xmm_aux);

    mov(r64_aux, reinterpret_cast<uintptr_t>(K_P_P));
    vmovdqu64(xmm_aux_2, ptr[r64_aux]);
    vpclmulqdq(xmm_aux, xmm_dst, xmm_aux_2, 0b00000001);
    mov(r64_aux, 0x0);
    vpinsrq(xmm_aux_1, xmm_dst, r64_aux, 0x0);
    uni_vpxorq(xmm_aux, xmm_aux, xmm_aux_1);
    vpinsrq(xmm_aux_1, xmm_aux, r64_aux, 0x0);
    vpclmulqdq(xmm_aux, xmm_aux, xmm_aux_2, 0b00010001);
    uni_vpxorq(xmm_aux, xmm_aux, xmm_aux_1);
    uni_vpxorq(xmm_dst, xmm_dst, xmm_aux);

    vpextrq(ptr[r64_dst], xmm_dst, 0x0);
// vmovdqu64(ptr[r64_tmp], xmm_dst);
}

template <cpu_isa_t isa>
void CombineHash<isa>::tailFold(const Vmm& v_dst) {
// vmovdqu(ptr[r64_tmp], Xbyak::Xmm(v_dst.getIdx()));
    Xbyak::Label l_fold_to_64, l_save_128, l_end;
    cmp(r64_work_amount, 0);
    jle(l_fold_to_64, T_NEAR);

    auto r64_aux = getReg64();
    auto xmm_shuf_mask = Xbyak::Xmm(v_shuf_mask.getIdx());
    auto xmm_k_1_2 = Xbyak::Xmm(v_k_1_2.getIdx());
    auto xmm_src = getXmm();
    auto xmm_dst = Xbyak::Xmm(v_dst.getIdx());
    auto xmm_aux = getXmm();
    auto xmm_aux_1 = getXmm();
    auto xmm_aux_2 = getXmm();
// vmovdqu(ptr[r64_tmp], xmm_dst);

    uni_vpxorq(xmm_src, xmm_src, xmm_src);
    partialLoad(xmm_src, ptr[r64_src], r64_work_amount);
    vpshufb(xmm_src, xmm_src, xmm_shuf_mask);

    vpclmulqdq(xmm_aux, xmm_dst, xmm_k_1_2, 0b00000000);
    vpclmulqdq(xmm_dst, xmm_dst, xmm_k_1_2, 0b00010001);
    uni_vpxorq(xmm_aux, xmm_aux, xmm_src);
    uni_vpxorq(xmm_dst, xmm_dst, xmm_aux);

    L(l_fold_to_64);
    cmp(r64_make_64_fold, 0);
    je(l_save_128, T_NEAR);

    mov(r64_aux, reinterpret_cast<uintptr_t>(K_1_0));
    vpclmulqdq(xmm_aux, xmm_dst, ptr[r64_aux], 0b00000001);
    vpslldq(xmm_dst, xmm_dst, 0x8);
    uni_vpxorq(xmm_dst, xmm_dst, xmm_aux);

    mov(r64_aux, reinterpret_cast<uintptr_t>(K_P_P));
    vmovdqu(xmm_aux_2, ptr[r64_aux]);
    vpclmulqdq(xmm_aux, xmm_dst, xmm_aux_2, 0b00000001);
// vmovdqu(ptr[r64_tmp], xmm_aux);
    mov(r64_aux, 0x0);
    vpinsrq(xmm_aux_1, xmm_dst, r64_aux, 0x0);
    uni_vpxorq(xmm_aux, xmm_aux, xmm_aux_1);
    vpinsrq(xmm_aux_1, xmm_aux, r64_aux, 0x0);
    vpclmulqdq(xmm_aux, xmm_aux, xmm_aux_2, 0b00010001);
    uni_vpxorq(xmm_aux, xmm_aux, xmm_aux_1);
    uni_vpxorq(xmm_dst, xmm_dst, xmm_aux);

    vpextrq(ptr[r64_dst], xmm_dst, 0x0);
    jmp(l_end, T_NEAR);

    L(l_save_128);
    vmovdqu(ptr[r64_dst], xmm_dst);

    L(l_end);
// vmovdqu(ptr[r64_tmp], xmm_dst);
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
const uint8_t CombineHash<isa>::SHUF_MASK[] = { 0b00001111, 0b00001110, 0b00001101, 0b00001100, 0b00001011, 0b00001010, 0b00001001, 0b00001000,
                                                0b00000111, 0b00000110, 0b00000101, 0b00000100, 0b00000011, 0b00000010, 0b00000001, 0b00000000 };

} // namespace jit
#endif // OPENVINO_ARCH_X86 || OPENVINO_ARCH_X86_64

size_t combine_hash(const void* src, size_t size) {
// if (size > 200000)
    printf("combine_hash size: %lu\n", size);
static uint64_t counter = 0;
static uint64_t sum = 0;
counter++;
auto t1 = std::chrono::high_resolution_clock::now();
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
    static jit::fn_t kernels[jit::KERNELS_NUM] = {nullptr, nullptr, nullptr, nullptr};

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
    if (kernels[0] == nullptr) {
printf("    init kernels\n");
        if (jit::Generator::mayiuse(jit::avx512_core)) {
            parallel_nt(jit::KERNELS_NUM, [&](const int ithr, const int nthr) {
                jit::CombineHashCompileParams params;
                switch (ithr) {
                    case 0: {
                        params.type = jit::SINGLE_THREAD;
                        static jit::CombineHash<jit::avx512_core> kernel(params);
                        kernels[jit::SINGLE_THREAD] = (jit::fn_t)kernel.getCode();
                    } break;
                    case 1: {
                        params.type = jit::FIRST_THREAD;
                        static jit::CombineHash<jit::avx512_core> kernel(params);
                        kernels[jit::FIRST_THREAD] = (jit::fn_t)kernel.getCode();
                    } break;
                    case 2: {
                        params.type = jit::N_THREAD;
                        static jit::CombineHash<jit::avx512_core> kernel(params);
                        kernels[jit::N_THREAD] = (jit::fn_t)kernel.getCode();
                    } break;
                    case 3: {
                        params.type = jit::FINAL_FOLD;
                        static jit::CombineHash<jit::avx512_core> kernel(params);
                        kernels[jit::FINAL_FOLD] = (jit::fn_t)kernel.getCode();
                    } break;
                    default:
                        OPENVINO_THROW("[ CORE ] Combine hash. Unexpected thread index: ", ithr);
                }
            });
            for (size_t i = 0lu; i < jit::KERNELS_NUM; i++) {
                if (kernels[i] == nullptr) {
                    OPENVINO_THROW("[ CORE ] Combine hash. Kernel #", i, " was not initialized.");
                }
            }
        } else if (jit::Generator::mayiuse(jit::avx2)) {
            // kernel = jit::CombineHash<jit::avx2>::get();
        }
    }

    if (kernels[0]) {
        size_t res = 0lu;

        static const size_t block_size = 2lu * jit::Generator::zmm_len; //kernels[0]->getVlen(); // TODO: vlen
        if (size >= 200000lu) {
// printf("    parallel_get_max_threads : %d\n", parallel_get_max_threads());
            // static const size_t thr_num = parallel_get_max_threads() > 1 ? parallel_get_max_threads() / 2 : 1lu; // TODO: WA for Hyper Threading
            const size_t thr_num = 2lu;
            static std::vector<uint64_t> intermediate(thr_num * 8lu); // zmm_len * thr_num
            const uint64_t blocks = size / block_size;
            const uint64_t el_per_thread = block_size * ((blocks + thr_num - 1) / thr_num);
std::vector<uint64_t> tmp_vec(thr_num * 4, 0lu);
std::vector<uint64_t> tmp_vec_2(thr_num * 4, 0lu);

            parallel_nt(thr_num, [&](const int ithr, const int nthr) {
            // parallel_nt(36, [&](int ithr, const int nthr) {
            //     if (ithr != 0 && ithr != 18) {
            //         return;
            //     }
            //     ithr /= 18;
                // if (ithr > 0)
                //     return;

                uint64_t start = ithr * el_per_thread;
                if (start >= size) {
                    return;
                }
                uint64_t work_amount = (el_per_thread + start > size) ? size - start : el_per_thread;
// printf("    [%d] start: %lu, work_amount: %lu\n", ithr, start, work_amount);

                jit::CombineHashCallArgs args;

                args.src_ptr = reinterpret_cast<const uint8_t *>(src) + jit::Generator::xmm_len * 4lu * ithr;
                args.dst_ptr = &(intermediate[ithr * 8lu]);
                args.work_amount = work_amount;
                args.threads_num = jit::Generator::xmm_len * 8lu; //thr_num;
args.tmp_ptr = &(tmp_vec[ithr * 4]);

                kernels[ithr == 0 ? jit::FIRST_THREAD : jit::N_THREAD](&args);
                // kernels[jit::FIRST_THREAD](&args);
// printf("    [%d] start: %lu, work_amount: %lu; made: %lu\n", ithr, start, work_amount, tmp_vec[ithr * 4]);
            });

//             jit::CombineHashCallArgs args;

//             args.src_ptr = reinterpret_cast<const uint8_t *>(src);
//             args.dst_ptr = &(intermediate[0]);
//             args.work_amount = el_per_thread;
//             args.threads_num = jit::Generator::xmm_len * 8lu; //thr_num;
// args.tmp_ptr = &(tmp_vec[0]);

//             kernels[jit::FIRST_THREAD](&args);

            jit::CombineHashCallArgs args;
            args.work_amount = size - el_per_thread * thr_num; //((size + el_per_thread - 1) / el_per_thread) * jit::Generator::xmm_len;
            args.src_ptr = reinterpret_cast<const uint8_t *>(src) + size - args.work_amount;
            args.dst_ptr = &res;
            args.intermediate_ptr = intermediate.data();
args.tmp_ptr = tmp_vec_2.data();
// printf("    FINAL_FOLD work_amount: %lu\n", args.work_amount);

            kernels[jit::FINAL_FOLD](&args);
        } else {
std::vector<uint64_t> tmp_vec(4, 0lu);

            jit::CombineHashCallArgs args;
            args.src_ptr = src;
            args.dst_ptr = &res;
            args.work_amount = size;
args.tmp_ptr = &(tmp_vec[0]);

            kernels[jit::SINGLE_THREAD](&args);
        }

auto t2 = std::chrono::high_resolution_clock::now();
auto ms_int = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1);
sum += ms_int.count();
if (size > 200000)
    std::cout << "[" << counter << "] combine_hash time: " << ms_int.count() << "; sum: " << sum << "; size: " << size << "; avg_time: " << sum / counter << " nanosec" << std::endl;
printf("    res: %lu\n", res);
        return res;
    }
//     if (kernel) {
//         size_t res = 0lu;

//         static const size_t block_size = 2lu * jit::Generator::zmm_len; // TODO: vlen
//         // There is no sense to perform parallel execution if there are less than 2 blocks.
//         // if (size >= 2lu * block_size) {
//         if (size >= 20000000lu) {
//             // static const auto nthr = parallel_get_max_threads() / 2; // TODO: WA for Hyper Threading
//             static const auto nthr = 1lu;
//             static std::vector<uint64_t> intermediate(nthr * 2); // xmm_len * nthr
//             const uint64_t blocks = size / block_size;
//             const uint64_t el_per_thread = block_size * ((blocks + nthr - 1) / nthr);

// std::vector<uint64_t> tmp_vec(nthr * 4);
// std::vector<uint64_t> tmp_vec_2(nthr * 4);

// // if (!(counter == 104)) {
// // if (!(counter == 88 || counter == 92 || counter == 96 || counter == 100 || counter == 104 || counter == 108)) {
//             parallel_nt(nthr, [&](const int ithr, const int nthr) {
//                 uint64_t start = ithr * el_per_thread;
//                 if (start >= size) {
//                     return;
//                 }
//                 uint64_t work_amount = (el_per_thread + start > size) ? size - start : el_per_thread;

//                 jit::CombineHashCallArgs args;

//                 args.src_ptr = reinterpret_cast<const uint8_t *>(src) + start;
//                 args.dst_ptr = &intermediate[ithr * 2];
//                 args.work_amount = work_amount;
//                 args.make_64_fold = 0lu;
// args.tmp_ptr = &(tmp_vec[ithr * 4]);

//                 kernel(&args);

// // if (counter == 8)
// //     printf("    [%d] start: %lu, work_amount: %lu\n", ithr, start, work_amount);
// // printf("    Parallel fold: %lu; tmp_vec {%lu; %lu; %lu; %lu}\n",
// //     size, tmp_vec[ithr * 4], tmp_vec[ithr * 4 + 1], tmp_vec[ithr * 4 + 2], tmp_vec[ithr * 4 + 3]);
//             });
// // } else {
// //     for (int ithr = 0; ithr < nthr; ithr++) {
// //         uint64_t start = ithr * el_per_thread;
// //         if (start >= size) {
// //             continue;
// //         }
// //         uint64_t work_amount = (el_per_thread + start > size) ? size - start : el_per_thread;

// //         size_t res = 0lu;
// //         jit::CombineHashCallArgs args;

// //         args.src_ptr = reinterpret_cast<const uint8_t *>(src) + start;
// //         args.dst_ptr = &(intermediate[ithr * 2]);
// //         args.work_amount = work_amount;
// //         args.make_64_fold = 0lu;
// // args.tmp_ptr = &(tmp_vec[ithr * 2]);
// //         kernel(&args);
// //     }
// // }

// // if (counter == 88 || counter == 92 || counter == 96 || counter == 100 || counter == 104 || counter == 108) {
// //     std::cout << "Combine hash " << counter << " Hash: " ;
// //     for (int i = 0; i < intermediate.size(); i++) {
// //         std::cout << intermediate[i] << "; ";
// //     }
// //     std::cout << std::endl << "    tmp vals: ";
// //     for (int i = 0; i < tmp_vec.size(); i++) {
// //         std::cout << tmp_vec[i] << "; ";
// //     }
// //     std::cout << std::endl;

// // //     auto data = reinterpret_cast<const uint8_t *>(src);// + 131072;
// // //     for (int i = 0; i < 131072; i++) {
// // //         std::cout << static_cast<uint32_t>(data[i]) << std::endl;
// // //     }
// // }

//             jit::CombineHashCallArgs args;
//             args.src_ptr = intermediate.data();
//             args.dst_ptr = &res;
//             args.work_amount = ((size + el_per_thread - 1) / el_per_thread) * jit::Generator::xmm_len;
//             args.make_64_fold = 1lu;
// args.tmp_ptr = tmp_vec_2.data();

//             kernel(&args);
            
// // if (size == 2359296)
// //     printf("    [single] work_amount: %lu\n", args.work_amount);
// // printf("    Final fold: %lu; tmp_vec {%lu; %lu; %lu; %lu}\n", size, tmp_vec_2[0], tmp_vec_2[1], tmp_vec_2[2], tmp_vec_2[3]);
//         } else {
// std::vector<uint64_t> tmp_vec(4, 0lu);

//             jit::CombineHashCallArgs args;
//             args.src_ptr = src;
//             args.dst_ptr = &res;
//             args.work_amount = size;
//             args.make_64_fold = 1lu;
// args.tmp_ptr = &(tmp_vec[0]);

//             kernel(&args);

// // if (size > 200000lu) {
//     // std::cout << "combine_hash size: " << size << "; tmp_vec: {" << tmp_vec[0] << "; " << tmp_vec[1] << "}" << std::endl;
//     // if (size == 4) {
//         // std::cout << "    Seq size: " << size << "; src: {" << reinterpret_cast<const int*>(src)[0]
//         //     << "} tmp_vec: {" << tmp_vec[0] << "; " << tmp_vec[1] << "; " << tmp_vec[2] << "; " << tmp_vec[3] << "}" << std::endl;
//     // }
// // }
//         }
// // static uint64_t counter = 0lu;
// // counter++;
// // // // if (counter < 200) {
// // if (size == 4) {
// //     std::cout << "combine_hash(" << counter << ") kernel res: " << res << "; size: " << size << std::endl;
// //     // if (res == 0 || size == 8) {
// //         auto src_u8 = reinterpret_cast<const uint8_t *>(src);
// //         for (int i = 0; i < size; i++) {
// //             std::cout << int(src_u8[i]) << "; ";
// //         }
// //         std::cout << std::endl;
// //     // }
// // }

// auto t2 = std::chrono::high_resolution_clock::now();
// auto ms_int = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1);
// sum += ms_int.count();
// // if (counter == 1 || counter == 8 || counter == 557 || counter == 564)
// // // if (size >= 100000 && size <= 200000)
// if (size > 200000)
//     std::cout << "[" << counter << "] combine_hash time: " << ms_int.count() << "; sum: " << sum << "; size: " << size << "; avg_time: " << sum / counter << " nanosec" << std::endl;
//     // std::cout << ms_int.count() << std::endl;
// printf("    res: %lu\n", res);
//         return res;
//     }
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
