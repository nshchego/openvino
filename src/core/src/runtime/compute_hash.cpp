// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// The CRC computation is used for x86.
// The calculations were taken from the article
// "Fast CRC Computation for Generic Polynomials Using PCLMULQDQ Instruction - Intel (December, 2009)".

#include "openvino/core/visibility.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/runtime/compute_hash.hpp"

#if defined(OPENVINO_ARCH_X86) || defined(OPENVINO_ARCH_X86_64)
#    include "openvino/reference/utils/registers_pool.hpp"
#endif // OPENVINO_ARCH_X86 || OPENVINO_ARCH_X86_64

#include <cstring>
#include <inttypes.h>

using namespace ov::reference::jit;

namespace ov {
namespace runtime {

#if defined(OPENVINO_ARCH_X86) || defined(OPENVINO_ARCH_X86_64)
namespace jit {

#define GET_OFF(field) offsetof(ComputeHashCallArgs, field)
#define getReg64() RegistersPool::Reg<Xbyak::Reg64>(m_registers_pool)
#define getVmm()   RegistersPool::Reg<Vmm>(m_registers_pool)
#define getXmm()   RegistersPool::Reg<Xbyak::Xmm>(m_registers_pool)

enum KernelType {
    SINGLE_THREAD = 0,
    FIRST_THREAD,
    N_THREAD,
    FINAL_FOLD
};

struct ComputeHashCompileParams {
    KernelType type;
};

struct ComputeHashCallArgs {
    const void* src_ptr    = nullptr;
    void* dst_ptr          = nullptr;
    const void* k_ptr      = nullptr;
    void* intermediate_ptr = nullptr;
    uint64_t work_amount   = 0lu;
    uint64_t size          = 0lu;
    uint64_t threads_num   = 1lu;
void* tmp_ptr; // TODO: remomve
};

typedef void (*hash_kernel)(const ComputeHashCallArgs*);

static const uint8_t SHUF_MASK[16] = { 0b00001111, 0b00001110, 0b00001101, 0b00001100, 0b00001011, 0b00001010, 0b00001001, 0b00001000,
                                       0b00000111, 0b00000110, 0b00000101, 0b00000100, 0b00000011, 0b00000010, 0b00000001, 0b00000000 };

constexpr uint64_t CRC_VAL = 0xffffffffffffffff;

// POLYNOM(x) = 0x42F0E1EBA9EA3693
constexpr uint64_t K_2 = 0x05f5c3c7eb52fab6;
constexpr uint64_t P_1 = 0x578d29d06cc4f872;
constexpr uint64_t P_2 = 0x42f0e1eba9ea3693;
static const uint64_t K_PULL[] = {
        K_2,                0x0000000000000000,  // x^(64*2),  x^(64*1) mod P(x)
        P_1,                P_2,                 // floor(x^128/P(x))-x^64, P(x)-x^64
        K_2,                0x4eb938a7d257740e,  // x^(64*2),  x^(64*3)
        0x571bee0a227ef92b, 0x44bef2a201b5200c,  // x^(64*4),  x^(64*5)
        0x54819d8713758b2c, 0x4a6b90073eb0af5a,  // x^(64*6),  x^(64*7)
        0x5f6843ca540df020, 0xddf4b6981205b83f,  // x^(64*8),  x^(64*9)
        0x097c516e98bd2e73, 0x0b76477b31e22e7b,  // x^(64*10), x^(64*11)
        0x9af04e1eff82d0dd, 0x6e82e609297f8fe8,  // x^(64*12), x^(64*13)
        0xe464f4df5fb60ac1, 0xb649c5b35a759cf2,  // x^(64*14), x^(64*14)
        0x05cf79dea9ac37d6, 0x001067e571d7d5c2   // x^(64*16), x^(64*17)
    };

constexpr uint64_t K_1_0_OFF   = 0lu * 2lu * sizeof(uint64_t);
constexpr uint64_t K_P_P_OFF   = 1lu * 2lu * sizeof(uint64_t);
constexpr uint64_t K_2_3_OFF   = 2lu * 2lu * sizeof(uint64_t);
constexpr uint64_t K_4_5_OFF   = 3lu * 2lu * sizeof(uint64_t);
constexpr uint64_t K_6_7_OFF   = 4lu * 2lu * sizeof(uint64_t);
constexpr uint64_t K_8_9_OFF   = 5lu * 2lu * sizeof(uint64_t);
constexpr uint64_t K_10_11_OFF = 6lu * 2lu * sizeof(uint64_t);
constexpr uint64_t K_12_13_OFF = 7lu * 2lu * sizeof(uint64_t);
constexpr uint64_t K_14_15_OFF = 8lu * 2lu * sizeof(uint64_t);
constexpr uint64_t K_16_17_OFF = 9lu * 2lu * sizeof(uint64_t);

// static const uint64_t K_2_3[]   = { K_2, 0x4eb938a7d257740e };  // x^(64*2),  x^(64*3)
// static const uint64_t K_3_4[]   = { 0x571bee0a227ef92b, 0x44bef2a201b5200c };  // x^(64*4),  x^(64*5)
// static const uint64_t K_5_6[]   = { 0x54819d8713758b2c, 0x4a6b90073eb0af5a };  // x^(64*6),  x^(64*7)
// static const uint64_t K_7_8[]   = { 0x5f6843ca540df020, 0xddf4b6981205b83f };  // x^(64*8),  x^(64*9)
// static const uint64_t K_9_10[]  = { 0x097c516e98bd2e73, 0x0b76477b31e22e7b };  // x^(64*10), x^(64*11)
// static const uint64_t K_11_12[] = { 0x9af04e1eff82d0dd, 0x6e82e609297f8fe8 };  // x^(64*12), x^(64*13)
// static const uint64_t K_13_14[] = { 0xe464f4df5fb60ac1, 0xb649c5b35a759cf2 };  // x^(64*14), x^(64*14)
// static const uint64_t K_15_16[] = { 0x05cf79dea9ac37d6, 0x001067e571d7d5c2 };  // x^(64*16), x^(64*17)
// static const uint64_t K_1_0[]   = { K_2, 0x0000000000000000 };  // x^(64*1),  x^(64*1) mod P(x)
// static const uint64_t K_P_P[]   = { P_1, P_2 };  // floor(x^128/P(x)) - x^64, P(x) - x^64

class HashBase : public Generator {
protected:
    void (*ker_fn)(const ComputeHashCallArgs*);
public:
    virtual void generate() = 0;

    void operator()(const ComputeHashCallArgs* args) {
        ker_fn(args);
    }

    virtual void create_kernel() {
        generate();
        ker_fn = (decltype(ker_fn))getCode();
        OPENVINO_ASSERT(ker_fn, "[ CORE ] Could not generate kernel code.");
    }
};

template <cpu_isa_t isa>
class ComputeHash : public HashBase {
public:
    explicit ComputeHash(const ComputeHashCompileParams& jcp) :
            m_jcp(jcp) {
        if (isa == avx512_core) {
printf("[CPU][ComputeHash] avx512_core\n");
            vlen = zmm_len;
        } else if (isa == avx2) {
printf("[CPU][ComputeHash] avx2\n");
            vlen = ymm_len;
        } else {
            OPENVINO_THROW("Unsupported isa: ", isa);
        }
        if (!mayiuse(cpu_isa_t::pclmulqdq)) {
            OPENVINO_THROW("The current CPU does not support pclmulqdq instruction, which is required for the CRC algorithm.");
        }
        if (mayiuse(cpu_isa_t::vpclmulqdq)) {
printf("[CPU][ComputeHash] supports vpclmulqdq\n");
            is_vpclmulqdq = true;
        }
    }

    void generate() override {
// printf("ComputeHash generate %d\n", static_cast<int>(m_jcp.type));
        m_registers_pool = RegistersPool::create(isa, {rax, rcx, rsp, rdi, k0});

        r64_src = getReg64();
        r64_dst = getReg64();
        r64_work_amount = getReg64();
        r64_k_ptr = getReg64();
        // r64_aux = getReg64();
r64_tmp = getReg64();
        // v_dst       = getVmm();
        // v_k_2_3     = getVmm();
        // v_shuf_mask = getVmm();

        this->preamble();

        mov(r64_src, ptr[r64_params + GET_OFF(src_ptr)]);
        mov(r64_dst, ptr[r64_params + GET_OFF(dst_ptr)]);
        mov(r64_work_amount, ptr[r64_params + GET_OFF(work_amount)]);
        mov(r64_k_ptr, ptr[r64_params + GET_OFF(k_ptr)]);
mov(r64_tmp, ptr[r64_params + GET_OFF(tmp_ptr)]);

        initialize();
        if (m_jcp.type == SINGLE_THREAD || m_jcp.type == FIRST_THREAD || m_jcp.type == N_THREAD) {
            bulk_fold(v_dst);
        }
        after_bulk_fold(v_dst);
        if (m_jcp.type == SINGLE_THREAD || m_jcp.type == FINAL_FOLD) {
            rest_fold(v_dst);
        }
        if (m_jcp.type == SINGLE_THREAD || m_jcp.type == FINAL_FOLD) {
            tail_fold(v_dst);
        }

        this->postamble();
        m_registers_pool.reset();
    }

    static std::shared_ptr<HashBase> create(const ComputeHashCompileParams& params) {
        auto kernel = std::make_shared<ComputeHash>(params);
        OPENVINO_ASSERT(kernel, "[ CORE ] Could not create ComputeHash kernel.");
        kernel->create_kernel();

        return kernel;
    }

    void fill_rest_work_mask(const Xbyak::Opmask& k_dst_mask,
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

    void partial_load(const Xbyak::Xmm&     xmm_dst,
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

    // void partial_load(const Xbyak::Ymm&     ymm_dst,
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
    // static const uint64_t CRC_VAL;
    // static const uint8_t SHUF_MASK[16];

    using Vmm = typename std::conditional<isa == avx512_core, Xbyak::Zmm, Xbyak::Ymm>::type;
    size_t vlen = xmm_len;
    bool is_vpclmulqdq = false;

    ComputeHashCompileParams m_jcp;
    RegistersPool::Ptr m_registers_pool;

    RegistersPool::Reg<Xbyak::Reg64> r64_src;
    RegistersPool::Reg<Xbyak::Reg64> r64_dst;
    RegistersPool::Reg<Xbyak::Reg64> r64_work_amount;
    RegistersPool::Reg<Xbyak::Reg64> r64_make_64_fold;
    RegistersPool::Reg<Xbyak::Reg64> r64_k_ptr;
    // RegistersPool::Reg<Xbyak::Reg64> r64_bulk_step;
    // RegistersPool::Reg<Xbyak::Reg64> r64_aux;
RegistersPool::Reg<Xbyak::Reg64> r64_tmp;

    const Xbyak::Reg64 r64_params = abi_param1;

    // Vector registers
    RegistersPool::Reg<Vmm> v_dst;
    RegistersPool::Reg<Vmm> v_k_2_3;
    RegistersPool::Reg<Vmm> v_shuf_mask;

    size_t get_vlen() {
        return vlen;
    }

    void initialize();

    void bulk_fold(const Vmm& v_dst);

    void after_bulk_fold(const Vmm& v_dst);

    void rest_fold(const Vmm& v_dst);

    void tail_fold(const Vmm& v_dst);

    void uni_vpxorq(const Xbyak::Xmm& v_dst, const Xbyak::Xmm& v_src_0, const Xbyak::Xmm& v_src_1);
};

class ComputeHash4 : public Generator {
public:
    explicit ComputeHash4(const ComputeHashCompileParams& jcp) {
        if (!mayiuse(cpu_isa_t::pclmulqdq)) {
            OPENVINO_THROW("The current CPU does not support pclmulqdq instruction, which is required for the CRC algorithm.");
        }

        generate();
    }

    void generate() {
        const auto& r64_params = abi_param1;

        const auto& r64_src   = r8;
        const auto& r64_dst   = r9;
        const auto& r64_aux   = r10;

        const auto& xmm_dst   = xmm0;
        const auto& xmm_aux   = xmm1;
        const auto& xmm_aux_1 = xmm2;
        const auto& xmm_aux_2 = xmm3;

        this->preamble();

        mov(r64_src, ptr[r64_params + GET_OFF(src_ptr)]);
        mov(r64_dst, ptr[r64_params + GET_OFF(dst_ptr)]);

        mov(r64_aux, 4);
        vpinsrq(xmm_aux, xmm_aux, r64_aux, 0x0);
        mov(r64_aux, CRC_VAL);
        vpinsrq(xmm_aux, xmm_aux, r64_aux, 0x1);

        vpxor(xmm_dst, xmm_dst, xmm_dst);
        // mov(r64_aux, ptr[r64_params + GET_OFF(src_ptr)]);
        vpinsrd(xmm_dst, xmm_dst, ptr[r64_src], 0x0);
        mov(r64_aux, reinterpret_cast<uintptr_t>(SHUF_MASK));
        vpshufb(xmm_dst, xmm_dst, ptr[r64_aux]);

        vpxor(xmm_dst, xmm_dst, xmm_aux);

        mov(r64_aux, K_2);
        vpinsrq(xmm_aux_2, xmm_aux_2, r64_aux, 0x0);
        // vpxor(xmm_aux_2, xmm_aux_2, xmm_aux_2);
        vpclmulqdq(xmm_aux, xmm_dst, xmm_aux_2, 0b00000001);
        // mov(r64_aux, ptr[r64_params + GET_OFF(k_ptr)]);
        // vpclmulqdq(xmm_aux, xmm_dst, ptr[r64_aux + K_1_0_OFF], 0b00000001);
        vpslldq(xmm_dst, xmm_dst, 0x8);
        vpxor(xmm_dst, xmm_dst, xmm_aux);

        mov(r64_aux, P_1);
        vpinsrq(xmm_aux_2, xmm_aux_2, r64_aux, 0x0);
        // vmovdqu64(xmm_aux_2, ptr[r64_aux + K_P_P_OFF]);
        vpclmulqdq(xmm_aux, xmm_dst, xmm_aux_2, 0b00000001);

        mov(r64_aux, 0x0);
        vpinsrq(xmm_aux_1, xmm_dst, r64_aux, 0x0);
        vpxor(xmm_aux, xmm_aux, xmm_aux_1);
        vpinsrq(xmm_aux_1, xmm_aux, r64_aux, 0x0);

        mov(r64_aux, P_2); // P(x) - x^64
        vpinsrq(xmm_aux_2, xmm_aux_2, r64_aux, 0x1);
        vpclmulqdq(xmm_aux, xmm_aux, xmm_aux_2, 0b00010001);
        vpxor(xmm_aux, xmm_aux, xmm_aux_1);
        vpxor(xmm_dst, xmm_dst, xmm_aux);

        // mov(r64_aux, ptr[r64_params + GET_OFF(dst_ptr)]);
        vpextrq(ptr[r64_dst], xmm_dst, 0x0);

        this->postamble();
    }

    // static hash_kernel get() {
    //     static const ComputeHashCompileParams params;
    //     static ComputeHash4 kernel(params);

    //     return (hash_kernel)kernel.getCode();
    // }
};

template <>
void ComputeHash<avx512_core>::uni_vpxorq(const Xbyak::Xmm& v_dst, const Xbyak::Xmm& v_src_0, const Xbyak::Xmm& v_src_1) {
    vpxorq(v_dst, v_src_0, v_src_1);
}

template <cpu_isa_t isa>
void ComputeHash<isa>::uni_vpxorq(const Xbyak::Xmm& v_dst, const Xbyak::Xmm& v_src_0, const Xbyak::Xmm& v_src_1) {
    vpxor(v_dst, v_src_0, v_src_1);
}

template <>
void ComputeHash<avx512_core>::initialize() {
    v_dst = getVmm();
    auto r64_aux = getReg64();

    v_k_2_3 = getVmm();
    vbroadcasti64x2(v_k_2_3, ptr[r64_k_ptr + K_2_3_OFF]);

    v_shuf_mask = getVmm();
    mov(r64_aux, reinterpret_cast<uintptr_t>(SHUF_MASK));
    vbroadcasti64x2(v_shuf_mask, ptr[r64_aux]);

    if (m_jcp.type == SINGLE_THREAD || m_jcp.type == FIRST_THREAD) {
        auto xmm_dst = Xbyak::Xmm(v_dst.getIdx());
        auto xmm_aux = getXmm();
        auto k_rest_mask = RegistersPool::Reg<Xbyak::Opmask>(m_registers_pool);

        // Initial CRC
        mov(r64_aux, ptr[r64_params + GET_OFF(size)]);
        vpinsrq(xmm_aux, xmm_aux, r64_aux, 0x0);
        mov(r64_aux, CRC_VAL);
        vpinsrq(xmm_aux, xmm_aux, r64_aux, 0x1);
// vmovdqu64(ptr[r64_tmp], xmm_aux);
        // Initial xor with source
        fill_rest_work_mask(k_rest_mask, r64_work_amount);
        vmovdqu8(Vmm(v_dst.getIdx()) | k_rest_mask | T_z, ptr[r64_src]);
        vpshufb(v_dst, v_dst, v_shuf_mask);
        pxor(xmm_dst, xmm_aux); // The SSE version is used to avoid zeroing out the rest of the Vmm.
        if (m_jcp.type == SINGLE_THREAD) {
            add(r64_src, xmm_len);
        }
    } else if (m_jcp.type == N_THREAD) {
        vmovdqu64(v_dst, ptr[r64_src]);
        vpshufb(v_dst, v_dst, v_shuf_mask);
    }
    if (m_jcp.type == SINGLE_THREAD || m_jcp.type == FIRST_THREAD || m_jcp.type == N_THREAD) {
        sub(r64_work_amount, xmm_len);
    }
    // add(r64_src, vlen);
}

template <cpu_isa_t isa>
void ComputeHash<isa>::initialize() {
    auto r64_aux = getReg64();

    v_k_2_3 = getVmm();
    vbroadcasti128(v_k_2_3, ptr[r64_k_ptr + K_2_3_OFF]);

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
    partial_load(xmm_aux, ptr[r64_src], r64_work_amount);
    vpshufb(xmm_aux, xmm_aux, xmm_shuf_mask);
    uni_vpxorq(xmm_dst, xmm_dst, xmm_aux);
// vmovdqu(ptr[r64_tmp], xmm_dst);
    sub(r64_work_amount, xmm_len);
    add(r64_src, xmm_len);
}

// template <>
// void ComputeHash<avx512_core>::bulk_fold(const Vmm& v_dst) {
//     Xbyak::Label l_fold_loop, l_end;
//     cmp(r64_work_amount, 4 * vlen - xmm_len);
//     jl(l_end, T_NEAR);

//     auto r64_aux = getReg64();

//     auto v_src_0 = getVmm();
//     auto v_dst_0 = getVmm();
//     auto v_dst_1 = getVmm();
//     auto v_dst_2 = getVmm();
//     auto& v_dst_3 = v_dst;

//     auto v_dst_4 = getVmm();
//     auto v_dst_5 = getVmm();
//     auto v_dst_6 = getVmm();
//     auto v_dst_7 = getVmm();

//     auto v_aux_0 = getVmm();
//     auto v_k_loop = getVmm();

//     auto xmm_k_loop = Xbyak::Xmm(v_k_loop.getIdx());
//     auto xmm_k_2_3 = Xbyak::Xmm(v_k_2_3.getIdx());
//     auto xmm_src_0 = Xbyak::Xmm(v_src_0.getIdx());
//     auto xmm_src_1 = getXmm();
//     auto xmm_dst_0 = Xbyak::Xmm(v_dst_0.getIdx());
//     auto xmm_dst_1 = Xbyak::Xmm(v_dst_1.getIdx());
//     auto xmm_dst_2 = Xbyak::Xmm(v_dst_2.getIdx());
//     auto xmm_dst_3 = Xbyak::Xmm(v_dst_3.getIdx());

//     auto xmm_dst_4 = Xbyak::Xmm(v_dst_4.getIdx());
//     auto xmm_dst_5 = Xbyak::Xmm(v_dst_5.getIdx());
//     auto xmm_dst_6 = Xbyak::Xmm(v_dst_6.getIdx());
//     auto xmm_dst_7 = Xbyak::Xmm(v_dst_7.getIdx());

//     auto xmm_aux_0 = Xbyak::Xmm(v_aux_0.getIdx());
//     auto xmm_shuf_mask = Xbyak::Xmm(v_shuf_mask.getIdx());

//     RegistersPool::Reg<Xbyak::Reg64> r64_bulk_step;
//     if (m_jcp.type == FIRST_THREAD || m_jcp.type == N_THREAD) {
//         r64_bulk_step = getReg64();
//         mov(r64_bulk_step, ptr[r64_params + GET_OFF(threads_num)]);
//         // if (is_vpclmulqdq) {
//             // sal(r64_bulk_step, 6); // *vlen
//         // }
//     }

//     // if (m_jcp.type == SINGLE_THREAD) {
//     //     mov(r64_aux, reinterpret_cast<uintptr_t>(K_7_8));
//     //     vbroadcasti64x2(v_k_loop, ptr[r64_aux]);
//     // } else {
//         mov(r64_aux, reinterpret_cast<uintptr_t>(K_15_16));
//         vbroadcasti64x2(v_k_loop, ptr[r64_aux]);
//         // ...
//     // }

//     vmovdqu64(v_dst_0, v_dst_3);

//     if (!is_vpclmulqdq) {
//         vextracti64x2(xmm_dst_1, v_dst, 0x1);
//         vextracti64x2(xmm_dst_2, v_dst, 0x2);
//         vextracti64x2(xmm_dst_3, v_dst, 0x3);
//     }

// // auto r64_counter = getReg64();
// // mov(r64_counter, vlen);

//     if (m_jcp.type == FIRST_THREAD || m_jcp.type == N_THREAD) {
//         add(r64_src, r64_bulk_step);
//     } else {
//         add(r64_src, vlen - xmm_len);
//     }
//     prefetcht0(ptr[r64_src + 1024]);
//     sub(r64_work_amount, 4 * vlen - xmm_len);
//     // sub(r64_work_amount, 2 * vlen);
    
//     vmovdqu64(xmm_dst_4, ptr[r64_src]);
//     vpshufb(xmm_dst_4, xmm_dst_4, xmm_shuf_mask);
//     add(r64_src, xmm_len);
//     vmovdqu64(xmm_dst_5, ptr[r64_src]);
//     vpshufb(xmm_dst_5, xmm_dst_5, xmm_shuf_mask);
//     add(r64_src, xmm_len);
//     vmovdqu64(xmm_dst_6, ptr[r64_src]);
//     vpshufb(xmm_dst_6, xmm_dst_6, xmm_shuf_mask);
//     add(r64_src, xmm_len);
//     vmovdqu64(xmm_dst_7, ptr[r64_src]);
//     vpshufb(xmm_dst_7, xmm_dst_7, xmm_shuf_mask);
//     add(r64_src, xmm_len);

//     L(l_fold_loop); {
//         vmovdqu64(v_src_0, ptr[r64_src]);
//         vpshufb(v_src_0, v_src_0, v_shuf_mask);

//         if (m_jcp.type == FIRST_THREAD || m_jcp.type == N_THREAD) {
//             add(r64_src, r64_bulk_step);
//         } else {
//             add(r64_src, vlen);
//         }
//         prefetcht0(ptr[r64_src + 1024]);

//         if (is_vpclmulqdq) {
//             vpclmulqdq(v_aux_0, v_dst_0, v_k_loop, 0b00000000);
//             vpclmulqdq(v_dst_0, v_dst_0, v_k_loop, 0b00010001);
//             uni_vpxorq(v_aux_0, v_aux_0, v_src_0);
//             uni_vpxorq(v_dst_0, v_dst_0, v_aux_0);
//         } else {
//             // 0
//             vpclmulqdq(xmm_aux_0, xmm_dst_0, xmm_k_loop, 0b00000000);
//             vpclmulqdq(xmm_dst_0, xmm_dst_0, xmm_k_loop, 0b00010001);
//             uni_vpxorq(xmm_aux_0, xmm_aux_0, xmm_src_0);
//             uni_vpxorq(xmm_dst_0, xmm_dst_0, xmm_aux_0);

//             // 1
//             vextracti64x2(xmm_src_1, v_src_0, 0x1);

//             vpclmulqdq(xmm_aux_0, xmm_dst_1, xmm_k_loop, 0b00000000);
//             vpclmulqdq(xmm_dst_1, xmm_dst_1, xmm_k_loop, 0b00010001);
//             uni_vpxorq(xmm_aux_0, xmm_aux_0, xmm_src_1);
//             uni_vpxorq(xmm_dst_1, xmm_dst_1, xmm_aux_0);

//             // 2
//             vextracti64x2(xmm_src_1, v_src_0, 0x2);

//             vpclmulqdq(xmm_aux_0, xmm_dst_2, xmm_k_loop, 0b00000000);
//             vpclmulqdq(xmm_dst_2, xmm_dst_2, xmm_k_loop, 0b00010001);
//             uni_vpxorq(xmm_aux_0, xmm_aux_0, xmm_src_1);
//             uni_vpxorq(xmm_dst_2, xmm_dst_2, xmm_aux_0);

//             // 3
//             vextracti64x2(xmm_src_1, v_src_0, 0x3);

//             vpclmulqdq(xmm_aux_0, xmm_dst_3, xmm_k_loop, 0b00000000);
//             vpclmulqdq(xmm_dst_3, xmm_dst_3, xmm_k_loop, 0b00010001);
//             uni_vpxorq(xmm_aux_0, xmm_aux_0, xmm_src_1);
//             uni_vpxorq(xmm_dst_3, xmm_dst_3, xmm_aux_0);
            
//             // 4
//             vmovdqu64(v_src_0, ptr[r64_src]);
//             vpshufb(v_src_0, v_src_0, v_shuf_mask);
//             add(r64_src, vlen);

//             vpclmulqdq(xmm_aux_0, xmm_dst_4, xmm_k_loop, 0b00000000);
//             vpclmulqdq(xmm_dst_4, xmm_dst_4, xmm_k_loop, 0b00010001);
//             uni_vpxorq(xmm_aux_0, xmm_aux_0, xmm_src_0);
//             uni_vpxorq(xmm_dst_4, xmm_dst_4, xmm_aux_0);

//             // 5
//             vextracti64x2(xmm_src_1, v_src_0, 0x1);

//             vpclmulqdq(xmm_aux_0, xmm_dst_5, xmm_k_loop, 0b00000000);
//             vpclmulqdq(xmm_dst_5, xmm_dst_5, xmm_k_loop, 0b00010001);
//             uni_vpxorq(xmm_aux_0, xmm_aux_0, xmm_src_1);
//             uni_vpxorq(xmm_dst_5, xmm_dst_5, xmm_aux_0);

//             // 6
//             vextracti64x2(xmm_src_1, v_src_0, 0x2);

//             vpclmulqdq(xmm_aux_0, xmm_dst_6, xmm_k_loop, 0b00000000);
//             vpclmulqdq(xmm_dst_6, xmm_dst_6, xmm_k_loop, 0b00010001);
//             uni_vpxorq(xmm_aux_0, xmm_aux_0, xmm_src_1);
//             uni_vpxorq(xmm_dst_6, xmm_dst_6, xmm_aux_0);

//             // 7
//             vextracti64x2(xmm_src_1, v_src_0, 0x3);

//             vpclmulqdq(xmm_aux_0, xmm_dst_7, xmm_k_loop, 0b00000000);
//             vpclmulqdq(xmm_dst_7, xmm_dst_7, xmm_k_loop, 0b00010001);
//             uni_vpxorq(xmm_aux_0, xmm_aux_0, xmm_src_1);
//             uni_vpxorq(xmm_dst_7, xmm_dst_7, xmm_aux_0);
//         }

// // add(r64_counter, vlen);
//         sub(r64_work_amount, vlen * 2lu);
//         jge(l_fold_loop, T_NEAR);
//     }
//     add(r64_work_amount, vlen * 2lu);
// // mov(ptr[r64_tmp], r64_counter);
// // if (m_jcp.type == FIRST_THREAD || m_jcp.type == N_THREAD) {
// //     mov(ptr[r64_tmp], r64_bulk_step);
// // }

// // vmovdqu64(ptr[r64_tmp], xmm_dst_3);

//     if (m_jcp.type == SINGLE_THREAD) {
//         if (is_vpclmulqdq) {
//             auto ymm_dst_0 = Xbyak::Ymm(v_dst_0.getIdx());
//             auto ymm_dst_1 = Xbyak::Ymm(v_dst_1.getIdx());
//             auto ymm_aux_0 = Xbyak::Ymm(v_aux_0.getIdx());

//             vextracti64x4(ymm_dst_1, v_dst_0, 0x1);
//             mov(r64_aux, reinterpret_cast<uintptr_t>(K_3_4));
//             vpclmulqdq(ymm_aux_0, ymm_dst_0, ptr[r64_aux], 0b00000000);
//             vpclmulqdq(ymm_dst_0, ymm_dst_0, ptr[r64_aux], 0b00010001);
//             uni_vpxorq(ymm_dst_1, ymm_dst_1, ymm_aux_0);
//             uni_vpxorq(ymm_dst_0, ymm_dst_0, ymm_dst_1);

//             vextracti64x2(xmm_dst_3, ymm_dst_0, 0x1);
//             vpclmulqdq(xmm_aux_0, xmm_dst_0, xmm_k_2_3, 0b00000000);
//             vpclmulqdq(xmm_dst_0, xmm_dst_0, xmm_k_2_3, 0b00010001);
//             uni_vpxorq(xmm_dst_3, xmm_dst_3, xmm_aux_0);
//             uni_vpxorq(xmm_dst_3, xmm_dst_3, xmm_dst_0);
//         } else {
// auto r64_intm = getReg64();
// mov(r64_intm, ptr[r64_params + GET_OFF(intermediate_ptr)]);

// vmovdqu64(ptr[r64_intm], xmm_dst_0);
// add(r64_intm, xmm_len);
// vmovdqu64(ptr[r64_intm], xmm_dst_1);
// add(r64_intm, xmm_len);
// vmovdqu64(ptr[r64_intm], xmm_dst_2);
// add(r64_intm, xmm_len);
// vmovdqu64(ptr[r64_intm], xmm_dst_3);
// add(r64_intm, xmm_len);
// vmovdqu64(ptr[r64_intm], xmm_dst_4);
// add(r64_intm, xmm_len);
// vmovdqu64(ptr[r64_intm], xmm_dst_5);
// add(r64_intm, xmm_len);
// vmovdqu64(ptr[r64_intm], xmm_dst_6);
// add(r64_intm, xmm_len);
// vmovdqu64(ptr[r64_intm], xmm_dst_7);

//             mov(r64_aux, reinterpret_cast<uintptr_t>(K_13_14));
//             vpclmulqdq(xmm_aux_0, xmm_dst_0, ptr[r64_aux], 0b00000000);
//             vpclmulqdq(xmm_dst_0, xmm_dst_0, ptr[r64_aux], 0b00010001);
//             uni_vpxorq(xmm_dst_7, xmm_dst_7, xmm_aux_0);
//             uni_vpxorq(xmm_dst_7, xmm_dst_7, xmm_dst_0);

//             mov(r64_aux, reinterpret_cast<uintptr_t>(K_11_12));
//             vpclmulqdq(xmm_aux_0, xmm_dst_1, ptr[r64_aux], 0b00000000);
//             vpclmulqdq(xmm_dst_1, xmm_dst_1, ptr[r64_aux], 0b00010001);
//             uni_vpxorq(xmm_dst_7, xmm_dst_7, xmm_aux_0);
//             uni_vpxorq(xmm_dst_7, xmm_dst_7, xmm_dst_1);

//             mov(r64_aux, reinterpret_cast<uintptr_t>(K_9_10));
//             vpclmulqdq(xmm_aux_0, xmm_dst_2, ptr[r64_aux], 0b00000000);
//             vpclmulqdq(xmm_dst_2, xmm_dst_2, ptr[r64_aux], 0b00010001);
//             uni_vpxorq(xmm_dst_7, xmm_dst_7, xmm_aux_0);
//             uni_vpxorq(xmm_dst_7, xmm_dst_7, xmm_dst_2);

//             mov(r64_aux, reinterpret_cast<uintptr_t>(K_7_8));
//             vpclmulqdq(xmm_aux_0, xmm_dst_3, ptr[r64_aux], 0b00000000);
//             vpclmulqdq(xmm_dst_3, xmm_dst_3, ptr[r64_aux], 0b00010001);
//             uni_vpxorq(xmm_dst_7, xmm_dst_7, xmm_aux_0);
//             uni_vpxorq(xmm_dst_7, xmm_dst_7, xmm_dst_3);

//             mov(r64_aux, reinterpret_cast<uintptr_t>(K_5_6));
//             vpclmulqdq(xmm_aux_0, xmm_dst_4, ptr[r64_aux], 0b00000000);
//             vpclmulqdq(xmm_dst_4, xmm_dst_4, ptr[r64_aux], 0b00010001);
//             uni_vpxorq(xmm_dst_7, xmm_dst_7, xmm_aux_0);
//             uni_vpxorq(xmm_dst_7, xmm_dst_7, xmm_dst_4);

//             mov(r64_aux, reinterpret_cast<uintptr_t>(K_3_4));
//             vpclmulqdq(xmm_aux_0, xmm_dst_5, ptr[r64_aux], 0b00000000);
//             vpclmulqdq(xmm_dst_5, xmm_dst_5, ptr[r64_aux], 0b00010001);
//             uni_vpxorq(xmm_dst_7, xmm_dst_7, xmm_aux_0);
//             uni_vpxorq(xmm_dst_7, xmm_dst_7, xmm_dst_5);

//             vpclmulqdq(xmm_aux_0, xmm_dst_6, xmm_k_2_3, 0b00000000);
//             vpclmulqdq(xmm_dst_6, xmm_dst_6, xmm_k_2_3, 0b00010001);
//             uni_vpxorq(xmm_dst_7, xmm_dst_7, xmm_aux_0);
//             uni_vpxorq(xmm_dst_7, xmm_dst_7, xmm_dst_6);
            
//             vmovdqu64(xmm_dst_3, xmm_dst_7);

//             // mov(r64_aux, reinterpret_cast<uintptr_t>(K_5_6));
//             // vpclmulqdq(xmm_aux_0, xmm_dst_0, ptr[r64_aux], 0b00000000);
//             // vpclmulqdq(xmm_dst_0, xmm_dst_0, ptr[r64_aux], 0b00010001);
//             // uni_vpxorq(xmm_dst_3, xmm_dst_3, xmm_aux_0);
//             // uni_vpxorq(xmm_dst_3, xmm_dst_3, xmm_dst_0);

//             // mov(r64_aux, reinterpret_cast<uintptr_t>(K_3_4));
//             // vpclmulqdq(xmm_aux_0, xmm_dst_1, ptr[r64_aux], 0b00000000);
//             // vpclmulqdq(xmm_dst_1, xmm_dst_1, ptr[r64_aux], 0b00010001);
//             // uni_vpxorq(xmm_dst_3, xmm_dst_3, xmm_aux_0);
//             // uni_vpxorq(xmm_dst_3, xmm_dst_3, xmm_dst_1);

//             // vpclmulqdq(xmm_aux_0, xmm_dst_2, xmm_k_2_3, 0b00000000);
//             // vpclmulqdq(xmm_dst_2, xmm_dst_2, xmm_k_2_3, 0b00010001);
//             // uni_vpxorq(xmm_dst_3, xmm_dst_3, xmm_aux_0);
//             // uni_vpxorq(xmm_dst_3, xmm_dst_3, xmm_dst_2);
//         }
//     } else {
//         if (is_vpclmulqdq) {
//             vmovdqu64(ptr[r64_dst], v_dst_0);
//         } else {
//             vmovdqu64(ptr[r64_dst + xmm_len * 0lu], xmm_dst_0);
//             vmovdqu64(ptr[r64_dst + xmm_len * 1lu], xmm_dst_1);
//             vmovdqu64(ptr[r64_dst + xmm_len * 2lu], xmm_dst_2);
//             vmovdqu64(ptr[r64_dst + xmm_len * 3lu], xmm_dst_3);
//         }
//     }

//     L(l_end);
// }

template <>
void ComputeHash<avx512_core>::bulk_fold(const Vmm& v_dst) {
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
    auto xmm_k_2_3 = Xbyak::Xmm(v_k_2_3.getIdx());
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
            sal(r64_bulk_step, 6); // *vlen
        // }
    }

    if (m_jcp.type == SINGLE_THREAD) {
        vbroadcasti64x2(v_k_loop, ptr[r64_k_ptr + K_8_9_OFF]);
    } else {
        vbroadcasti64x2(v_k_loop, ptr[r64_k_ptr + K_16_17_OFF]);
    }

    vmovdqu64(v_dst_0, v_dst);
// vmovdqu64(ptr[r64_tmp], xmm_dst_0);

    if (!is_vpclmulqdq) {
        vextracti64x2(xmm_dst_1, v_dst_0, 0x1);
        vextracti64x2(xmm_dst_2, v_dst_0, 0x2);
        vextracti64x2(xmm_dst_3, v_dst_0, 0x3);
    }

// auto r64_counter = getReg64();
// mov(r64_counter, vlen);

    if (m_jcp.type == FIRST_THREAD || m_jcp.type == N_THREAD) {
        add(r64_src, r64_bulk_step);
        // prefetcht2(ptr[r64_src + 8192]);
        prefetcht2(ptr[r64_src + 16384]);
    } else {
        add(r64_src, vlen - xmm_len);
        prefetcht2(ptr[r64_src + 4096]);
        // prefetcht2(ptr[r64_src + 16384]);
    }
    prefetcht1(ptr[r64_src + 1024]);
    // prefetcht0(ptr[r64_src + 1024]);
    prefetcht0(ptr[r64_src + 64]);
    // prefetcht0(ptr[r64_src]);

    sub(r64_work_amount, 2 * vlen - xmm_len);
    // sub(r64_work_amount, 2 * vlen);

    L(l_fold_loop); {
        vmovdqu64(v_src_0, ptr[r64_src]);
        vpshufb(v_src_0, v_src_0, v_shuf_mask);

        if (m_jcp.type == FIRST_THREAD || m_jcp.type == N_THREAD) {
            add(r64_src, r64_bulk_step);
            // prefetcht2(ptr[r64_src + 8192]);
            prefetcht2(ptr[r64_src + 16384]);
        } else {
            add(r64_src, vlen);
            prefetcht2(ptr[r64_src + 4096]);
            // prefetcht2(ptr[r64_src + 8192]);
            // prefetcht2(ptr[r64_src + 16384]);
        }
        prefetcht1(ptr[r64_src + 1024]);
        // prefetcht0(ptr[r64_src + 1024]);
        prefetcht0(ptr[r64_src + 64]);

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
            vpclmulqdq(ymm_aux_0, ymm_dst_0, ptr[r64_k_ptr + K_4_5_OFF], 0b00000000);
            vpclmulqdq(ymm_dst_0, ymm_dst_0, ptr[r64_k_ptr + K_4_5_OFF], 0b00010001);
            uni_vpxorq(ymm_dst_1, ymm_dst_1, ymm_aux_0);
            uni_vpxorq(ymm_dst_0, ymm_dst_0, ymm_dst_1);

            vextracti64x2(xmm_dst_3, ymm_dst_0, 0x1);
            vpclmulqdq(xmm_aux_0, xmm_dst_0, xmm_k_2_3, 0b00000000);
            vpclmulqdq(xmm_dst_0, xmm_dst_0, xmm_k_2_3, 0b00010001);
            uni_vpxorq(xmm_dst_3, xmm_dst_3, xmm_aux_0);
            uni_vpxorq(xmm_dst_3, xmm_dst_3, xmm_dst_0);
        } else {
            vpclmulqdq(xmm_aux_0, xmm_dst_0, ptr[r64_k_ptr + K_6_7_OFF], 0b00000000);
            vpclmulqdq(xmm_dst_0, xmm_dst_0, ptr[r64_k_ptr + K_6_7_OFF], 0b00010001);
            uni_vpxorq(xmm_dst_3, xmm_dst_3, xmm_aux_0);
            uni_vpxorq(xmm_dst_3, xmm_dst_3, xmm_dst_0);

            vpclmulqdq(xmm_aux_0, xmm_dst_1, ptr[r64_k_ptr + K_4_5_OFF], 0b00000000);
            vpclmulqdq(xmm_dst_1, xmm_dst_1, ptr[r64_k_ptr + K_4_5_OFF], 0b00010001);
            uni_vpxorq(xmm_dst_3, xmm_dst_3, xmm_aux_0);
            uni_vpxorq(xmm_dst_3, xmm_dst_3, xmm_dst_1);

            vpclmulqdq(xmm_aux_0, xmm_dst_2, xmm_k_2_3, 0b00000000);
            vpclmulqdq(xmm_dst_2, xmm_dst_2, xmm_k_2_3, 0b00010001);
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
void ComputeHash<isa>::bulk_fold(const Vmm& v_dst) {
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
    auto xmm_k_2_3 = Xbyak::Xmm(v_k_2_3.getIdx());
    auto xmm_src_0 = Xbyak::Xmm(v_src_0.getIdx());
    auto xmm_src_1 = getXmm();
    auto xmm_dst_0 = Xbyak::Xmm(v_dst_0.getIdx());
    auto xmm_dst_1 = Xbyak::Xmm(v_dst_1.getIdx());
    // auto xmm_dst_2 = Xbyak::Xmm(v_dst_2.getIdx());
    // auto xmm_dst_3 = Xbyak::Xmm(v_dst_3.getIdx());
    auto xmm_aux_0 = Xbyak::Xmm(v_aux_0.getIdx());

    vbroadcasti128(v_k_loop, ptr[r64_k_ptr + K_4_5_OFF]);

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
        vpclmulqdq(xmm_aux_0, xmm_dst_0, xmm_k_2_3, 0b00000000);
        vpclmulqdq(xmm_dst_0, xmm_dst_0, xmm_k_2_3, 0b00010001);
        uni_vpxorq(xmm_dst_1, xmm_dst_1, xmm_aux_0);
        uni_vpxorq(xmm_dst_1, xmm_dst_1, xmm_dst_0);
    } else {
        // vpclmulqdq(xmm_aux_0, xmm_dst_2, xmm_k_2_3, 0b00000000);
        // vpclmulqdq(xmm_dst_2, xmm_dst_2, xmm_k_2_3, 0b00010001);
        // uni_vpxorq(xmm_dst_3, xmm_dst_3, xmm_aux_0);
        // uni_vpxorq(xmm_dst_3, xmm_dst_3, xmm_dst_2);
    }

    L(l_end);
}

template <>
void ComputeHash<avx512_core>::after_bulk_fold(const Vmm& v_dst) {
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
        // vpclmulqdq(xmm_aux_0, xmm_dst_0, xmm_k_2_3, 0b00000000);
        // vpclmulqdq(xmm_dst_0, xmm_dst_0, xmm_k_2_3, 0b00010001);
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
        auto xmm_k_2_3 = Xbyak::Xmm(v_k_2_3.getIdx());

        vmovdqu64(xmm_src_last, ptr[r64_intm + xmm_len * 7]);

        vmovdqu64(xmm_src_0, ptr[r64_intm]);
        vpclmulqdq(xmm_aux_0, xmm_src_0, ptr[r64_k_ptr + K_14_15_OFF], 0b00000000);
        vpclmulqdq(xmm_src_0, xmm_src_0, ptr[r64_k_ptr + K_14_15_OFF], 0b00010001);
        uni_vpxorq(xmm_src_last, xmm_src_last, xmm_aux_0);
        uni_vpxorq(xmm_src_last, xmm_src_last, xmm_src_0);

        vmovdqu64(xmm_src_0, ptr[r64_intm + xmm_len]);
        vpclmulqdq(xmm_aux_0, xmm_src_0, ptr[r64_k_ptr + K_12_13_OFF], 0b00000000);
        vpclmulqdq(xmm_src_0, xmm_src_0, ptr[r64_k_ptr + K_12_13_OFF], 0b00010001);
        uni_vpxorq(xmm_src_last, xmm_src_last, xmm_aux_0);
        uni_vpxorq(xmm_src_last, xmm_src_last, xmm_src_0);

        vmovdqu64(xmm_src_0, ptr[r64_intm + xmm_len * 2lu]);
        vpclmulqdq(xmm_aux_0, xmm_src_0, ptr[r64_k_ptr + K_10_11_OFF], 0b00000000);
        vpclmulqdq(xmm_src_0, xmm_src_0, ptr[r64_k_ptr + K_10_11_OFF], 0b00010001);
        uni_vpxorq(xmm_src_last, xmm_src_last, xmm_aux_0);
        uni_vpxorq(xmm_src_last, xmm_src_last, xmm_src_0);

        vmovdqu64(xmm_src_0, ptr[r64_intm + xmm_len * 3lu]);
        vpclmulqdq(xmm_aux_0, xmm_src_0, ptr[r64_k_ptr + K_8_9_OFF], 0b00000000);
        vpclmulqdq(xmm_src_0, xmm_src_0, ptr[r64_k_ptr + K_8_9_OFF], 0b00010001);
        uni_vpxorq(xmm_src_last, xmm_src_last, xmm_aux_0);
        uni_vpxorq(xmm_src_last, xmm_src_last, xmm_src_0);

        vmovdqu64(xmm_src_0, ptr[r64_intm + xmm_len * 4lu]);
        vpclmulqdq(xmm_aux_0, xmm_src_0, ptr[r64_k_ptr + K_6_7_OFF], 0b00000000);
        vpclmulqdq(xmm_src_0, xmm_src_0, ptr[r64_k_ptr + K_6_7_OFF], 0b00010001);
        uni_vpxorq(xmm_src_last, xmm_src_last, xmm_aux_0);
        uni_vpxorq(xmm_src_last, xmm_src_last, xmm_src_0);

        vmovdqu64(xmm_src_0, ptr[r64_intm + xmm_len * 5lu]);
        vpclmulqdq(xmm_aux_0, xmm_src_0, ptr[r64_k_ptr + K_4_5_OFF], 0b00000000);
        vpclmulqdq(xmm_src_0, xmm_src_0, ptr[r64_k_ptr + K_4_5_OFF], 0b00010001);
        uni_vpxorq(xmm_src_last, xmm_src_last, xmm_aux_0);
        uni_vpxorq(xmm_src_last, xmm_src_last, xmm_src_0);

        vmovdqu64(xmm_src_0, ptr[r64_intm + xmm_len * 6lu]);
        vpclmulqdq(xmm_aux_0, xmm_src_0, xmm_k_2_3, 0b00000000);
        vpclmulqdq(xmm_src_0, xmm_src_0, xmm_k_2_3, 0b00010001);
        uni_vpxorq(xmm_src_last, xmm_src_last, xmm_aux_0);
        uni_vpxorq(xmm_src_last, xmm_src_last, xmm_src_0);
    }
}

template <cpu_isa_t isa>
void ComputeHash<isa>::after_bulk_fold(const Vmm& v_dst) {
    if (m_jcp.type != FINAL_FOLD) {
        return;
    }
}

template <>
void ComputeHash<avx512_core>::rest_fold(const Vmm& v_dst) {
    Xbyak::Label l_fold_loop, l_end;
    cmp(r64_work_amount, xmm_len);
    jl(l_end, T_NEAR);

    auto xmm_shuf_mask = Xbyak::Xmm(v_shuf_mask.getIdx());
    auto xmm_k_2_3 = Xbyak::Xmm(v_k_2_3.getIdx());
    auto xmm_src = getXmm();
    auto xmm_dst = Xbyak::Xmm(v_dst.getIdx());
    auto xmm_aux = getXmm();

    L(l_fold_loop); {
        vmovdqu64(xmm_src, ptr[r64_src]); // TODO: compare assembled code with vmovdqu
        vpshufb(xmm_src, xmm_src, xmm_shuf_mask);

        vpclmulqdq(xmm_aux, xmm_dst, xmm_k_2_3, 0b00000000);
        vpclmulqdq(xmm_dst, xmm_dst, xmm_k_2_3, 0b00010001);
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
void ComputeHash<isa>::rest_fold(const Vmm& v_dst) {
    Xbyak::Label l_fold_loop, l_end;
    cmp(r64_work_amount, xmm_len);
    jl(l_end, T_NEAR);

    auto xmm_shuf_mask = Xbyak::Xmm(v_shuf_mask.getIdx());
    auto xmm_k_2_3 = Xbyak::Xmm(v_k_2_3.getIdx());
    auto xmm_src = getXmm();
    auto xmm_dst = Xbyak::Xmm(v_dst.getIdx());
    auto xmm_aux = getXmm();

    L(l_fold_loop); {
        vmovdqu(xmm_src, ptr[r64_src]);
        vpshufb(xmm_src, xmm_src, xmm_shuf_mask);

        vpclmulqdq(xmm_aux, xmm_dst, xmm_k_2_3, 0b00000000);
        vpclmulqdq(xmm_dst, xmm_dst, xmm_k_2_3, 0b00010001);
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
void ComputeHash<avx512_core>::tail_fold(const Vmm& v_dst) {
Xbyak::Label l_fold_to_64;
    cmp(r64_work_amount, 0);
    jle(l_fold_to_64, T_NEAR);

    auto r64_aux = getReg64();
    auto xmm_shuf_mask = Xbyak::Xmm(v_shuf_mask.getIdx());
    auto xmm_k_2_3 = Xbyak::Xmm(v_k_2_3.getIdx());
    auto xmm_src = getXmm();
    auto xmm_dst = Xbyak::Xmm(v_dst.getIdx());
    auto xmm_aux = getXmm();
    auto xmm_aux_1 = getXmm();
    auto xmm_aux_2 = getXmm();
    auto k_rest_mask = RegistersPool::Reg<Xbyak::Opmask>(m_registers_pool);

    fill_rest_work_mask(k_rest_mask, r64_work_amount);
    vmovdqu8(Xbyak::Xmm(xmm_src.getIdx()) | k_rest_mask | T_z, ptr[r64_src]);
    vpshufb(xmm_src, xmm_src, xmm_shuf_mask);

    vpclmulqdq(xmm_aux, xmm_dst, xmm_k_2_3, 0b00000000);
    vpclmulqdq(xmm_dst, xmm_dst, xmm_k_2_3, 0b00010001);
    uni_vpxorq(xmm_aux, xmm_aux, xmm_src);
    uni_vpxorq(xmm_dst, xmm_dst, xmm_aux);

    L(l_fold_to_64);

    mov(r64_aux, K_2);
    vpinsrq(xmm_aux, xmm_aux, r64_aux, 0x0);
    vpclmulqdq(xmm_aux, xmm_dst, xmm_aux, 0b00000001);
    vpslldq(xmm_dst, xmm_dst, 0x8);
    uni_vpxorq(xmm_dst, xmm_dst, xmm_aux);

    mov(r64_aux, P_1);
    vpinsrq(xmm_aux_2, xmm_aux_2, r64_aux, 0x0);
    vpclmulqdq(xmm_aux, xmm_dst, xmm_aux_2, 0b00000001);
    mov(r64_aux, 0x0);
    vpinsrq(xmm_aux_1, xmm_dst, r64_aux, 0x0);
    uni_vpxorq(xmm_aux, xmm_aux, xmm_aux_1);
    vpinsrq(xmm_aux_1, xmm_aux, r64_aux, 0x0);

    mov(r64_aux, P_2);
    vpinsrq(xmm_aux_2, xmm_aux_2, r64_aux, 0x1);
    vpclmulqdq(xmm_aux, xmm_aux, xmm_aux_2, 0b00010001);
    uni_vpxorq(xmm_aux, xmm_aux, xmm_aux_1);
    uni_vpxorq(xmm_dst, xmm_dst, xmm_aux);

    vpextrq(ptr[r64_dst], xmm_dst, 0x0);
}

template <cpu_isa_t isa>
void ComputeHash<isa>::tail_fold(const Vmm& v_dst) {
// vmovdqu(ptr[r64_tmp], Xbyak::Xmm(v_dst.getIdx()));
    Xbyak::Label l_fold_to_64, l_save_128, l_end;
    cmp(r64_work_amount, 0);
    jle(l_fold_to_64, T_NEAR);

    auto r64_aux = getReg64();
    auto xmm_shuf_mask = Xbyak::Xmm(v_shuf_mask.getIdx());
    auto xmm_k_2_3 = Xbyak::Xmm(v_k_2_3.getIdx());
    auto xmm_src = getXmm();
    auto xmm_dst = Xbyak::Xmm(v_dst.getIdx());
    auto xmm_aux = getXmm();
    auto xmm_aux_1 = getXmm();
    auto xmm_aux_2 = getXmm();
// vmovdqu(ptr[r64_tmp], xmm_dst);

    uni_vpxorq(xmm_src, xmm_src, xmm_src);
    partial_load(xmm_src, ptr[r64_src], r64_work_amount);
    vpshufb(xmm_src, xmm_src, xmm_shuf_mask);

    vpclmulqdq(xmm_aux, xmm_dst, xmm_k_2_3, 0b00000000);
    vpclmulqdq(xmm_dst, xmm_dst, xmm_k_2_3, 0b00010001);
    uni_vpxorq(xmm_aux, xmm_aux, xmm_src);
    uni_vpxorq(xmm_dst, xmm_dst, xmm_aux);

    L(l_fold_to_64);
    cmp(r64_make_64_fold, 0);
    je(l_save_128, T_NEAR);

    vpclmulqdq(xmm_aux, xmm_dst, ptr[r64_k_ptr + K_1_0_OFF], 0b00000001);
    vpslldq(xmm_dst, xmm_dst, 0x8);
    uni_vpxorq(xmm_dst, xmm_dst, xmm_aux);

    vmovdqu(xmm_aux_2, ptr[r64_k_ptr + K_P_P_OFF]);
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
// const uint64_t ComputeHash<isa>::K12 = 0x7B4BC8789D65B2A5;

// template <cpu_isa_t isa>
// const uint64_t ComputeHash<isa>::CRC_VAL = 0xffffffffffffffff;

// Auxiliary fn to obtain K constant multipliers.
// uint32_t gen_k_value(int t, uint32_t poly = 0x04C11DB7) {
// uint32_t gen_k_value(int t, uint32_t poly = 0xD663B05D) {
// uint32_t gen_k_value(int t, uint32_t poly = 0x741B8CD7) {
    // uint32_t res = poly, mask = 0x80000000;
// uint64_t gen_k_value(int t, uint64_t poly = 0xD663B05D) {

uint64_t gen_k_value(int64_t degree) {
    constexpr uint64_t POLYNOM = 0x42F0E1EBA9EA3693;
    constexpr uint64_t MASK    = 0x8000000000000000;
    uint64_t result = POLYNOM;
    do {
        if (result & MASK) {
            result = (result << 1) ^ POLYNOM;
        } else {
            result = (result << 1);
        }
    } while (degree--);

    return result;
}

uint32_t gen_k_value_32(int64_t degree) {
    // constexpr uint64_t POLYNOM = 0x0000000004c11db7;
    // constexpr uint64_t MASK    = 0x0000000080000000;
    constexpr uint32_t POLYNOM = 0x04c11db7;
    constexpr uint32_t MASK    = 0x80000000;

    // uint32_t result = POLYNOM;
    // do {
    //     if (result & MASK) {
    //         result = (result << 1) ^ POLYNOM;
    //     } else {
    //         result = (result << 1);
    //     }
    // } while (degree--);

    // return result;
    
    uint32_t N = MASK;
    if (degree <= 32)
        return 0ull;
    degree -= 31;
    if (degree == 33) {
        std::cout << std::endl;
    }
    for (size_t i = 0lu; i < degree; i++) {
        if (degree == 33) {
            std::cout << "    N: " << N << "; N >> 31: " << (N >> 31) << "; 0x00ul - (N >> 31): " << (0x00ul - (N >> 31))
                << "; (0x00ul - (N >> 31)) & POLYNOM: " << ((0x00ul - (N >> 31)) & POLYNOM) << "; N << 1: " << (N << 1)
                << "; RES: " << ((N << 1) ^ ((0x00ul - (N >> 31)) & POLYNOM)) << std::endl;
        }
        N = (N << 1) ^ ((0x00ul - (N >> 31)) & POLYNOM); // 2^(E)%poly
    }

    return N;
}

void crc_gen_table_32() {
    constexpr uint32_t poly = 0x04c11db7;
    const int bits = 64;
    const int size = 256;
    // constexpr uint32_t MASK = 0x80000000;

	uint32_t table[size]; // = {0};
	uint32_t p = poly;
	int i,j;
	table[0] = 0;
	table[1] = p;
	for (i = 1; (1 << i) < size; i++)
	{
		if (p&(1<<(bits-1))) {
			p &= ~((~0)<<(bits-1));
			p = (p<<1) ^ poly;
		} else {
			p = (p << 1);
        }
		table[(1<<i)] = p;
		for(j=1; j<(1<<i); j++) {
			table[(1<<i)+j] = p ^ table[j];
		}
	}
	printf("POLY=0x%0*X\n", bits/4, poly);
	for(i=0;i<size;i++){
		printf("0x%0*X, ", bits/4, table[i]);
		if ((i&0x7)==0x7) printf("\n");
	}
	printf("\n");
}

void crc_gen_inv_table() {
    constexpr uint32_t poly = 0x04c11db7;
    const int bits = 64;
    const int size = 256;

	uint32_t table[size] = {0};
	uint32_t p = poly;
	int i,j;
	table[0] = 0;
	table[1] = p;
	for (i = 1; (1 << i) < size; i++) {
		if (p&1) {
			p = (p>>1) ^ poly;
        } else {
			p = (p>>1);
        }
		table[(1<<i)] = p;
		for (j=1; j<(1<<i); j++) {
			table[(1<<i)+j] = p ^ table[j];
		}
	}
	printf("POLY=0x%0*X\n", bits/4, poly);
	for(i=0;i<size;i++){
		int ri;
		ri = ( i&0x3)<<2 | ( i&0xC)>>2;
		ri = (ri&0x5)<<1 | (ri&0xA)>>1;
		printf("0x%0*X, ", bits/4, table[ri]);
		if ((i&0x7)==0x7) printf("\n");
	}
	printf("\n");
}

void crc_gen_table_64() {
    // constexpr uint64_t poly = 0x0000000004c11db7;
    const uint64_t poly = 0x42F0E1EBA9EA3693;
    const int bits = 64;
    const int size = 16;
    uint64_t table[size];
    const uint64_t MASK_0 = ~((~0lu) << (bits - 1));
    const uint64_t MASK_1 = 1lu << (bits - 1);
std::cout << "MASK_0: " << MASK_0 << "; MASK_1: " << MASK_1 << std::endl;

    uint64_t p = poly;
    int i, j;
    table[0] = 0;
    table[1] = p;
    for (i = 1; (1 << i) < size; i++) {
        if (p & MASK_1) {
            p &= MASK_0;
            p = (p << 1) ^ poly;
        } else {
            p = (p << 1);
        }
        table[1 << i] = p;
        for (j = 1; j < (1 << i); j++) {
            table[(1 << i) + j] = p ^ table[j];
        }
    }

    printf("POLY=0x%0*"PRIX64"\n", bits/4, poly);
    for (i = 0; i < size; i++) {
        printf("0x%0*"PRIX64", ", bits/4, table[i]);
        // printf("0x%0*"PRIX64", ", 16, table[i]);
        if ((i & 0x3) == 0x3) printf("\n");
    }
}

uint64_t grk(uint64_t E) {
    constexpr uint64_t POLYNOM = 0x42F0E1EBA9EA3693;
    uint64_t N = 0x8000000000000000;
    if (E <= 64)
        return 0ull;
    E -= 63;
    for (size_t i = 0lu; i < E; i++)
        N = (N << 1) ^ ((0x00ul - (N >> 63)) & POLYNOM);
    return N;                       // 2^(E)%poly
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

} // namespace jit
#endif // OPENVINO_ARCH_X86 || OPENVINO_ARCH_X86_64

size_t compute_hash(const void* src, size_t size) {
if (size > 131072 * 2)
    printf("combine_hash size: %lu\n", size);
static uint64_t counter = 0lu;
static uint64_t sum = 0lu;
// counter++;
std::chrono::high_resolution_clock::time_point t1, t2;
t1 = std::chrono::high_resolution_clock::now();
// std::cout << "barrett_calc 64: " << std::hex << jit::barrett_calc()
//           << "; barrett_calc 32: " << std::hex << jit::barrett_calc(0xD663B05D, 32)
//           << "; barrett_calc 32: " << std::hex << jit::barrett_calc(0x04C11DB7, 32)
//           << "; barrett_calc 16: " << std::hex << jit::barrett_calc(0x8408, 16)
//           << "; barrett_calc 16: " << std::hex << jit::barrett_calc(0xA001, 16)
//           << "; barrett_calc 16m: " << std::hex << jit::barrett_calc(0x4003, 16)
//           << "; barrett_calc 16b: " << std::hex << jit::barrett_calc(0x0811, 16) << std::endl;
#if defined(OPENVINO_ARCH_X86) || defined(OPENVINO_ARCH_X86_64)
    // static std::vector<uint64_t> k_array = {
    //     0x05f5c3c7eb52fab6, 0x0000000000000000,  // x^(64*1),  x^(64*1) mod P(x)
    //     0x578d29d06cc4f872, 0x42f0e1eba9ea3693,  // floor(x^128/P(x)) - x^64, P(x) - x^64
    //     0x05f5c3c7eb52fab6, 0x4eb938a7d257740e,  // x^(64*1),  x^(64*2)
    //     0x571bee0a227ef92b, 0x44bef2a201b5200c,  // x^(64*3),  x^(64*4)
    //     0x54819d8713758b2c, 0x4a6b90073eb0af5a,  // x^(64*5),  x^(64*6)
    //     0x5f6843ca540df020, 0xddf4b6981205b83f   // x^(64*7),  x^(64*8)
    // };

    // static bool dump_k = true;
    // if (dump_k) {
    //     std::cout << std::hex;
    //     dump_k = false;
    //     // jit::crc_gen_inv_table();
    //     jit::crc_gen_table_32();
    //     // for (int i = 0; i < 2048; i += 64) {
    //     // for (int i = 0; i < 1024; i += 32) {
    //     // // for (int i = 0; i < 256; i += 8) {
    //     //     // std::cout << std::dec << "\nx^" << i << ": " << std::hex << jit::grk(i);
    //     //     std::cout << std::dec << "\nx^" << i << ": " << std::hex << jit::gen_k_value_32(i);
    //     //     // std::cout << std::dec << "\nx^" << i << ": " << std::hex << jit::gen_k_value(i);
    //     // //     std::cout << std::dec << "\nx^" << i << ": " << std::hex << jit::xt_mod_P_neg(i);
    //     // }
    //     std::cout << std::endl;
    // }
    // std::cout << std::dec << "\nx^" << 128 << ": " << std::hex << jit::xt_mod_P_neg(128);
    
    if (Generator::mayiuse(avx2)) {
        uint64_t result = 0lu;

        constexpr size_t min_wa_per_thread = 131072lu; // 2^17
//         if (size == 4lu) {
            // params.type = jit::CUSTOM_4B;
            // static jit::ComputeHash4 kernel_5(params);
            // kernels[jit::CUSTOM_4B] = (jit::hash_kernel)kernel_5.getCode();

//             // const uint64_t crc_xmm[2] = { 4lu, jit::CRC_VAL };
//             jit::ComputeHashCallArgs args;

//             args.src_ptr = src;
//             args.dst_ptr = &result;
//             args.k_ptr   = k_array.data();

// // t1 = std::chrono::high_resolution_clock::now();
//             kernels[jit::CUSTOM_4B](&args);
// // t2 = std::chrono::high_resolution_clock::now();
        // } else if (size >= 200000000lu) {
        // Parallel section
        if (size >= min_wa_per_thread * 2lu) {
        // if (size >= 2000000000lu) {
            static auto first_thr_kernel = Generator::mayiuse(avx512_core) ?
                jit::ComputeHash<avx512_core>::create({jit::FIRST_THREAD}) : jit::ComputeHash<avx2>::create({jit::FIRST_THREAD});
            static auto n_thr_kernel = Generator::mayiuse(avx512_core) ?
                jit::ComputeHash<avx512_core>::create({jit::N_THREAD}) : jit::ComputeHash<avx2>::create({jit::N_THREAD});
            static auto final_fold_kernel = Generator::mayiuse(avx512_core) ?
                jit::ComputeHash<avx512_core>::create({jit::FINAL_FOLD}) : jit::ComputeHash<avx2>::create({jit::FINAL_FOLD});
                
            
            // parallel_nt_static(3, [&](const int ithr, const int nthr) {
            //     if 
            //     static auto first_thr_kernel = Generator::mayiuse(avx512_core) ?
            //         jit::ComputeHash<avx512_core>::create({jit::FIRST_THREAD}) : jit::ComputeHash<avx2>::create({jit::FIRST_THREAD});
            //     static auto n_thr_kernel = Generator::mayiuse(avx512_core) ?
            //         jit::ComputeHash<avx512_core>::create({jit::N_THREAD}) : jit::ComputeHash<avx2>::create({jit::N_THREAD});
            //     static auto final_fold_kernel = Generator::mayiuse(avx512_core) ?
            //         jit::ComputeHash<avx512_core>::create({jit::FINAL_FOLD}) : jit::ComputeHash<avx2>::create({jit::FINAL_FOLD});
            // });

// printf("    parallel_get_max_threads : %d\n", parallel_get_max_threads());
            // static const size_t max_thr_num = parallel_get_max_threads() > 1 ? parallel_get_max_threads() / 2 : 1lu; // TODO: WA for Hyper Threading
            static const size_t max_thr_num = 2lu;
            size_t thr_num = std::min(size / min_wa_per_thread, max_thr_num);
            // const size_t k_size = thr_num * 8lu + 4lu;
            // if (k_array.size() < k_size) {
            //     auto prev_size = k_array.size();
            //     k_array.resize(k_size);
            //     for (size_t i = prev_size; i < k_size; i++) {
            //         k_array[i] = jit::gen_k_value(i - 3lu);
            //     }
            // }
            // const size_t block_size = thr_num * jit::Generator::zmm_len; //kernels[0]->get_vlen(); // TODO: vlen
            // const uint64_t blocks = size / block_size;
            // const uint64_t el_per_thread = block_size * ((blocks + thr_num - 1) / thr_num);
            // const uint64_t el_per_thread = block_size * ( (size / block_size) / thr_num );
            const uint64_t el_per_thread = Generator::zmm_len * ( (size / thr_num) / Generator::zmm_len);
            std::vector<uint64_t> intermediate(thr_num * 8lu); // zmm_len * thr_num

// std::vector<uint64_t> tmp_vec(thr_num * 4, 0lu);
// std::vector<uint64_t> tmp_vec_2(thr_num * 4, 0lu);

            parallel_nt_static(thr_num, [&](const int ithr, const int nthr) {
                uint64_t start = ithr * el_per_thread;
                if (start >= size) {
                    return;
                }
                uint64_t work_amount = (el_per_thread + start > size) ? size - start : el_per_thread;
// printf("    [%d] start: %lu, work_amount: %lu\n", ithr, start, work_amount);

                jit::ComputeHashCallArgs args;

                args.src_ptr = reinterpret_cast<const uint8_t *>(src) + Generator::zmm_len * ithr;
                args.dst_ptr = &(intermediate[8lu * ithr]);
                args.k_ptr = jit::K_PULL;
                args.work_amount = work_amount;
                args.size = size;
                args.threads_num = thr_num; // jit::Generator::xmm_len * 8lu;
// args.tmp_ptr = &(tmp_vec[ithr * 4]);

                if (ithr == 0) {
                    (*first_thr_kernel)(&args);
                } else {
                    (*n_thr_kernel)(&args);
                }
// printf("    [%d] start: %lu, work_amount: %lu; {%lu; %lu}\n", ithr, start, work_amount, tmp_vec[ithr * 4], tmp_vec[ithr * 4 + 1]);
// printf("    [%d] {%lu; %lu; %lu; %lu; %lu; %lu; %lu; %lu}\n", ithr,
//     intermediate[ithr * 8 + 0], intermediate[ithr * 8 + 1], intermediate[ithr * 8 + 2], intermediate[ithr * 8 + 3],
//     intermediate[ithr * 8 + 4], intermediate[ithr * 8 + 5], intermediate[ithr * 8 + 6], intermediate[ithr * 8 + 7]);
            });

            jit::ComputeHashCallArgs args;
            args.src_ptr = reinterpret_cast<const uint8_t *>(src) + size - args.work_amount;
            args.dst_ptr = &result;
            args.k_ptr = jit::K_PULL;
            args.work_amount = size - el_per_thread * thr_num; //((size + el_per_thread - 1) / el_per_thread) * jit::Generator::xmm_len;
            args.size = size;
            args.intermediate_ptr = intermediate.data();
// args.tmp_ptr = tmp_vec_2.data();
// printf("    FINAL_FOLD work_amount: %lu\n", args.work_amount);

            (*final_fold_kernel)(&args);
        } else {
            static auto single_thr_kernel = Generator::mayiuse(avx512_core) 
                ? jit::ComputeHash<avx512_core>::create({jit::SINGLE_THREAD}) : jit::ComputeHash<avx2>::create({jit::SINGLE_THREAD});
// std::vector<uint64_t> tmp_vec(4, 0lu);
// static std::vector<uint64_t> intermediate(2 * 8lu); // zmm_len * thr_num

            jit::ComputeHashCallArgs args;
            args.src_ptr = src;
            args.dst_ptr = &result;
            args.k_ptr = jit::K_PULL;
            args.work_amount = size;
            args.size = size;
// args.intermediate_ptr = intermediate.data();
// args.tmp_ptr = &(tmp_vec[0]);

// t1 = std::chrono::high_resolution_clock::now();
            (*single_thr_kernel)(&args);
// t2 = std::chrono::high_resolution_clock::now();

// if (size > 200000) {
// printf("    {%lu; %lu; %lu; %lu}\n", tmp_vec[0], tmp_vec[1], tmp_vec[2], tmp_vec[3]);
// printf("    {%lu; %lu; %lu; %lu; %lu; %lu; %lu; %lu; %lu; %lu; %lu; %lu; %lu; %lu; %lu; %lu}\n",
//     intermediate[0], intermediate[1], intermediate[2], intermediate[3],
//     intermediate[4], intermediate[5], intermediate[6], intermediate[7],
//     intermediate[8], intermediate[9], intermediate[10], intermediate[11],
//     intermediate[12], intermediate[13], intermediate[14], intermediate[15]);
// }
        }

// t2 = std::chrono::high_resolution_clock::now();
// if (size >= 131072 * 2) {
// // if (size <= 16lu) {
// // if (size == 4lu) {
//     auto ms_int = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1);
//     counter++;
//     sum += ms_int.count();
// //     // if (ms_int.count() > 100)
//     std::cout << "[" << counter << "] compute_hash time: " << ms_int.count() << "; sum: " << sum << "; size: " << size << "; avg_time: " << sum / counter << " nanosec" << std::endl;
// }
// if (size >= 131072 * 2)
    printf("    res: %lu\n", result);
        return result;
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

//                 jit::ComputeHashCallArgs args;

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
// //         jit::ComputeHashCallArgs args;

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

//             jit::ComputeHashCallArgs args;
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

//             jit::ComputeHashCallArgs args;
//             args.src_ptr = src;
//             args.dst_ptr = &res;
//             args.work_amount = size;
//             args.make_64_fold = 1lu;
// args.tmp_ptr = &(tmp_vec[0]);

//             kernel(&args);

// // if (size > 200000lu) {
//     // std::cout << "compute_hash size: " << size << "; tmp_vec: {" << tmp_vec[0] << "; " << tmp_vec[1] << "}" << std::endl;
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
// //     std::cout << "compute_hash(" << counter << ") kernel res: " << res << "; size: " << size << std::endl;
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
//     std::cout << "[" << counter << "] compute_hash time: " << ms_int.count() << "; sum: " << sum << "; size: " << size << "; avg_time: " << sum / counter << " nanosec" << std::endl;
//     // std::cout << ms_int.count() << std::endl;
// printf("    res: %lu\n", res);
//         return res;
//     }
#endif // OPENVINO_ARCH_X86 || OPENVINO_ARCH_X86_64

    constexpr auto cel_size = sizeof(size_t);
    size_t seed = size;
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
//     std::cout << "compute_hash ref res: " << seed << "; size: " << size << std::endl;
    return seed;
}

}   // namespace runtime
}   // namespace ov
