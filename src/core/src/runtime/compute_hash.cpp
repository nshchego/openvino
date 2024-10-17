// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// The CRC computation is used for x86.
// The calculations were taken from the article
// "Fast CRC Computation for Generic Polynomials Using PCLMULQDQ Instruction - Intel (December, 2009)".

#include "openvino/core/visibility.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/runtime/compute_hash.hpp"

#include <cmath>

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

class HashBase : public Generator {
protected:
    void (*ker_fn)(const ComputeHashCallArgs*);
public:
    HashBase(cpu_isa_t isa) : Generator(isa) {}

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
    explicit ComputeHash(const ComputeHashCompileParams& jcp) : HashBase(isa), m_jcp(jcp) {
        if (!mayiuse(cpu_isa_t::pclmulqdq)) {
            OPENVINO_THROW("The current CPU does not support pclmulqdq instruction, which is required for the CRC algorithm.");
        }
        if (mayiuse(cpu_isa_t::vpclmulqdq)) {
printf("[CPU][ComputeHash] supports vpclmulqdq\n");
            is_vpclmulqdq = true;
        }
    }

    void generate() override {
        m_registers_pool = RegistersPool::create(isa, {rax, rcx, rsp, rdi, k0});

        r64_src_ptr     = getReg64();
        r64_dst_ptr     = getReg64();
        r64_work_amount = getReg64();
        r64_k_ptr       = getReg64();
        r64_aux         = getReg64();
        v_k_2_3         = getVmm();
        v_shuf_mask     = getVmm();
        auto v_dst      = getVmm();
r64_tmp = getReg64();

        this->preamble();

        initialize(v_dst);
        bulk_fold(v_dst);
        join(v_dst);
        fold_to_128(v_dst);
        fold_to_64(v_dst);

        this->postamble();
        m_registers_pool.reset();
    }

    static std::shared_ptr<HashBase> create(const ComputeHashCompileParams& params) {
        auto kernel = std::make_shared<ComputeHash>(params);
        OPENVINO_ASSERT(kernel, "[ CORE ] Could not create ComputeHash kernel.");
        kernel->create_kernel();

        return kernel;
    }

private:
    using Vmm = typename std::conditional<isa == avx512_core, Xbyak::Zmm, Xbyak::Ymm>::type;
    bool is_vpclmulqdq = false;

    ComputeHashCompileParams m_jcp;
    RegistersPool::Ptr m_registers_pool;
    
    const Xbyak::Reg64 r64_params = abi_param1;

    RegistersPool::Reg<Xbyak::Reg64> r64_src_ptr;
    RegistersPool::Reg<Xbyak::Reg64> r64_dst_ptr;
    RegistersPool::Reg<Xbyak::Reg64> r64_work_amount;
    RegistersPool::Reg<Xbyak::Reg64> r64_k_ptr;
    RegistersPool::Reg<Xbyak::Reg64> r64_aux;
RegistersPool::Reg<Xbyak::Reg64> r64_tmp;

    // Vector registers
    RegistersPool::Reg<Vmm> v_k_2_3;
    RegistersPool::Reg<Vmm> v_shuf_mask;

    void initialize(const Vmm& v_dst);

    void bulk_fold(const Vmm& v_dst);

    void join(const Vmm& v_dst);

    void fold_to_128(const Vmm& v_dst);

    void fold_to_64(const Vmm& v_dst);

    void uni_vpxorq(const Xbyak::Xmm& v_dst, const Xbyak::Xmm& v_src_0, const Xbyak::Xmm& v_src_1);

    void uni_vmovdqu64(const Xbyak::Xmm& v_dst, const Xbyak::Operand& v_src_0);

    void uni_vmovdqu64(const Xbyak::Address& v_dst, const Xbyak::Xmm& v_src_0);

    void uni_vbroadcasti64x2(const Xbyak::Ymm& v_dst, const Xbyak::Address& v_src_0);

    void partial_load(const Xbyak::Xmm& xmm_dst, const Xbyak::Address& src_addr, const Xbyak::Reg64& r64_load_num);

    void partial_load(const Xbyak::Ymm& ymm_dst, const Xbyak::Address& src_addr, const Xbyak::Reg64& r64_load_num);
    
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

        const auto& r64_src_ptr   = r8;
        const auto& r64_dst_ptr   = r9;
        const auto& r64_aux   = r10;

        const auto& xmm_dst   = xmm0;
        const auto& xmm_aux   = xmm1;
        const auto& xmm_aux_1 = xmm2;
        const auto& xmm_aux_2 = xmm3;

        this->preamble();

        mov(r64_src_ptr, ptr[r64_params + GET_OFF(src_ptr)]);
        mov(r64_dst_ptr, ptr[r64_params + GET_OFF(dst_ptr)]);

        mov(r64_aux, 4);
        vpinsrq(xmm_aux, xmm_aux, r64_aux, 0x0);
        mov(r64_aux, CRC_VAL);
        vpinsrq(xmm_aux, xmm_aux, r64_aux, 0x1);

        vpxor(xmm_dst, xmm_dst, xmm_dst);
        // mov(r64_aux, ptr[r64_params + GET_OFF(src_ptr)]);
        vpinsrd(xmm_dst, xmm_dst, ptr[r64_src_ptr], 0x0);
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
        // uni_vmovdqu64(xmm_aux_2, ptr[r64_aux + K_P_P_OFF]);
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
        vpextrq(ptr[r64_dst_ptr], xmm_dst, 0x0);

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
void ComputeHash<avx512_core>::uni_vmovdqu64(const Xbyak::Xmm& v_dst, const Xbyak::Operand& v_src_0) {
    vmovdqu64(v_dst, v_src_0);
}
template <cpu_isa_t isa>
void ComputeHash<isa>::uni_vmovdqu64(const Xbyak::Xmm& v_dst, const Xbyak::Operand& v_src_0) {
    vmovdqu(v_dst, v_src_0);
}
template <>
void ComputeHash<avx512_core>::uni_vmovdqu64(const Xbyak::Address& v_dst, const Xbyak::Xmm& v_src_0) {
    vmovdqu64(v_dst, v_src_0);
}
template <cpu_isa_t isa>
void ComputeHash<isa>::uni_vmovdqu64(const Xbyak::Address& v_dst, const Xbyak::Xmm& v_src_0) {
    vmovdqu(v_dst, v_src_0);
}
template <>
void ComputeHash<avx512_core>::uni_vbroadcasti64x2(const Xbyak::Ymm& v_dst, const Xbyak::Address& v_src_0) {
    vbroadcasti64x2(v_dst, v_src_0);
}
template <cpu_isa_t isa>
void ComputeHash<isa>::uni_vbroadcasti64x2(const Xbyak::Ymm& v_dst, const Xbyak::Address& v_src_0) {
    vbroadcasti128(v_dst, v_src_0);
}
template <>
void ComputeHash<avx512_core>::partial_load(const Xbyak::Xmm&     xmm_dst,
                                            const Xbyak::Address& src_addr,
                                            const Xbyak::Reg64&   r64_load_num) {
    Xbyak::Label l_mv_mask;
    auto rOnes = getReg64();
    auto k_load_mask = RegistersPool::Reg<Xbyak::Opmask>(m_registers_pool);

    mov(rOnes, 0xFFFFFFFFFFFFFFFF);
    cmp(r64_load_num, 0x3f);
    jg(l_mv_mask);

    shlx(rOnes, rOnes, r64_load_num);
    not_(rOnes);

    L(l_mv_mask);
    kmovq(k_load_mask, rOnes);

    vmovdqu8(Vmm(xmm_dst.getIdx()) | k_load_mask | T_z, ptr[r64_src_ptr]);
}
template <cpu_isa_t isa>
void ComputeHash<isa>::partial_load(const Xbyak::Xmm&     xmm_dst,
                                    const Xbyak::Address& src_addr,
                                    const Xbyak::Reg64&   r64_load_num) {
    Xbyak::Label l_partial, l_end;

    cmp(r64_load_num, xmm_len);
    jl(l_partial, T_NEAR);
    uni_vmovdqu64(xmm_dst, ptr[src_addr.getRegExp()]);
    jmp(l_end, T_NEAR);

    L(l_partial); {
        uni_vpxorq(xmm_dst, xmm_dst, xmm_dst);
        for (size_t j = 0lu; j < xmm_len - 1; j++) {
            cmp(r64_load_num, j);
            jle(l_end, T_NEAR);
            pinsrb(xmm_dst, ptr[src_addr.getRegExp() + j], j);
        }
    }

    L(l_end);
}
template <>
void ComputeHash<avx512_core>::partial_load(const Xbyak::Ymm&     xmm_dst,
                                            const Xbyak::Address& src_addr,
                                            const Xbyak::Reg64&   r64_load_num) {
    partial_load(Xbyak::Xmm(xmm_dst.getIdx()), src_addr, r64_load_num);
}
template <cpu_isa_t isa>
void ComputeHash<isa>::partial_load(const Xbyak::Ymm&     ymm_dst,
                                    const Xbyak::Address& src_addr,
                                    const Xbyak::Reg64&   r64_load_num) {
    Xbyak::Label l_xmm, l_partial, l_end;
    auto xmm_dst = Xbyak::Xmm(ymm_dst.getIdx());

    cmp(r64_load_num, ymm_len);
    jl(l_xmm, T_NEAR);
    uni_vmovdqu64(ymm_dst, ptr[src_addr.getRegExp()]);
    jmp(l_end, T_NEAR);

    L(l_xmm);
    uni_vpxorq(ymm_dst, ymm_dst, ymm_dst);
    cmp(r64_load_num, xmm_len);
    jl(l_partial, T_NEAR);
    uni_vmovdqu64(xmm_dst, ptr[src_addr.getRegExp()]);
    je(l_end, T_NEAR);

    {
        Xbyak::Label l_rest_loop, l_perm;

        vperm2i128(ymm_dst, ymm_dst, ymm_dst, 0x1);
        for (size_t j = 0lu; j < xmm_len - 1; j++) {
            cmp(r64_load_num, xmm_len + j);
            jle(l_perm, T_NEAR);
            pinsrb(xmm_dst, ptr[src_addr.getRegExp() + xmm_len + j], j);
        }
        L(l_perm);
        vperm2i128(ymm_dst, ymm_dst, ymm_dst, 0x1);
    }
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

template <cpu_isa_t isa>
void ComputeHash<isa>::initialize(const Vmm& v_dst) {
    mov(r64_src_ptr,     ptr[r64_params + GET_OFF(src_ptr)]);
    mov(r64_dst_ptr,     ptr[r64_params + GET_OFF(dst_ptr)]);
    mov(r64_k_ptr,       ptr[r64_params + GET_OFF(k_ptr)]);
    mov(r64_work_amount, ptr[r64_params + GET_OFF(work_amount)]);
mov(r64_tmp, ptr[r64_params + GET_OFF(tmp_ptr)]);

    uni_vbroadcasti64x2(v_k_2_3, ptr[r64_k_ptr + K_2_3_OFF]);

    mov(r64_aux, reinterpret_cast<uintptr_t>(SHUF_MASK));
    uni_vbroadcasti64x2(v_shuf_mask, ptr[r64_aux]);

    if (m_jcp.type == SINGLE_THREAD || m_jcp.type == FIRST_THREAD) {
        auto xmm_dst = Xbyak::Xmm(v_dst.getIdx());
        auto xmm_aux = getXmm();

        // Initial CRC
        mov(r64_aux, ptr[r64_params + GET_OFF(size)]);
        vpinsrq(xmm_aux, xmm_aux, r64_aux, 0x0);
        mov(r64_aux, CRC_VAL);
        vpinsrq(xmm_aux, xmm_aux, r64_aux, 0x1);

        // First xor with source.
        partial_load(v_dst, ptr[r64_src_ptr], r64_work_amount);
        vpshufb(v_dst, v_dst, v_shuf_mask);
        pxor(xmm_dst, xmm_aux); // The SSE version is used to avoid zeroing out the rest of the Vmm.
        if (m_jcp.type == SINGLE_THREAD) {
            add(r64_src_ptr, xmm_len);
        }
    } else if (m_jcp.type == N_THREAD) {
        uni_vmovdqu64(v_dst, ptr[r64_src_ptr]);
        vpshufb(v_dst, v_dst, v_shuf_mask);
    }
    if (m_jcp.type == SINGLE_THREAD || m_jcp.type == FIRST_THREAD || m_jcp.type == N_THREAD) {
        sub(r64_work_amount, xmm_len);
    }
}

template <>
void ComputeHash<avx512_core>::bulk_fold(const Vmm& v_dst) {
    if (m_jcp.type != SINGLE_THREAD && m_jcp.type != FIRST_THREAD && m_jcp.type != N_THREAD) {
        return;
    }
    Xbyak::Label l_fold_loop, l_end;
    cmp(r64_work_amount, 2 * get_vlen() - xmm_len);
    jl(l_end, T_NEAR);

    auto v_src_0  = getVmm();
    auto v_dst_0  = getVmm();
    auto v_dst_1  = getVmm();
    auto v_dst_2  = getVmm();
    auto& v_dst_3 = v_dst;
    auto v_k_loop = getVmm();
    auto v_aux_0  = getVmm();

    auto xmm_src_0  = Xbyak::Xmm(v_src_0.getIdx());
    auto xmm_src_1  = getXmm();
    auto xmm_dst_0  = Xbyak::Xmm(v_dst_0.getIdx());
    auto xmm_dst_1  = Xbyak::Xmm(v_dst_1.getIdx());
    auto xmm_dst_2  = Xbyak::Xmm(v_dst_2.getIdx());
    auto xmm_dst_3  = Xbyak::Xmm(v_dst_3.getIdx());
    auto xmm_k_loop = Xbyak::Xmm(v_k_loop.getIdx());
    auto xmm_k_2_3  = Xbyak::Xmm(v_k_2_3.getIdx());
    auto xmm_aux_0  = Xbyak::Xmm(v_aux_0.getIdx());

    RegistersPool::Reg<Xbyak::Reg64> r64_bulk_step;
    if (m_jcp.type == FIRST_THREAD || m_jcp.type == N_THREAD) {
        r64_bulk_step = getReg64();
        mov(r64_bulk_step, ptr[r64_params + GET_OFF(threads_num)]);
        sal(r64_bulk_step, static_cast<int>(std::log2(get_vlen()))); // * vlen
    }

    if (m_jcp.type == SINGLE_THREAD) {
        uni_vbroadcasti64x2(v_k_loop, ptr[r64_k_ptr + K_8_9_OFF]);
    } else {
        uni_vbroadcasti64x2(v_k_loop, ptr[r64_k_ptr + K_16_17_OFF]);
    }

    uni_vmovdqu64(v_dst_0, v_dst);

    if (!is_vpclmulqdq) {
        vextracti64x2(xmm_dst_1, v_dst_0, 0x1);
        vextracti64x2(xmm_dst_2, v_dst_0, 0x2);
        vextracti64x2(xmm_dst_3, v_dst_0, 0x3);
    }

    if (m_jcp.type == FIRST_THREAD || m_jcp.type == N_THREAD) {
        add(r64_src_ptr, r64_bulk_step);
        // prefetcht2(ptr[r64_src_ptr + 8192]);
        prefetcht2(ptr[r64_src_ptr + 16384]);
    } else {
        add(r64_src_ptr, get_vlen() - xmm_len);
        prefetcht2(ptr[r64_src_ptr + 4096]);
        // prefetcht2(ptr[r64_src_ptr + 16384]);
    }
    prefetcht1(ptr[r64_src_ptr + 1024]);
    // prefetcht0(ptr[r64_src_ptr + 1024]);
    prefetcht0(ptr[r64_src_ptr + 64]);
    // prefetcht0(ptr[r64_src_ptr]);

    sub(r64_work_amount, 2 * get_vlen() - xmm_len);
    // sub(r64_work_amount, 2 * get_vlen());

    L(l_fold_loop); {
        uni_vmovdqu64(v_src_0, ptr[r64_src_ptr]);
        vpshufb(v_src_0, v_src_0, v_shuf_mask);

        if (m_jcp.type == FIRST_THREAD || m_jcp.type == N_THREAD) {
            add(r64_src_ptr, r64_bulk_step);
            // prefetcht2(ptr[r64_src_ptr + 8192]);
            prefetcht2(ptr[r64_src_ptr + 16384]);
        } else {
            add(r64_src_ptr, get_vlen());
            prefetcht2(ptr[r64_src_ptr + 4096]);
            // prefetcht2(ptr[r64_src_ptr + 8192]);
            // prefetcht2(ptr[r64_src_ptr + 16384]);
        }
        prefetcht1(ptr[r64_src_ptr + 1024]);
        // prefetcht0(ptr[r64_src_ptr + 1024]);
        prefetcht0(ptr[r64_src_ptr + 64]);

        if (is_vpclmulqdq) {
            vpclmulqdq(v_aux_0, v_dst_0, v_k_loop, 0b00000000);
            vpclmulqdq(v_dst_0, v_dst_0, v_k_loop, 0b00010001);
            uni_vpxorq(v_aux_0, v_aux_0, v_src_0);
            uni_vpxorq(v_dst_0, v_dst_0, v_aux_0);
        } else {
            // 0
            vpclmulqdq(xmm_aux_0, xmm_dst_0, xmm_k_loop, 0b00000000);
            vpclmulqdq(xmm_dst_0, xmm_dst_0, xmm_k_loop, 0b00010001);
            uni_vpxorq(xmm_aux_0, xmm_aux_0, xmm_src_0);
            uni_vpxorq(xmm_dst_0, xmm_dst_0, xmm_aux_0);

            // 1
            vextracti64x2(xmm_src_1, v_src_0, 0x1);
            vpclmulqdq(xmm_aux_0, xmm_dst_1, xmm_k_loop, 0b00000000);
            vpclmulqdq(xmm_dst_1, xmm_dst_1, xmm_k_loop, 0b00010001);
            uni_vpxorq(xmm_aux_0, xmm_aux_0, xmm_src_1);
            uni_vpxorq(xmm_dst_1, xmm_dst_1, xmm_aux_0);

            // 2
            vextracti64x2(xmm_src_1, v_src_0, 0x2);
            vpclmulqdq(xmm_aux_0, xmm_dst_2, xmm_k_loop, 0b00000000);
            vpclmulqdq(xmm_dst_2, xmm_dst_2, xmm_k_loop, 0b00010001);
            uni_vpxorq(xmm_aux_0, xmm_aux_0, xmm_src_1);
            uni_vpxorq(xmm_dst_2, xmm_dst_2, xmm_aux_0);

            // 3
            vextracti64x2(xmm_src_1, v_src_0, 0x3);
            vpclmulqdq(xmm_aux_0, xmm_dst_3, xmm_k_loop, 0b00000000);
            vpclmulqdq(xmm_dst_3, xmm_dst_3, xmm_k_loop, 0b00010001);
            uni_vpxorq(xmm_aux_0, xmm_aux_0, xmm_src_1);
            uni_vpxorq(xmm_dst_3, xmm_dst_3, xmm_aux_0);
        }

        sub(r64_work_amount, get_vlen());
        jge(l_fold_loop, T_NEAR);
    }
    add(r64_work_amount, get_vlen());

    if (m_jcp.type == SINGLE_THREAD) {
        if (is_vpclmulqdq) {
            vextracti64x2(xmm_dst_1, v_dst_0, 0x1);
            vextracti64x2(xmm_dst_2, v_dst_0, 0x2);
            vextracti64x2(xmm_dst_3, v_dst_0, 0x3);
        }

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
    } else {
        if (is_vpclmulqdq) {
            uni_vmovdqu64(ptr[r64_dst_ptr], v_dst_0);
        } else {
            uni_vmovdqu64(ptr[r64_dst_ptr + xmm_len * 0lu], xmm_dst_0);
            uni_vmovdqu64(ptr[r64_dst_ptr + xmm_len * 1lu], xmm_dst_1);
            uni_vmovdqu64(ptr[r64_dst_ptr + xmm_len * 2lu], xmm_dst_2);
            uni_vmovdqu64(ptr[r64_dst_ptr + xmm_len * 3lu], xmm_dst_3);
        }
    }

    L(l_end);
}

template <cpu_isa_t isa>
void ComputeHash<isa>::bulk_fold(const Vmm& v_dst) {
    if (m_jcp.type != SINGLE_THREAD && m_jcp.type != FIRST_THREAD && m_jcp.type != N_THREAD) {
        return;
    }
    Xbyak::Label l_fold_loop, l_end;
    cmp(r64_work_amount, 2 * get_vlen() - xmm_len);
    jl(l_end, T_NEAR);

    auto v_src_0 = getVmm();
    auto v_dst_0 = getVmm();
    auto& v_dst_1 = v_dst;
    auto v_aux_0 = getVmm();
    auto v_k_loop = getVmm();

    auto xmm_src_0  = Xbyak::Xmm(v_src_0.getIdx());
    auto xmm_src_1  = getXmm();
    auto xmm_dst_0  = Xbyak::Xmm(v_dst_0.getIdx());
    auto xmm_dst_1  = Xbyak::Xmm(v_dst_1.getIdx());
    auto xmm_k_loop = Xbyak::Xmm(v_k_loop.getIdx());
    auto xmm_k_2_3  = Xbyak::Xmm(v_k_2_3.getIdx());
    auto xmm_aux_0  = Xbyak::Xmm(v_aux_0.getIdx());

    RegistersPool::Reg<Xbyak::Reg64> r64_bulk_step;
    if (m_jcp.type == FIRST_THREAD || m_jcp.type == N_THREAD) {
        r64_bulk_step = getReg64();
        mov(r64_bulk_step, ptr[r64_params + GET_OFF(threads_num)]);
        sal(r64_bulk_step, static_cast<int>(std::log2(get_vlen()))); // * vlen
    }

    if (m_jcp.type == SINGLE_THREAD) {
        uni_vbroadcasti64x2(v_k_loop, ptr[r64_k_ptr + K_4_5_OFF]);
    } else {
        uni_vbroadcasti64x2(v_k_loop, ptr[r64_k_ptr + K_8_9_OFF]);
    }

    uni_vmovdqu64(v_dst_0, v_dst);

    if (!is_vpclmulqdq) {
        vextracti128(xmm_dst_1, v_dst_0, 0x1);
    }

    if (m_jcp.type == SINGLE_THREAD) {
        add(r64_src_ptr, get_vlen() - xmm_len);
    } else {
        add(r64_src_ptr, r64_bulk_step);
    }
    prefetcht2(ptr[r64_src_ptr + 4096]);
    prefetcht1(ptr[r64_src_ptr + 1024]);
    prefetcht0(ptr[r64_src_ptr + 64]);

    sub(r64_work_amount, 2 * get_vlen() - xmm_len);

    L(l_fold_loop); {
        uni_vmovdqu64(v_src_0, ptr[r64_src_ptr]);
        vpshufb(v_src_0, v_src_0, v_shuf_mask);
        
        if (m_jcp.type == SINGLE_THREAD) {
            add(r64_src_ptr, get_vlen());
        } else {
            add(r64_src_ptr, r64_bulk_step);
        }
        prefetcht2(ptr[r64_src_ptr + 4096]);
        prefetcht1(ptr[r64_src_ptr + 1024]);
        prefetcht0(ptr[r64_src_ptr + 64]);

        if (is_vpclmulqdq) {
            vpclmulqdq(v_aux_0, v_dst_0, v_k_loop, 0b00000000);
            vpclmulqdq(v_dst_0, v_dst_0, v_k_loop, 0b00010001);
            uni_vpxorq(v_aux_0, v_aux_0, v_src_0);
            uni_vpxorq(v_dst_0, v_dst_0, v_aux_0);
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
        }

        sub(r64_work_amount, get_vlen());
        jge(l_fold_loop, T_NEAR);
    }
    add(r64_work_amount, get_vlen());

    if (m_jcp.type == SINGLE_THREAD) {
        if (is_vpclmulqdq) {
            vextracti128(xmm_dst_1, v_dst_0, 0x1);
        }
        vpclmulqdq(xmm_aux_0, xmm_dst_0, xmm_k_2_3, 0b00000000);
        vpclmulqdq(xmm_dst_0, xmm_dst_0, xmm_k_2_3, 0b00010001);
        uni_vpxorq(xmm_dst_1, xmm_dst_1, xmm_aux_0);
        uni_vpxorq(xmm_dst_1, xmm_dst_1, xmm_dst_0);
    } else {
        if (is_vpclmulqdq) {
            uni_vmovdqu64(ptr[r64_dst_ptr], v_dst_0);
        } else {
            uni_vmovdqu64(ptr[r64_dst_ptr + xmm_len * 0lu], xmm_dst_0);
            uni_vmovdqu64(ptr[r64_dst_ptr + xmm_len * 1lu], xmm_dst_1);
        }
    }

    L(l_end);
}

template <>
void ComputeHash<avx512_core>::join(const Vmm& v_dst) {
    if (m_jcp.type != FINAL_FOLD) {
        return;
    }
    
    mov(r64_aux, ptr[r64_params + GET_OFF(intermediate_ptr)]);
    prefetcht0(ptr[r64_aux + 1024]);

    auto xmm_src_0 = getXmm();
    auto xmm_src_last = Xbyak::Xmm(v_dst.getIdx());
    auto xmm_aux_0 = getXmm();
    auto xmm_k_2_3 = Xbyak::Xmm(v_k_2_3.getIdx());

    uni_vmovdqu64(xmm_src_last, ptr[r64_aux + xmm_len * 7]);

    uni_vmovdqu64(xmm_src_0, ptr[r64_aux]);
    vpclmulqdq(xmm_aux_0, xmm_src_0, ptr[r64_k_ptr + K_14_15_OFF], 0b00000000);
    vpclmulqdq(xmm_src_0, xmm_src_0, ptr[r64_k_ptr + K_14_15_OFF], 0b00010001);
    uni_vpxorq(xmm_src_last, xmm_src_last, xmm_aux_0);
    uni_vpxorq(xmm_src_last, xmm_src_last, xmm_src_0);

    uni_vmovdqu64(xmm_src_0, ptr[r64_aux + xmm_len]);
    vpclmulqdq(xmm_aux_0, xmm_src_0, ptr[r64_k_ptr + K_12_13_OFF], 0b00000000);
    vpclmulqdq(xmm_src_0, xmm_src_0, ptr[r64_k_ptr + K_12_13_OFF], 0b00010001);
    uni_vpxorq(xmm_src_last, xmm_src_last, xmm_aux_0);
    uni_vpxorq(xmm_src_last, xmm_src_last, xmm_src_0);

    uni_vmovdqu64(xmm_src_0, ptr[r64_aux + xmm_len * 2lu]);
    vpclmulqdq(xmm_aux_0, xmm_src_0, ptr[r64_k_ptr + K_10_11_OFF], 0b00000000);
    vpclmulqdq(xmm_src_0, xmm_src_0, ptr[r64_k_ptr + K_10_11_OFF], 0b00010001);
    uni_vpxorq(xmm_src_last, xmm_src_last, xmm_aux_0);
    uni_vpxorq(xmm_src_last, xmm_src_last, xmm_src_0);

    uni_vmovdqu64(xmm_src_0, ptr[r64_aux + xmm_len * 3lu]);
    vpclmulqdq(xmm_aux_0, xmm_src_0, ptr[r64_k_ptr + K_8_9_OFF], 0b00000000);
    vpclmulqdq(xmm_src_0, xmm_src_0, ptr[r64_k_ptr + K_8_9_OFF], 0b00010001);
    uni_vpxorq(xmm_src_last, xmm_src_last, xmm_aux_0);
    uni_vpxorq(xmm_src_last, xmm_src_last, xmm_src_0);

    uni_vmovdqu64(xmm_src_0, ptr[r64_aux + xmm_len * 4lu]);
    vpclmulqdq(xmm_aux_0, xmm_src_0, ptr[r64_k_ptr + K_6_7_OFF], 0b00000000);
    vpclmulqdq(xmm_src_0, xmm_src_0, ptr[r64_k_ptr + K_6_7_OFF], 0b00010001);
    uni_vpxorq(xmm_src_last, xmm_src_last, xmm_aux_0);
    uni_vpxorq(xmm_src_last, xmm_src_last, xmm_src_0);

    uni_vmovdqu64(xmm_src_0, ptr[r64_aux + xmm_len * 5lu]);
    vpclmulqdq(xmm_aux_0, xmm_src_0, ptr[r64_k_ptr + K_4_5_OFF], 0b00000000);
    vpclmulqdq(xmm_src_0, xmm_src_0, ptr[r64_k_ptr + K_4_5_OFF], 0b00010001);
    uni_vpxorq(xmm_src_last, xmm_src_last, xmm_aux_0);
    uni_vpxorq(xmm_src_last, xmm_src_last, xmm_src_0);

    uni_vmovdqu64(xmm_src_0, ptr[r64_aux + xmm_len * 6lu]);
    vpclmulqdq(xmm_aux_0, xmm_src_0, xmm_k_2_3, 0b00000000);
    vpclmulqdq(xmm_src_0, xmm_src_0, xmm_k_2_3, 0b00010001);
    uni_vpxorq(xmm_src_last, xmm_src_last, xmm_aux_0);
    uni_vpxorq(xmm_src_last, xmm_src_last, xmm_src_0);
}

template <cpu_isa_t isa>
void ComputeHash<isa>::join(const Vmm& v_dst) {
    if (m_jcp.type != FINAL_FOLD) {
        return;
    }

    mov(r64_aux, ptr[r64_params + GET_OFF(intermediate_ptr)]);
    prefetcht0(ptr[r64_aux + 1024]);

    auto xmm_src_0 = getXmm();
    auto xmm_src_last = Xbyak::Xmm(v_dst.getIdx());
    auto xmm_aux_0 = getXmm();
    auto xmm_k_2_3 = Xbyak::Xmm(v_k_2_3.getIdx());

    uni_vmovdqu64(xmm_src_last, ptr[r64_aux + xmm_len * 3]);

    uni_vmovdqu64(xmm_src_0, ptr[r64_aux + xmm_len * 0lu]);
    vpclmulqdq(xmm_aux_0, xmm_src_0, ptr[r64_k_ptr + K_6_7_OFF], 0b00000000);
    vpclmulqdq(xmm_src_0, xmm_src_0, ptr[r64_k_ptr + K_6_7_OFF], 0b00010001);
    uni_vpxorq(xmm_src_last, xmm_src_last, xmm_aux_0);
    uni_vpxorq(xmm_src_last, xmm_src_last, xmm_src_0);

    uni_vmovdqu64(xmm_src_0, ptr[r64_aux + xmm_len * 1lu]);
    vpclmulqdq(xmm_aux_0, xmm_src_0, ptr[r64_k_ptr + K_4_5_OFF], 0b00000000);
    vpclmulqdq(xmm_src_0, xmm_src_0, ptr[r64_k_ptr + K_4_5_OFF], 0b00010001);
    uni_vpxorq(xmm_src_last, xmm_src_last, xmm_aux_0);
    uni_vpxorq(xmm_src_last, xmm_src_last, xmm_src_0);

    uni_vmovdqu64(xmm_src_0, ptr[r64_aux + xmm_len * 2lu]);
    vpclmulqdq(xmm_aux_0, xmm_src_0, xmm_k_2_3, 0b00000000);
    vpclmulqdq(xmm_src_0, xmm_src_0, xmm_k_2_3, 0b00010001);
    uni_vpxorq(xmm_src_last, xmm_src_last, xmm_aux_0);
    uni_vpxorq(xmm_src_last, xmm_src_last, xmm_src_0);
}

template <cpu_isa_t isa>
void ComputeHash<isa>::fold_to_128(const Vmm& v_dst) {
    if (m_jcp.type != SINGLE_THREAD && m_jcp.type != FINAL_FOLD) {
        return;
    }
    Xbyak::Label l_fold_loop, l_end;
    cmp(r64_work_amount, xmm_len);
    jl(l_end, T_NEAR);

    auto xmm_src       = getXmm();
    auto xmm_dst       = Xbyak::Xmm(v_dst.getIdx());
    auto xmm_k_2_3     = Xbyak::Xmm(v_k_2_3.getIdx());
    auto xmm_shuf_mask = Xbyak::Xmm(v_shuf_mask.getIdx());
    auto xmm_aux       = getXmm();

    L(l_fold_loop); {
        uni_vmovdqu64(xmm_src, ptr[r64_src_ptr]);
        vpshufb(xmm_src, xmm_src, xmm_shuf_mask);

        vpclmulqdq(xmm_aux, xmm_dst, xmm_k_2_3, 0b00000000);
        vpclmulqdq(xmm_dst, xmm_dst, xmm_k_2_3, 0b00010001);
        uni_vpxorq(xmm_dst, xmm_dst, xmm_aux);
        uni_vpxorq(xmm_dst, xmm_dst, xmm_src);

        add(r64_src_ptr, xmm_len);
        sub(r64_work_amount, xmm_len);
        cmp(r64_work_amount, xmm_len);
        jge(l_fold_loop, T_NEAR);
    }

    L(l_end);
}

template <cpu_isa_t isa>
void ComputeHash<isa>::fold_to_64(const Vmm& v_dst) {
    if (m_jcp.type != SINGLE_THREAD && m_jcp.type != FINAL_FOLD) {
        return;
    }
    Xbyak::Label l_fold_to_64;
    cmp(r64_work_amount, 0);
    jle(l_fold_to_64, T_NEAR);

    auto xmm_src       = getXmm();
    auto xmm_dst       = Xbyak::Xmm(v_dst.getIdx());
    auto xmm_k_2_3     = Xbyak::Xmm(v_k_2_3.getIdx());
    auto xmm_shuf_mask = Xbyak::Xmm(v_shuf_mask.getIdx());
    auto xmm_aux       = getXmm();
    auto xmm_aux_1     = getXmm();
    auto xmm_aux_2     = getXmm();

    partial_load(xmm_src, ptr[r64_src_ptr], r64_work_amount);
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

    vpextrq(ptr[r64_dst_ptr], xmm_dst, 0x0);
}

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
// if (size > 131072 * 2)
    // printf("combine_hash size: %lu\n", size);
// static uint64_t counter = 0lu;
// static uint64_t sum = 0lu;
// // counter++;
// std::chrono::high_resolution_clock::time_point t1, t2;
// t1 = std::chrono::high_resolution_clock::now();
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

            // static bool initialized = false;
            // static std::shared_ptr<jit::HashBase> krnels[3];
            // if (!initialized) {
            //     initialized = true;
            //     parallel_nt_static(3, [&](const int ithr, const int nthr) {
            //         switch(ithr) {
            //             case 0: krnels[0] = Generator::mayiuse(avx512_core) ?
            //                 jit::ComputeHash<avx512_core>::create({jit::FIRST_THREAD}) : jit::ComputeHash<avx2>::create({jit::FIRST_THREAD});
            //             case 1: krnels[1] = Generator::mayiuse(avx512_core) ?
            //                 jit::ComputeHash<avx512_core>::create({jit::N_THREAD}) : jit::ComputeHash<avx2>::create({jit::N_THREAD});
            //             case 2: krnels[2] = Generator::mayiuse(avx512_core) ?
            //                 jit::ComputeHash<avx512_core>::create({jit::FINAL_FOLD}) : jit::ComputeHash<avx2>::create({jit::FINAL_FOLD});
            //         }
            //     });
            // }

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
            const uint64_t el_per_thread = first_thr_kernel->get_vlen() * ( (size / thr_num) / first_thr_kernel->get_vlen());
            std::vector<uint8_t> intermediate(thr_num * first_thr_kernel->get_vlen());

std::vector<uint64_t> tmp_vec(thr_num * 4, 0lu);
std::vector<uint64_t> tmp_vec_2(thr_num * 4, 0lu);

            parallel_nt_static(thr_num, [&](const int ithr, const int nthr) {
                uint64_t start = ithr * el_per_thread;
                if (start >= size) {
                    return;
                }
                uint64_t work_amount = (el_per_thread + start > size) ? size - start : el_per_thread;
// printf("    [%d] start: %lu, work_amount: %lu\n", ithr, start, work_amount);

                jit::ComputeHashCallArgs args;

                args.src_ptr = reinterpret_cast<const uint8_t *>(src) + first_thr_kernel->get_vlen() * ithr;
                args.dst_ptr = &(intermediate[ithr * first_thr_kernel->get_vlen()]);
                args.k_ptr = jit::K_PULL;
                args.work_amount = work_amount;
                args.size = size;
                args.threads_num = thr_num;
args.tmp_ptr = &(tmp_vec[ithr * 4]);

                if (ithr == 0) {
                    (*first_thr_kernel)(&args);
                    // (*krnels[0])(&args);
                } else {
                    (*n_thr_kernel)(&args);
                    // (*krnels[1])(&args);
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
args.tmp_ptr = tmp_vec_2.data();
// printf("    FINAL_FOLD work_amount: %lu\n", args.work_amount);

            (*final_fold_kernel)(&args);
            // (*krnels[2])(&args);
        } else {
            static auto single_thr_kernel = Generator::mayiuse(avx512_core) 
                ? jit::ComputeHash<avx512_core>::create({jit::SINGLE_THREAD}) : jit::ComputeHash<avx2>::create({jit::SINGLE_THREAD});
std::vector<uint64_t> tmp_vec(4, 0lu);
// static std::vector<uint8_t> intermediate(2 * single_thr_kernel->get_vlen());

            jit::ComputeHashCallArgs args;
            args.src_ptr = src;
            args.dst_ptr = &result;
            args.k_ptr = jit::K_PULL;
            args.work_amount = size;
            args.size = size;
// args.intermediate_ptr = intermediate.data();
args.tmp_ptr = &(tmp_vec[0]);

// t1 = std::chrono::high_resolution_clock::now();
            (*single_thr_kernel)(&args);
// t2 = std::chrono::high_resolution_clock::now();

// if (size > 200000) {
// printf("    Single: %lu; tmp_vec {%lu; %lu; %lu; %lu}\n", size, tmp_vec[0], tmp_vec[1], tmp_vec[2], tmp_vec[3]);
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
    // printf("    res: %lu\n", result);
        return result;
    }

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

    return seed;
}

}   // namespace runtime
}   // namespace ov
