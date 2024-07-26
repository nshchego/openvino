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
        r64_work_amount = getReg64();

        mov(r64_src, ptr[r64_params + GET_OFF(src_ptr)]);
        mov(r64_dst, ptr[r64_params + GET_OFF(dst_ptr)]);
        mov(r64_work_amount, ptr[r64_params + GET_OFF(work_amount)]);

        initVectors();
        bulkFold(v_dst);
        restFold(v_dst);
        tailFold(v_dst);

        vpextrq(ptr[r64_dst], Xbyak::Xmm(v_dst.getIdx()), 0x0); // TODO 0x1?

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
    static const uint64_t CRC_VAL;
    static const uint64_t CONST_K[54];

    // using Vmm = typename std::conditional<true, Xbyak::Zmm, Xbyak::Ymm>::type;
    // using Vmm = typename std::conditional<true, Xbyak::Zmm, typename std::conditional<isa == sse41, Xbyak::Xmm, Xbyak::Ymm>::type>::type;
    using Vmm = typename std::conditional<isa == avx512_core, Xbyak::Zmm, typename std::conditional<isa == sse42, Xbyak::Xmm, Xbyak::Ymm>::type>::type;
    size_t vlen = xmm_len;
    bool is_vpclmulqdq = false;

    CombineHashCompileParams m_jcp;
    RegistersPool::Ptr registersPool;

    RegistersPool::Reg<Xbyak::Reg64> r64_src;
    RegistersPool::Reg<Xbyak::Reg64> r64_dst;
    RegistersPool::Reg<Xbyak::Reg64> r64_work_amount;

    const Xbyak::Reg64 r64_params = abi_param1;

    // Vector registers
    RegistersPool::Reg<Vmm> v_dst;
    RegistersPool::Reg<Vmm> v_k_1_2;
    // RegistersPool::Reg<Vmm> v_k_56;
    RegistersPool::Reg<Vmm> v_k_8_9;
    RegistersPool::Reg<Vmm> v_k_16_17;
    RegistersPool::Reg<Vmm> v_shuf_mask;

    // static const uint8_t shuf_mask[]

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
            // cmp(r64_work_amount, zmm_len); // TODO check
            jge(l_fold_loop, T_NEAR);
        }

        L(l_end);
    }

    void tailFold(const Vmm& v_dst);
};

template <cpu_isa_t isa>
void CombineHash<isa>::initVectors() {
    auto r64_aux = getReg64();

    v_dst = getVmm();
    auto xmm_dst = Xbyak::Xmm(v_dst.getIdx());
    mov(r64_aux, CRC_VAL);
    vpxorq(v_dst, v_dst, v_dst);
    vpinsrq(xmm_dst, xmm_dst, r64_aux, 0x1);

    v_k_1_2 = getVmm();
    mov(r64_aux, reinterpret_cast<uintptr_t>(CONST_K));
    vbroadcasti64x2(v_k_1_2, ptr[r64_aux]);
    v_k_8_9 = getVmm();
    mov(r64_aux, reinterpret_cast<uintptr_t>(CONST_K + 14));
    vbroadcasti64x2(v_k_8_9, ptr[r64_aux]);

    v_shuf_mask = getVmm();
    static const uint8_t shuf_mask[] = //{ 0b00000000, 0b00000001, 0b00000010, 0b00000011, 0b00000100, 0b00000101, 0b00000110, 0b00000111,
                                        //  0b00001000, 0b00001001, 0b00001010, 0b00001011, 0b00001100, 0b00001101, 0b00001110, 0b00001111,
                                        //  0b00000000, 0b00000001, 0b00000010, 0b00000011, 0b00000100, 0b00000101, 0b00000110, 0b00000111,
                                        //  0b00001000, 0b00001001, 0b00001010, 0b00001011, 0b00001100, 0b00001101, 0b00001110, 0b00001111 };
                                       { 0b00001111, 0b00001110, 0b00001101, 0b00001100, 0b00001011, 0b00001010, 0b00001001, 0b00001000,
                                         0b00000111, 0b00000110, 0b00000101, 0b00000100, 0b00000011, 0b00000010, 0b00000001, 0b00000000 };
    mov(r64_aux, reinterpret_cast<uintptr_t>(shuf_mask));
    if (isa == avx512_core) {
        // vmovdqu8(v_shuf_mask, ptr[r64_aux]);
        vbroadcasti64x2(v_shuf_mask, ptr[r64_aux]);
    } else {
        // vmovdqu(v_shuf_mask, ptr[r64_aux]);
        vbroadcasti128(v_shuf_mask, ptr[r64_aux]);
    }
}

template <>
void CombineHash<avx512_core>::bulkFold(const Vmm& v_dst) {
    Xbyak::Label l_fold_loop, l_end;
    cmp(r64_work_amount, 2 * zmm_len);
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

    vmovdqu64(v_dst_0, ptr[r64_src]);
    vpshufb(v_dst_0, v_dst_0, v_shuf_mask);
    pxor(xmm_dst_0, xmm_dst_3); // The SSE version is used to avoid zeroing out the rest of the vector.
    if (!is_vpclmulqdq) {
        vextracti64x2(xmm_dst_1, v_dst_0, 0x1);
        vextracti64x2(xmm_dst_2, v_dst_0, 0x2);
        vextracti64x2(xmm_dst_3, v_dst_0, 0x3);
    }

    add(r64_src, zmm_len);
    sub(r64_work_amount, zmm_len);

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

    if (is_vpclmulqdq) {
        // mov(r64_aux, reinterpret_cast<uintptr_t>(CONST_K + 16)); // TODO modify K table
        // vpclmulqdq(v_aux_0, v_dst_0, ptr[r64_aux], 0b00000000);
        // vpclmulqdq(v_dst_0, v_dst_0, ptr[r64_aux], 0b00010001);
        // vpxorq(v_dst_3, v_dst_3, v_aux_0);
        // vpxorq(v_dst_3, v_dst_3, v_dst_0);
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
    cmp(r64_work_amount, xmm_len);
    jl(l_end, T_NEAR);

    // auto xmm_dst = Xbyak::Xmm(v_dst.getIdx());
    // auto xmm_k_12 = Xbyak::Xmm(v_k_12.getIdx());
    // auto xmm_shuf_mask = Xbyak::Xmm(v_shuf_mask.getIdx());
    // auto xmm_src = getXmm();
    // auto xmm_aux = getXmm();

    // // prefetchnta(ptr[r64_src]); // TODO compare perf
    // vmovups(xmm_src, ptr[r64_src]);
    // vpshufb(xmm_src, xmm_src, xmm_shuf_mask); // Endianness swap
    // vxorps(xmm_dst, xmm_dst, xmm_src);

    // // Bulk fold
    // L(l_fold_loop); {
    //     add(r64_src, xmm_len);
    //     sub(r64_work_amount, xmm_len);
    //     cmp(r64_work_amount, xmm_len);
    //     jl(l_end, T_NEAR);

    //     vmovups(xmm_src, ptr[r64_src]);
    //     vpshufb(xmm_src, xmm_src, xmm_shuf_mask); // Endianness swap

    //     vpclmulqdq(xmm_aux, xmm_dst, xmm_k_12, 0b00000000);
    //     vpclmulqdq(xmm_dst, xmm_dst, xmm_k_12, 0b00010001);
    //     vxorps(xmm_src, xmm_src, xmm_dst);
    //     vxorps(xmm_dst, xmm_src, xmm_aux);

    //     jmp(l_fold_loop, T_NEAR);
    // }

    L(l_end);
}


template <>
void CombineHash<avx512_core>::tailFold(const Vmm& v_dst) {
    Xbyak::Label l_end;
    cmp(r64_work_amount, 0);
    jle(l_end, T_NEAR);

    auto r64_aux = getReg64();
    auto xmm_shuf_mask = Xbyak::Xmm(v_shuf_mask.getIdx());
    auto xmm_k_1_2 = Xbyak::Xmm(v_k_1_2.getIdx()); // TODO calc a new table for bytes
    auto xmm_src = getXmm();
    auto xmm_dst = Xbyak::Xmm(v_dst.getIdx());
    auto xmm_aux = getXmm();
    auto k_rest_mask = RegistersPool::Reg<Xbyak::Opmask>(registersPool);

    mov(r64_aux, 0xFFFFFFFFFFFFFFFF);
    shlx(r64_aux, r64_aux, r64_work_amount);
    not_(r64_aux);
    kmovq(k_rest_mask, r64_aux);

    vpxorq(xmm_src, xmm_src, xmm_src);
    vmovdqu64(Xbyak::Xmm(xmm_src.getIdx()) | k_rest_mask, ptr[r64_src]);
    vpshufb(xmm_src, xmm_src, xmm_shuf_mask); // Swap bytes

    shl(r64_work_amount, sizeof(uint64_t));
    mov(r64_aux, reinterpret_cast<uintptr_t>(CONST_K + 22));
    vpclmulqdq(xmm_aux, xmm_dst, ptr[r64_aux + r64_work_amount], 0b00000000);
    vpclmulqdq(xmm_dst, xmm_dst, xmm_k_1_2, 0b00010001);
    vpxorq(xmm_aux, xmm_aux, xmm_src);
    vpxorq(xmm_dst, xmm_dst, xmm_aux);

    // 128 -> 64
    vpclmulqdq(xmm_aux, xmm_dst, xmm_k_1_2, 0b00000000); // TODO KBP
    vpclmulqdq(xmm_dst, xmm_dst, xmm_k_1_2, 0b00010001);
    vpxorq(xmm_aux, xmm_aux, xmm_src);
    vpxorq(xmm_dst, xmm_dst, xmm_aux);

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

template <cpu_isa_t isa>
const uint64_t CombineHash<isa>::CONST_K[54] = { 0x05f5c3c7eb52fab6, 0x4eb938a7d257740e,  // x^(64*1), x^(64*2) U
                                                 0x05cf79dea9ac37d6, 0x001067e571d7d5c2,  // x^(64*15), x^(64*16)
                                                 0x05f5c3c7eb52fab6, 0x0000000000000000,  // x^(64*1), x^(64*)
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

}   // namespace jit
#endif // OPENVINO_ARCH_X86 || OPENVINO_ARCH_X86_64

size_t combine_hash(const void* src, size_t size) {
if (size == 0)
    std::cout << "combine_hash size: " << size << std::endl;
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
        // ov::parallel_nt(0, [&](const int ithr, const int nthr) {
        //     uint64_t work_amount = (size + jit::Generator::xmm_len - 1) / (2 * jit::Generator::xmm_len);
        // });
        size_t res = 0lu;
        jit::CombineHashCallArgs args;
        args.src_ptr = src;
        args.dst_ptr = &res;
        args.work_amount = size;
if (size > 16 && size < 32)
    std::cout << "combine_hash size: " << size << std::endl;
        kernel(&args);
// static uint64_t counter = 0lu;
// if (counter++ < 200)
//     std::cout << "combine_hash kernel res: " << res << "; size: " << size << std::endl;
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
