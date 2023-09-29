// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "random_uniform.hpp"

using namespace dnnl::impl::cpu;

namespace ov {
namespace intel_cpu {
namespace kernel {

#define GET_OFF(field) offsetof(RandomUniformCallArgs, field)

template <x64::cpu_isa_t isa>
RandomUniform<isa>::RandomUniform(const RandomUniformCompileParams& jcp) :
        JitKernel(jit_name(), jcp, isa) {
    // vlen = x64::cpu_isa_traits<isa>::vlen;
    // dataTypeSize = jcp.inDataPrc.size();
    // gridTypeSize = jcp.gridPrc.size();
    // dataElPerVec = vlen / dataTypeSize;
    // gridElPerVec = vlen / gridTypeSize;
    // if (dataTypeSize == 2)
    //     dataTypeShift = 1;
    // else if (dataTypeSize == 4)
    //     dataTypeShift = 2;
}

template <x64::cpu_isa_t isa>
void RandomUniform<isa>::generate() {
    this->preamble();
    registersPool = RegistersPool::create(isa, {rax, rcx, rsp, rdi, k0});

    r64_dst = getReg64();
    r64_work_amount = getReg64();

    mov(r64_work_amount, ptr[regParams + GET_OFF(work_amount)]);
    mov(r64_dst,  ptr[regParams + GET_OFF(dst_ptr)]);

    initVectors();
    process();

    registersPool.reset();
    this->postamble();
}

template <>
void RandomUniform<x64::avx512_core>::initVectors() {
    const auto r64_aux = getReg64();
    const auto r32_aux = Xbyak::Reg32(r64_aux.getIdx());
    const auto r16_aux = Xbyak::Reg16(r64_aux.getIdx());

    v_max_mul_n_64 = getVmm();
    v_max_mul_c_64 = getVmm();
    v_add_low_k    = getVmm();
    v_add_up_k     = getVmm();
    v_convert_0    = getVmm();
    v_convert_1    = getVmm();
    // v_one          = getVmm();
    v_n_inc        = getVmm();
    v_max_min      = getVmm();
    v_min          = getVmm();

    v_key_64       = getVmm();
    v_counter_64   = getVmm();
    v_n_64         = getVmm();
    // v_sep_perm     = getVmm();
    // v_sep_perm_1   = getVmm();
    // v_sep_perm_2   = getVmm();
    v_res_perm     = getVmm();
    // v_res_perm_1   = getVmm();

    // Initialize constants.
    mov(r64_aux, 0xd2511f53);
    vpbroadcastq(v_max_mul_n_64, r64_aux);

    mov(r64_aux, 0xcd9e8d57);
    vpbroadcastq(v_max_mul_c_64, r64_aux);

    mov(r32_aux, 0x9e3779b9);
    vpbroadcastd(v_add_low_k, r32_aux);

    mov(r32_aux, 0xbb67ae85);
    vpbroadcastd(v_add_up_k, r32_aux);

    mov(r64_aux, 0x00000008);
    vpbroadcastq(v_n_inc, r64_aux);

    if (m_jcp.out_data_type == element::f32) {
        mov(r32_aux, 0x3f800000);
        vpbroadcastd(v_convert_0, r32_aux);

        mov(r32_aux, 0x007fffff);
        vpbroadcastd(v_convert_1, r32_aux);

        mov(r64_aux, ptr[regParams + GET_OFF(max_ptr)]);
        vpbroadcastd(v_max_min, ptr[r64_aux]);

        mov(r64_aux, ptr[regParams + GET_OFF(min_ptr)]);
        vpbroadcastd(v_min, ptr[r64_aux]);

        uni_vsubps(v_max_min, v_max_min, v_min);
    } else if (m_jcp.out_data_type == element::f16) {
        mov(r16_aux, 0x3c00);
        vpbroadcastw(v_convert_0, r16_aux);

        mov(r16_aux, 0x03ff);
        vpbroadcastw(v_convert_1, r16_aux);
    } else if (m_jcp.out_data_type == element::bf16) {
        mov(r16_aux, 0x3f80);
        vpbroadcastw(v_convert_0, r16_aux);

        mov(r16_aux, 0x007f);
        vpbroadcastw(v_convert_1, r16_aux);
    } else if (m_jcp.out_data_type == element::i32) {
        const auto ymm_max_min = Xbyak::Ymm(v_max_min.getIdx());

        mov(r64_aux, ptr[regParams + GET_OFF(max_ptr)]);
        vpbroadcastd(v_max_min, ptr[r64_aux]);

        mov(r64_aux, ptr[regParams + GET_OFF(min_ptr)]);
        vpbroadcastd(v_min, ptr[r64_aux]);

        uni_vpsubd(v_max_min, v_max_min, v_min);
        vcvtdq2pd(v_max_min, ymm_max_min);
    } else if (m_jcp.out_data_type == element::i64) {
    }

    // Initialize inputs.
    mov(r64_aux, ptr[regParams + GET_OFF(key_ptr)]);
    vpbroadcastq(v_key_64, ptr[r64_aux]);

    mov(r64_aux, ptr[regParams + GET_OFF(counter_ptr)]);
    vpbroadcastq(v_counter_64, ptr[r64_aux]);

    mov(r64_aux, ptr[regParams + GET_OFF(n_ptr)]);
    vpbroadcastq(v_n_64, ptr[r64_aux]);
    if (m_jcp.out_data_type.size() <= 4) {
        static const uint64_t n_inc_arr[8]  = { 0, 1, 2, 3, 4, 5, 6, 7 };
        mov(r64_aux, reinterpret_cast<uintptr_t>(n_inc_arr));
    } else {
        static const uint64_t n_inc_arr[8]  = { 0, 1, 2, 3, 4, 5, 6, 7 }; // TODO: i64
        mov(r64_aux, reinterpret_cast<uintptr_t>(n_inc_arr));
    }
    vpaddq(v_n_64, v_n_64, ptr[r64_aux]);

    // Initialize auxiliary vectors.
    // static const uint32_t sep_perm_mask[16]  = { 0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15 };
    // mov(r64_aux, reinterpret_cast<uintptr_t>(sep_perm_mask));
    // uni_vmovups(v_sep_perm, ptr[r64_aux]);

    // static const uint32_t sep_perm_mask_1[16]  = { 1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14 };
    // mov(r64_aux, reinterpret_cast<uintptr_t>(sep_perm_mask_1));
    // uni_vmovups(v_sep_perm_1, ptr[r64_aux]);

    static const uint32_t res_perm_mask[16]  = { 0b00000000, 0b00010000, 0b00001000, 0b00011000, 0b00000010, 0b00010010, 0b00001010, 0b00011010,
                                                 0b00000100, 0b00010100, 0b00001100, 0b00011100, 0b00000110, 0b00010110, 0b00001110, 0b00011110 };
    // static const uint32_t res_perm_mask[16]  = { 0b00000000, 0b00000010, 0b00010000, 0b00010010, 0b00000001, 0b00000011, 0b00010001, 0b00010011,
    //                                              0b00000100, 0b00000110, 0b00010100, 0b00010110, 0b00000101, 0b00000111, 0b00010101, 0b00010111 };
    // static const uint32_t res_perm_mask[16]  = { 0b00000000, 0b00000001, 0b00010000, 0b00010001, 0b00000010, 0b00000011, 0b00010010, 0b00010011,
    //                                              0b00000100, 0b00000101, 0b00010100, 0b00010101, 0b00000110, 0b00000111, 0b00010110, 0b00010111 };
    mov(r64_aux, reinterpret_cast<uintptr_t>(res_perm_mask));
    uni_vmovups(v_res_perm, ptr[r64_aux]);

    // static const uint32_t res_perm_mask_1[16]  = { 0b00001000, 0b00001001, 0b00011000, 0b00011001, 0b00001010, 0b00001011, 0b00011010, 0b00011011,
    //                                                0b00001100, 0b00001101, 0b00011100, 0b00011101, 0b00001110, 0b00001111, 0b00011110, 0b00011111 };
    // static const uint32_t res_perm_mask_1[16]  = { 0b00001000, 0b00001010, 0b00011000, 0b00011010, 0b00001001, 0b00001011, 0b00011001, 0b00011011,
    //                                                0b00001100, 0b00001110, 0b00011100, 0b00011110, 0b00001101, 0b00001111, 0b00011101, 0b00011111 };
    // mov(r64_aux, reinterpret_cast<uintptr_t>(res_perm_mask_1));
    // uni_vmovups(v_res_perm_1, ptr[r64_aux]);

    // static const uint32_t sep_perm_mask_2[16]  = { 0b00000001, 0b00010000, 0b00000011, 0b00010010, 0b00000101, 0b00010100, 0b00000111, 0b00010110,
    //                                                0b00001001, 0b00011000, 0b00001011, 0b00011010, 0b00001101, 0b00011100, 0b00001111, 0b00011110 };
    // mov(r64_aux, reinterpret_cast<uintptr_t>(sep_perm_mask_2));
    // uni_vmovups(v_sep_perm_2, ptr[r64_aux]);
}

template <x64::cpu_isa_t isa> // Works for AVX2, SSE41
void RandomUniform<isa>::initVectors() {
    const auto r64_aux = getReg64();
    const auto r32_aux = Xbyak::Reg32(r64_aux.getIdx());
    const auto r16_aux = Xbyak::Reg16(r64_aux.getIdx());

    v_max_mul_n_64 = getVmm();
    v_max_mul_c_64 = getVmm();
    v_add_low_k    = getVmm();
    v_add_up_k     = getVmm();
    v_max_min      = getVmm();
    v_key_64       = getVmm();
    v_counter_64   = getVmm();
    v_n_64         = getVmm();

    r64_n_inc      = getReg64();
    r64_convert_0  = getReg64();
    r64_convert_1  = getReg64();
    r64_min        = getReg64();

#define INIT_ARR(A, V, R, T)                                                                \
    static const T A[8] = { V, V, V, V, V, V, V, V };                                       \
    if (isa == x64::avx2) {                                                                 \
        mov(R, reinterpret_cast<uintptr_t>(A));                                             \
    } else {                                                                                \
        static const T* A##_aligned = A + (reinterpret_cast<int64_t>(A) % 16) / sizeof(T);  \
        mov(R, reinterpret_cast<uintptr_t>(A##_aligned));                                   \
    }

    // Initialize constants.
    static const uint64_t max_mul_n_64 = 0xd2511f53;
    mov(r64_aux, reinterpret_cast<uintptr_t>(&max_mul_n_64));
    vpbroadcastq(v_max_mul_n_64, ptr[r64_aux]);

    static const uint64_t max_mul_c_64 = 0xcd9e8d57;
    mov(r64_aux, reinterpret_cast<uintptr_t>(&max_mul_c_64));
    vpbroadcastq(v_max_mul_c_64, ptr[r64_aux]);

    static const uint32_t add_low_k = 0x9e3779b9;
    mov(r64_aux, reinterpret_cast<uintptr_t>(&add_low_k));
    vpbroadcastd(v_add_low_k, ptr[r64_aux]);

    static const uint32_t add_up_k = 0xbb67ae85;
    mov(r64_aux, reinterpret_cast<uintptr_t>(&add_up_k));
    vpbroadcastd(v_add_up_k, ptr[r64_aux]);

    static const uint64_t n_inc_step[4] = { 4, 4, 4, 4 };
    if (isa == x64::avx2) {
        mov(r64_n_inc, reinterpret_cast<uintptr_t>(n_inc_step));
    } else {
        static const uint64_t* n_inc_step_align = n_inc_step + (reinterpret_cast<int64_t>(n_inc_step) % 16) / sizeof(uint64_t);
        mov(r64_n_inc, reinterpret_cast<uintptr_t>(n_inc_step_align));
    }

    if (m_jcp.out_data_type == element::f32) {
        static const uint32_t convert_0[8] = { 0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000 };
        if (isa == x64::avx2) {
            mov(r64_convert_0, reinterpret_cast<uintptr_t>(convert_0));
        } else {
            static const uint32_t* convert_0_align = convert_0 + (reinterpret_cast<int64_t>(convert_0) % 16) / sizeof(uint32_t);
            mov(r64_convert_0, reinterpret_cast<uintptr_t>(convert_0_align));
        }

        static const uint32_t convert_1[8] = { 0x007fffff, 0x007fffff, 0x007fffff, 0x007fffff, 0x007fffff, 0x007fffff, 0x007fffff, 0x007fffff };
        if (isa == x64::avx2) {
            mov(r64_convert_1, reinterpret_cast<uintptr_t>(convert_1));
        } else {
            static const uint32_t* convert_1_align = convert_1 + (reinterpret_cast<int64_t>(convert_1) % 16) / sizeof(uint32_t);
            mov(r64_convert_1, reinterpret_cast<uintptr_t>(convert_1_align));
        }

        mov(r64_aux, ptr[regParams + GET_OFF(max_ptr)]);
        vpbroadcastd(v_max_min, ptr[r64_aux]);

        auto v_aux = getVmm();
        mov(r64_aux, ptr[regParams + GET_OFF(min_ptr)]);
        vpbroadcastd(v_aux, ptr[r64_aux]);
        static uint32_t min_arr[8];
        mov(r64_min, reinterpret_cast<uintptr_t>(min_arr));
        uni_vmovups(ptr[r64_min], v_aux);

        uni_vsubps(v_max_min, v_max_min, v_aux);
    } else if (m_jcp.out_data_type == element::f16) {
        mov(r16_aux, 0x00003c00);
        vpbroadcastw(v_convert_0, r16_aux);
        mov(r16_aux, 0x000003ff);
        vpbroadcastw(v_convert_1, r16_aux);
    } else if (m_jcp.out_data_type == element::bf16) {
        mov(r16_aux, 0x00003f80);
        vpbroadcastw(v_convert_0, r16_aux);
        mov(r16_aux, 0x0000007f);
        vpbroadcastw(v_convert_1, r16_aux);
    } else if (m_jcp.out_data_type == element::i32) {
        const auto ymm_max_min = Xbyak::Ymm(v_max_min.getIdx());

        mov(r64_aux, ptr[regParams + GET_OFF(max_ptr)]);
        vpbroadcastd(v_max_min, ptr[r64_aux]);

        mov(r64_aux, ptr[regParams + GET_OFF(min_ptr)]);
        vpbroadcastd(v_min, ptr[r64_aux]);

        uni_vpsubd(v_max_min, v_max_min, v_min);
        vcvtdq2pd(v_max_min, ymm_max_min);
    } else if (m_jcp.out_data_type == element::i64) {
    }

    // Initialize inputs.
    mov(r64_aux, ptr[regParams + GET_OFF(key_ptr)]);
    vpbroadcastq(v_key_64, ptr[r64_aux]);

    mov(r64_aux, ptr[regParams + GET_OFF(counter_ptr)]);
    vpbroadcastq(v_counter_64, ptr[r64_aux]);

    mov(r64_aux, ptr[regParams + GET_OFF(n_ptr)]);
    vpbroadcastq(v_n_64, ptr[r64_aux]);
    if (m_jcp.out_data_type.size() <= 4) {
        static const uint64_t n_inc_arr[4]  = { 0, 1, 2, 3 };
        mov(r64_aux, reinterpret_cast<uintptr_t>(n_inc_arr));
    } else {
        static const uint64_t n_inc_arr[4]  = { 0, 1, 2, 3 }; // TODO: i64
        mov(r64_aux, reinterpret_cast<uintptr_t>(n_inc_arr));
    }
    vpaddq(v_n_64, v_n_64, ptr[r64_aux]);
}

template <x64::cpu_isa_t isa>
void RandomUniform<isa>::process() {
    auto v_dst_0 = getVmm();
    auto v_dst_1 = getVmm();
    std::vector<Vmm> v_res{ v_dst_0, v_dst_1 };

    auto step = vlen;
    if (one_of(m_jcp.out_data_type.size(), 2, 4)) {
        step = vlen * 2 / sizeof(uint32_t);
    } else if (m_jcp.out_data_type.size() == 8) {
        step = vlen / sizeof(uint32_t);
    }

    Xbyak::Label l_loop, l_tail;
    L(l_loop); {
        cmp(r64_work_amount, step);
        jl(l_tail, T_NEAR);

        runPhilox(v_res, v_key_64, v_counter_64, v_n_64);
        convert(v_res, v_res);

        uni_vmovups(ptr[r64_dst], v_dst_0);
        add(r64_dst, vlen);
        if (one_of(m_jcp.out_data_type.size(), 4, 8)) {
            uni_vmovups(ptr[r64_dst], v_dst_1);
            add(r64_dst, vlen);
        }

        if (isa == x64::avx512_core) {
            uni_vpaddd(v_n_64, v_n_64, v_n_inc);
        } else {
            uni_vpaddd(v_n_64, v_n_64, ptr[r64_n_inc]);
        }

        sub(r64_work_amount, step);
        jmp(l_loop, T_NEAR);
    }

    L(l_tail);
    tail(v_res);
}

template <x64::cpu_isa_t isa>
void RandomUniform<isa>::calculateRound(const Vmm& vmm_k_0, const Vmm& vmm_k_1, const Vmm& vmm_c_0, const Vmm& vmm_c_1,
                                        const Vmm& vmm_n_0, const Vmm& vmm_n_1, const Vmm& vmm_aux_0, const Vmm& vmm_aux_1) {
    uni_vpmuludq(vmm_aux_0, vmm_n_0, v_max_mul_n_64); // {p0,p1,p0,p1} = {n0,_,n0,_} * {m0,_,m0,_}
    uni_vpmuludq(vmm_aux_1, vmm_c_0, v_max_mul_c_64); // {r0,r1,r0,r1} = {c0,_,c0,_} * {m0,_,m0,_}

    uni_vpshufd(vmm_c_0, vmm_aux_0, 0b10110001);      // {p1,p0,p1,p0} = shuf {p0,p1,p0,p1}
    uni_vxorps(vmm_c_0, vmm_c_0, vmm_c_1);            // {c0,_,c0,_} = {p1,_,p1,_} ^ {c1,_,c1,_}
    uni_vxorps(vmm_c_0, vmm_c_0, vmm_k_1);            // {c0,_,c0,_} = {c0,_,c0,_} ^ {k1,_,k1,_}

    uni_vpshufd(vmm_n_0, vmm_aux_1, 0b10110001);      // {r1,r0,r1,r0} = shuf {r0,r1,r0,r1}
    uni_vxorps(vmm_n_0, vmm_n_0, vmm_n_1);            // {n0,_,n0,_} = {r1,_,r1,_} ^ {n1,_,n1,_}
    uni_vxorps(vmm_n_0, vmm_n_0, vmm_k_0);            // {n0,_,n0,_} = {n0,_,n0,_} ^ {k0,_,k0,_}
}

template <x64::cpu_isa_t isa>
void RandomUniform<isa>::runPhilox(const std::vector<Vmm>& vmm_dst, const Vmm& vmm_key, const Vmm& vmm_counter, const Vmm& vmm_n) {
    auto vmm_k_0 = getVmm();
    auto vmm_k_1 = getVmm();
    auto vmm_n_0 = getVmm();
    auto vmm_n_1 = vmm_dst[0];
    auto vmm_c_0 = getVmm();
    auto vmm_c_1 = getVmm();
    auto vmm_aux_0 = getVmm();
    auto vmm_aux_1 = vmm_dst[1];

    uni_vmovups(vmm_k_0, vmm_key);                        // {k0,k1,k0,k1} -> {k0,_,k0,_}
    vpshufd(vmm_k_1, vmm_key, 0b10110001);                // {k0,k1,k0,k1} -> {k1,_,k1,_}

    uni_vpmuludq(vmm_aux_0, vmm_n, v_max_mul_n_64);       // {p0,p1,p0,p1} = {n0,_,n0,_} * {m0,_,m0,_}
    uni_vpmuludq(vmm_aux_1, vmm_counter, v_max_mul_c_64); // {r0,r1,r0,r1} = {c0,_,c0,_} * {m0,_,m0,_}

    uni_vxorps(vmm_c_0, vmm_aux_0, vmm_counter);          // {_,c0,_,c0} = {_,p1,_,p1} ^ {_,c1,_,c1}
    uni_vxorps(vmm_c_0, vmm_c_0, vmm_key);                // {_,c0,_,c0} = {_,c0,_,c0} ^ {_,k1,_,k1}
    uni_vpshufd(vmm_c_0, vmm_c_0, 0b10110001);            // {_,c0,_,c0} -> {c0,_,c0,_}

    uni_vxorps(vmm_n_0, vmm_aux_1, vmm_n);                // {_,n0,_,n0} = {_,r1,_,r1} ^ {_,n1,_,n1}
    uni_vpshufd(vmm_n_0, vmm_n_0, 0b10110001);            // {_,n0,_,n0} -> {n0,_,n0,_}
    uni_vxorps(vmm_n_0, vmm_n_0, vmm_key);                // {n0,_,n0,_} = {n0,_,n0,_} ^ {k0,_,k0,_}

    for (size_t i = 0lu; i < ROUNDS_NUMBER - 1; i++) {
        raiseKey(vmm_k_0, vmm_k_1);

        std::swap(vmm_c_1, vmm_aux_0);
        std::swap(vmm_n_1, vmm_aux_1);
        calculateRound(vmm_k_0, vmm_k_1, vmm_c_0, vmm_c_1, vmm_n_0, vmm_n_1, vmm_aux_0, vmm_aux_1);
    }
    std::swap(vmm_c_1, vmm_aux_0);
    std::swap(vmm_n_1, vmm_aux_1);

    if (isa == x64::avx512_core) {
        vpermt2d(vmm_n_0, v_res_perm, vmm_n_1);
        vpermt2d(vmm_c_0, v_res_perm, vmm_c_1);
        vshufpd(vmm_dst[0], vmm_n_0, vmm_c_0, 0b00000000);
        vshufpd(vmm_dst[1], vmm_n_0, vmm_c_0, 0b11111111);
    } else if (isa == x64::avx2) {
        auto ymm_dst_0 = Xbyak::Ymm(vmm_dst[0].getIdx());
        auto ymm_dst_1 = Xbyak::Ymm(vmm_dst[1].getIdx());
        auto ymm_n_0 = Xbyak::Ymm(vmm_n_0.getIdx());
        auto ymm_c_0 = Xbyak::Ymm(vmm_c_0.getIdx());

        uni_vshufps(vmm_n_0, vmm_n_0, vmm_n_1, 0b10001000);
        uni_vshufps(vmm_c_0, vmm_c_0, vmm_c_1, 0b10001000);
        uni_vshufps(ymm_dst_1, vmm_n_0, vmm_c_0, 0b10001000);
        uni_vshufps(vmm_c_0, vmm_n_0, vmm_c_0, 0b11011101);
        vperm2f128(ymm_dst_0, ymm_dst_1, ymm_c_0, 0b00100000);
        vperm2f128(ymm_dst_1, ymm_dst_1, ymm_c_0, 0b00110001);
    } else {
        uni_vshufps(vmm_n_0, vmm_n_0, vmm_n_1, 0b10001000);
        uni_vshufps(vmm_c_0, vmm_c_0, vmm_c_1, 0b10001000);
        uni_vshufps(vmm_dst[0], vmm_n_0, vmm_n_0, 0b10001000);
        uni_vshufps(vmm_dst[1], vmm_c_0, vmm_c_0, 0b10001000);
    }
}

// template <x64::cpu_isa_t isa>
// void RandomUniform<isa>::runPhilox(const std::vector<Vmm>& vmm_dst, const Vmm& vmm_key, const Vmm& vmm_counter, const Vmm& vmm_n) {
//     // Define sparse vectors.
//     auto vmm_k_0 = getVmm();
//     auto vmm_k_1 = getVmm();
//     auto vmm_c_0 = getVmm();
//     auto vmm_c_1 = getVmm();
//     auto vmm_n_0 = getVmm();
//     auto vmm_n_1 = getVmm();
//     auto vmm_aux_0 = getVmm();
//     auto vmm_aux_1 = getVmm();

//     uni_vmovups(vmm_k_1, vmm_key);               // {k0,k1,k0,k1} -> {_,k1,_,k1}
//     // vpermps(vmm_k_0, v_sep_perm_1, vmm_key);     // {k0,k1,k0,k1} -> {_,k0,_,k0}
//     uni_vmovups(vmm_c_0, vmm_counter);           // {c0,c1,c0,c1} -> {c0,_,c0,_}
//     uni_vmovups(vmm_n_0, vmm_n);                 // {n0,n1,n0,n1} -> {n0,_,n0,_}

//     for (size_t i = 0lu; i < ROUNDS_NUMBER; i++) {
//         calculateRound(vmm_k_0, vmm_k_1, vmm_c_0, vmm_c_1, vmm_n_0, vmm_n_1, vmm_aux_0, vmm_aux_1);
//         if (i < ROUNDS_NUMBER - 1) {
//             raiseKey(vmm_k_0, vmm_k_1);
//         }
//     }

//     // uni_vmovups(vmm_dst[0], vmm_n_0);            // {n0,n1,n0,n1}
//     // uni_vmovups(vmm_dst[1], vmm_n_0);            // {n0,n1,n0,n1}
//     // vpermt2d(vmm_dst[0], v_res_perm, vmm_c_0);   // {n0,n1,c0,c1,n0,n1,c0,c1} = perm( {n0,n1,n0,n1}, {c0,c1,c0,c1} )
//     // vpermt2d(vmm_dst[1], v_res_perm_1, vmm_c_0); // {n0,n1,c0,c1,n0,n1,c0,c1} = perm( {n0,n1,n0,n1}, {c0,c1,c0,c1} )
// }

template <x64::cpu_isa_t isa>
void RandomUniform<isa>::raiseKey(const Vmm& vmm_k_0, const Vmm& vmm_k_1) {
    uni_vpaddd(vmm_k_0, vmm_k_0, v_add_low_k); // {_,k0,_,k0} + {_,l0,_,l0}
    uni_vpaddd(vmm_k_1, vmm_k_1, v_add_up_k);  // {_,k1,_,k1} + {_,u0,_,u0}
}

template <>
void RandomUniform<x64::avx512_core>::convert(const std::vector<Vmm>& v_dst, const std::vector<Vmm>& v_src) {
    if (m_jcp.out_data_type.size() == 4) {
        for (const auto& vmm_dst : v_dst) { // TODO: change to v_src
            if (m_jcp.out_data_type == element::f32) {
                uni_vandps(vmm_dst, vmm_dst, v_convert_1);
                uni_vorps(vmm_dst, vmm_dst, v_convert_0);
                uni_vsubps(vmm_dst, vmm_dst, v_convert_0);
                vfmadd132ps(vmm_dst, v_min, v_max_min);
            } else if (m_jcp.out_data_type == element::i32) {
                // x % (max - min) + min
                const auto v_aux_0 = getVmm();
                const auto v_aux_1 = getVmm();
                const auto ymm_dst = Xbyak::Ymm(vmm_dst.getIdx());
                const auto ymm_aux_1 = Xbyak::Ymm(v_aux_1.getIdx());

                // Divide in the f64 due to the f32 loses accuracy here.
                vcvtudq2pd(v_aux_0, ymm_dst);
                vdivpd(v_aux_1, v_aux_0, v_max_min);
                vrndscalepd(v_aux_1, v_aux_1, 3);
                vfnmadd132pd(v_aux_1, v_aux_0, v_max_min);

                vextractf64x4(ymm_dst, vmm_dst, 1);
                vcvtudq2pd(v_aux_0, ymm_dst);
                vcvtpd2dq(ymm_dst, v_aux_1);
                vdivpd(v_aux_1, v_aux_0, v_max_min);
                vrndscalepd(v_aux_1, v_aux_1, 3);
                vfnmadd132pd(v_aux_1, v_aux_0, v_max_min);
                vcvtpd2dq(ymm_aux_1, v_aux_1);
                vshuff64x2(vmm_dst, vmm_dst, v_aux_1, 0b01000100);

                uni_vpaddd(vmm_dst, vmm_dst, v_min);
            } else {
                OPENVINO_THROW("RandomUniform kernel does not support precision ", m_jcp.out_data_type, " for ", x64::get_isa_info());
            }
        }
    } else if (m_jcp.out_data_type.size() == 2) {
        if (m_jcp.out_data_type == element::f16 && x64::mayiuse(x64::avx512_core_fp16)) {
            auto ymm_dst_0 = Xbyak::Ymm(v_dst[0].getIdx());
            auto ymm_dst_1 = Xbyak::Ymm(v_dst[1].getIdx());

            vpmovusdw(ymm_dst_0, v_dst[0]);
            vpmovusdw(ymm_dst_1, v_dst[1]);
            vshuff64x2(v_dst[0], v_dst[0], v_dst[1], 0x01000100);

            uni_vandps(v_dst[0], v_dst[0], v_convert_1);
            uni_vorps(v_dst[0], v_dst[0], v_convert_0);
            vsubph(v_dst[0], v_dst[0], v_convert_0);
            vfmadd132ph(v_dst[0], v_min, v_max_min);
        } else if (m_jcp.out_data_type == element::bf16 && x64::mayiuse(x64::avx512_core_bf16)) {
            auto ymm_convert_0 = Xbyak::Ymm(v_convert_0.getIdx());
            auto ymm_convert_1 = Xbyak::Ymm(v_convert_1.getIdx());

            for (const auto& vmm_dst : v_dst) {
                auto ymm_dst = Xbyak::Ymm(vmm_dst.getIdx());

                vpmovusdw(ymm_dst, vmm_dst); // TODO: truncate instead?
                uni_vandps(ymm_dst, ymm_dst, ymm_convert_1);
                uni_vorps(ymm_dst, ymm_dst, ymm_convert_0);

                vpmovzxwd(vmm_dst, ymm_dst);
                uni_vpslld(vmm_dst, vmm_dst, 16);

                uni_vsubps(vmm_dst, vmm_dst, v_convert_0);
                vfmadd132ps(vmm_dst, v_min, v_max_min);
            }

            vcvtne2ps2bf16(v_dst[0], v_dst[0], v_dst[1]);
        } else {
            OPENVINO_THROW("RandomUniform kernel does not support precision ", m_jcp.out_data_type, " for ", x64::get_isa_info());
        }
    } else if (m_jcp.out_data_type.size() == 8) {
        if (m_jcp.out_data_type == element::i64) {
            // TODO: in scope of i64 enabling.
        }
        OPENVINO_THROW("RandomUniform kernel does not support precision ", m_jcp.out_data_type, " for ", x64::get_isa_info());
    } else {
        OPENVINO_THROW("RandomUniform kernel does not support precision ", m_jcp.out_data_type, " for ", x64::get_isa_info());
    }
}

template <x64::cpu_isa_t isa>
void RandomUniform<isa>::convert(const std::vector<Vmm>& v_dst, const std::vector<Vmm>& v_src) {
    if (m_jcp.out_data_type.size() == 4) {
        for (const auto& vmm_dst : v_dst) {
            if (m_jcp.out_data_type == element::f32) {
                uni_vandps(vmm_dst, vmm_dst, ptr[r64_convert_1]);
                uni_vorps(vmm_dst, vmm_dst, ptr[r64_convert_0]);
                uni_vsubps(vmm_dst, vmm_dst, ptr[r64_convert_0]);
                vfmadd213ps(vmm_dst, v_max_min, ptr[r64_min]);
            } else if (m_jcp.out_data_type == element::i32) {
                // x % (max - min) + min
                const auto v_aux_0 = getVmm();
                const auto v_aux_1 = getVmm();
                const auto ymm_dst = Xbyak::Ymm(vmm_dst.getIdx());
                const auto ymm_aux_1 = Xbyak::Ymm(v_aux_1.getIdx());

                // Divide in the f64 due to the f32 loses accuracy here.
                // vcvtudq2pd(v_aux_0, ymm_dst);
                // vdivpd(v_aux_1, v_aux_0, v_max_min);
                // vrndscalepd(v_aux_1, v_aux_1, 3);
                // vfnmadd132pd(v_aux_1, v_aux_0, v_max_min);

                // vextractf64x4(ymm_dst, vmm_dst, 1);
                // vcvtudq2pd(v_aux_0, ymm_dst);
                // vcvtpd2dq(ymm_dst, v_aux_1);
                // vdivpd(v_aux_1, v_aux_0, v_max_min);
                // vrndscalepd(v_aux_1, v_aux_1, 3);
                // vfnmadd132pd(v_aux_1, v_aux_0, v_max_min);
                // vcvtpd2dq(ymm_aux_1, v_aux_1);
                // vshuff64x2(vmm_dst, vmm_dst, v_aux_1, 0b01000100);

                uni_vpaddd(vmm_dst, vmm_dst, v_min);
            } else {
                OPENVINO_THROW("RandomUniform kernel does not support precision ", m_jcp.out_data_type, " for ", x64::get_isa_info());
            }
        }
    } else if (m_jcp.out_data_type.size() == 8) {
        if (m_jcp.out_data_type == element::i64) {
            // TODO: in scope of i64 enabling.
        }
        OPENVINO_THROW("RandomUniform kernel does not support precision ", m_jcp.out_data_type, " for ", x64::get_isa_info());
    } else {
        OPENVINO_THROW("RandomUniform kernel does not support precision ", m_jcp.out_data_type, " for ", x64::get_isa_info());
    }
}

template <>
void RandomUniform<x64::avx512_core>::tail(const std::vector<Vmm>& vmm_dst) {
    Xbyak::Label l_0, l_end;
    const auto k_rest_mask = getMask();
    const auto step = vlen / sizeof(uint32_t);

    cmp(r64_work_amount, 0);
    jle(l_end, T_NEAR);

    runPhilox(vmm_dst, v_key_64, v_counter_64, v_n_64);
    convert(vmm_dst, vmm_dst);

    cmp(r64_work_amount, step);
    jl(l_0, T_NEAR);

    uni_vmovups(ptr[r64_dst], vmm_dst[0]);
    add(r64_dst, vlen);
    sub(r64_work_amount, step);
    fillRestWorkMask(k_rest_mask, r64_work_amount);
    uni_vmovups(ptr[r64_dst] | k_rest_mask, vmm_dst[1]);
    jmp(l_end, T_NEAR);

    L(l_0);
    fillRestWorkMask(k_rest_mask, r64_work_amount);
    uni_vmovups(ptr[r64_dst] | k_rest_mask, vmm_dst[0]);

    L(l_end);
}

template <x64::cpu_isa_t isa>
void RandomUniform<isa>::tail(const std::vector<Vmm>& vmm_dst) {
    Xbyak::Label l_0, l_end;
    const auto step = vlen / sizeof(uint32_t);

    cmp(r64_work_amount, 0);
    jle(l_end, T_NEAR);

    runPhilox(vmm_dst, v_key_64, v_counter_64, v_n_64);
    convert(vmm_dst, vmm_dst);
    const auto v_rest_mask = getVmm();

    cmp(r64_work_amount, step);
    jl(l_0, T_NEAR);

    uni_vmovups(ptr[r64_dst], vmm_dst[0]);
    add(r64_dst, vlen);
    sub(r64_work_amount, step);
    fillRestWorkMask(v_rest_mask, r64_work_amount, m_jcp.out_data_type.size());
    vmaskmovps(ptr[r64_dst], v_rest_mask, vmm_dst[1]);
    jmp(l_end, T_NEAR);

    L(l_0);
    fillRestWorkMask(v_rest_mask, r64_work_amount, m_jcp.out_data_type.size());
    vmaskmovps(ptr[r64_dst],  v_rest_mask, vmm_dst[0]);

    L(l_end);
}

// template <x64::cpu_isa_t isa> // Works for AVX512, AVX2, AVX, SSE41
// void RandomUniform<isa>::interpolation(const Vmm& vWCoord, const Vmm& vHCoord, bool tail) {
//     if (jcp.interpolationMode == GridSampleInterpolationMode::BILINEAR) {
//         bilinearInterpolation(vWCoord, vHCoord, tail);
//     } else if (jcp.interpolationMode == GridSampleInterpolationMode::BICUBIC) {
//         bicubicInterpolation(vWCoord, vHCoord, tail);
//     } else if (jcp.interpolationMode == GridSampleInterpolationMode::NEAREST) {
//         nearestInterpolation(vWCoord, vHCoord, tail);
//     }
// }

// template <x64::cpu_isa_t isa> // Works for AVX512, AVX2, AVX, SSE41
// void RandomUniform<isa>::tail() {
//     Xbyak::Label lEnd;
//     cmp(r64_work_amount, 0);
//     jle(lEnd, T_NEAR);

//     auto vHCoord = getVmm();
//     auto vWCoord = getVmm();

//     getTailCoordinates(vHCoord, vWCoord);
//     denormalizeRawCoordinates(vWCoord, vHCoord);
//     interpolation(vWCoord, vHCoord, true);

//     if (dataTypeSize > 1)
//         sal(r64_work_amount, dataTypeShift); // Multiply by source data type size.
//     add(r64_dst, r64_work_amount);

//     L(lEnd);
// }

// template <>
// void RandomUniform<x64::avx512_core>::getCoordinates(const Vmm& vHCoord, const Vmm& vWCoord) {
//     vpermd(vWCoord, vGridPermMask, ptr[regGrid]);      // Permute to XXXX.XXXX.YYYY.YYYY
//     vshuff64x2(vHCoord, vWCoord, vHCoord, 0B11101110); // Extract Y component

//     add(regGrid, vlen);

//     auto vAux = getVmm();
//     vpermd(vAux, vGridPermMask, ptr[regGrid]);         // Permute to XXXX.XXXX.YYYY.YYYY
//     vshuff64x2(vWCoord, vWCoord, vAux, 0B01000100);    // Extract X component
//     vshuff64x2(vHCoord, vHCoord, vAux, 0B11100100);    // Extract Y component

//     add(regGrid, vlen);
// }

// template <>
// void RandomUniform<x64::avx2>::getCoordinates(const Vmm& vHCoord, const Vmm& vWCoord) {
//     auto vAux = getVmm();
//     Vmm vPermMask;
//     RegistersPool::Reg<Vmm> permMaskHolder;

//     if (vGridPermMask.isInitialized()) {
//         vPermMask = vGridPermMask;
//     } else {
//         static const unsigned gridPermMask[8] = {0, 2, 4, 6, 1, 3, 5, 7};
//         auto rAux = getReg64();
//         permMaskHolder = getVmm();
//         vPermMask = permMaskHolder;
//         mov(rAux, reinterpret_cast<uintptr_t>(gridPermMask));
//         uni_vmovups(vPermMask, ptr[rAux]);
//     }

//     vpermd(vWCoord, vPermMask, ptr[regGrid]);          // Permute to XXXX.YYYY
//     vperm2f128(vHCoord, vHCoord, vWCoord, 0B00000011); // Extract Y component

//     add(regGrid, vlen);

//     vpermd(vAux, vPermMask, ptr[regGrid]);             // Permute to XXXX.YYYY
//     vperm2f128(vWCoord, vWCoord, vAux, 0B00100000);    // Extract X component
//     vperm2f128(vHCoord, vHCoord, vAux, 0B00110000);    // Extract Y component

//     add(regGrid, vlen);
// }

// template <x64::cpu_isa_t isa> // Works for AVX, SSE41
// void RandomUniform<isa>::getCoordinates(const Vmm& vHCoord, const Vmm& vWCoord) {
//     auto vAux = getVmm();
//     Xbyak::Xmm xmmWCoord(vWCoord.getIdx());
//     Xbyak::Xmm xmmHCoord(vHCoord.getIdx());
//     Xbyak::Xmm xmmAux(vAux.getIdx());
//     const uint64_t xmmVlen = x64::cpu_isa_traits<x64::sse41>::vlen;

//     uni_vmovups(xmmWCoord, ptr[regGrid]);
//     uni_vpshufd(xmmWCoord, xmmWCoord, 0xD8);
//     shufpd(xmmHCoord, xmmWCoord, 0x2);

//     add(regGrid, xmmVlen);

//     uni_vmovups(xmmAux, ptr[regGrid]);
//     uni_vpshufd(xmmAux, xmmAux, 0xD8);
//     shufpd(xmmWCoord, xmmAux, 0x0);
//     shufpd(xmmHCoord, xmmAux, 0x3);

//     add(regGrid, xmmVlen);

//     if (isa == x64::avx) {
//         Xbyak::Ymm ymmWCoord(vWCoord.getIdx());
//         Xbyak::Ymm ymmHCoord(vHCoord.getIdx());

//         vperm2f128(ymmWCoord, ymmWCoord, ymmWCoord, 0x1);
//         vperm2f128(ymmHCoord, ymmHCoord, ymmHCoord, 0x1);

//         // Here is movups + pshufd instead of vpshufd for two reasons:
//         // 1. vpshufd zeroes the rest ov YMM.
//         // 2. pshufd does not work with not aligned address.
//         movups(xmmWCoord, ptr[regGrid]);
//         pshufd(xmmWCoord, xmmWCoord, 0xD8);
//         shufpd(xmmHCoord, xmmWCoord, 0x2);

//         add(regGrid, xmmVlen);

//         uni_vpshufd(xmmAux, ptr[regGrid], 0xD8);
//         shufpd(xmmWCoord, xmmAux, 0x0);
//         shufpd(xmmHCoord, xmmAux, 0x3);

//         add(regGrid, xmmVlen);

//         vperm2f128(ymmWCoord, ymmWCoord, ymmWCoord, 0x1);
//         vperm2f128(ymmHCoord, ymmHCoord, ymmHCoord, 0x1);
//     }
// }

// template <>
// void RandomUniform<x64::avx512_core>::getTailCoordinates(const Vmm& vHCoord, const Vmm& vWCoord) {
//     Xbyak::Label lEnd, lGridShift, lRest;

//     auto vAux = getVmm();
//     auto rAux = getReg64();

//     mov(rAux, r64_work_amount);
//     sal(rAux, 0x1); // Multiply by gridShape[3].
//     cmp(r64_work_amount, dataElPerVec / 2);
//     jl(lRest, T_NEAR);
//     {
//         vpermd(vWCoord, vGridPermMask, ptr[regGrid]);
//         vshuff64x2(vHCoord, vWCoord, vHCoord, 0B11101110); // Extract Y component

//         add(regGrid, vlen);
//         sub(rAux, dataElPerVec);
//         cmp(rAux, 0);
//         jle(lEnd, T_NEAR);

//         fillRestWorkMask(kTailMask, rAux);
//         uni_vmovups((Vmm)vAux | kTailMask, ptr[regGrid]);
//         vpermd(vAux, vGridPermMask, vAux);
//         Xbyak::Ymm ymmAux(vAux.getIdx());
//         vshuff64x2(vWCoord, vWCoord, vAux, 0B01000100);    // Extract X component
//         vshuff64x2(vHCoord, vHCoord, vAux, 0B11100100);    // Extract Y component

//         jmp(lGridShift, T_NEAR);
//     }
//     L(lRest);
//     {
//         fillRestWorkMask(kTailMask, rAux);
//         uni_vmovups(vWCoord | kTailMask, ptr[regGrid]);
//         vpermd(vWCoord, vGridPermMask, vWCoord);
//         vshuff64x2(vHCoord, vWCoord, vHCoord, 0B11101110); // Extract Y component
//     }

//     L(lGridShift);
//     if (dataTypeSize > 1)
//         sal(rAux, dataTypeShift); // Multiply by source data type size.
//     add(regGrid, rAux);

//     L(lEnd);

//     fillRestWorkMask(kTailMask, r64_work_amount);
// }

// template <>
// void RandomUniform<x64::avx2>::getTailCoordinates(const Vmm& vHCoord, const Vmm& vWCoord) {
//     Xbyak::Label lRest, lGridShift, lEnd;

//     auto rAux = getReg64();
//     Vmm vPermMask;
//     RegistersPool::Reg<Vmm> permMaskHolder;

//     if (vGridPermMask.isInitialized()) {
//         vPermMask = vGridPermMask;
//     } else {
//         static const unsigned gridPermMask[8] = {0, 2, 4, 6, 1, 3, 5, 7};
//         permMaskHolder = getVmm();
//         vPermMask = permMaskHolder;
//         mov(rAux, reinterpret_cast<uintptr_t>(gridPermMask));
//         uni_vmovups(vPermMask, ptr[rAux]);
//     }

//     mov(rAux, r64_work_amount);
//     sal(rAux, 0x1); // multiply by gridShape[3] == 2
//     cmp(r64_work_amount, dataElPerVec / 2);
//     jl(lRest, T_NEAR);
//     {
//         vpermd(vWCoord, vPermMask, ptr[regGrid]);          // Permute to XXXX.YYYY
//         vperm2f128(vHCoord, vHCoord, vWCoord, 0B00000011); // Extract Y component

//         add(regGrid, vlen);
//         sub(rAux, dataElPerVec);
//         cmp(rAux, 0);
//         jle(lEnd, T_NEAR);

//         auto vAux  = getVmm();
//         load(vAux, ptr[regGrid], rAux, dataTypeSize);
//         vpermd(vAux, vPermMask, vAux);
//         vperm2f128(vWCoord, vWCoord, vAux, 0B00100000); // Extract X component
//         vperm2f128(vHCoord, vHCoord, vAux, 0B00110000); // Extract Y component

//         jmp(lGridShift, T_NEAR);
//     }
//     L(lRest);
//     {
//         load(vWCoord, ptr[regGrid], rAux, dataTypeSize);
//         vpermd(vWCoord, vPermMask, vWCoord);               // Permute to XXXX.YYYY
//         vperm2f128(vHCoord, vHCoord, vWCoord, 0B00000011); // Extract Y component
//     }

//     L(lGridShift);
//     if (dataTypeSize > 1)
//         sal(rAux, dataTypeShift); // Multiply by source data type size.
//     add(regGrid, rAux);

//     L(lEnd);
// }

// template <>
// void RandomUniform<x64::avx>::getTailCoordinates(const Vmm& vHCoord, const Vmm& vWCoord) {
//     Xbyak::Label lLoop2End, lEnd;

//     Xbyak::Xmm xmmWCoord(vWCoord.getIdx());
//     Xbyak::Xmm xmmHCoord(vHCoord.getIdx());

//     auto rGridRest = getReg64();
//     mov(rGridRest, r64_work_amount);
//     sal(rGridRest, 0x1); // multiply by gridShape[3] == 2

//     for (size_t i = 0; i < dataElPerVec; i++) {
//         cmp(rGridRest, 0);
//         jle(lEnd, T_NEAR);

//         if (gridTypeSize == 4)
//             pinsrd(i % 2 == 0 ? xmmWCoord : xmmHCoord, ptr[regGrid], i / 2);
//         else if (gridTypeSize == 2)
//             pinsrw(i % 2 == 0 ? xmmWCoord : xmmHCoord, ptr[regGrid], i / 2);

//         add(regGrid, gridTypeSize);
//         dec(rGridRest);
//     }

//     cmp(rGridRest, 0);
//     jle(lEnd, T_NEAR);

//     vperm2f128(vWCoord, vWCoord, vWCoord, 0x1);
//     vperm2f128(vHCoord, vHCoord, vHCoord, 0x1);

//     for (size_t i = 0; i < dataElPerVec; i++) {
//         cmp(rGridRest, 0);
//         jle(lLoop2End, T_NEAR);

//         if (gridTypeSize == 4)
//             pinsrd(i % 2 == 0 ? xmmWCoord : xmmHCoord, ptr[regGrid], i / 2);
//         else if (gridTypeSize == 2)
//             pinsrw(i % 2 == 0 ? xmmWCoord : xmmHCoord, ptr[regGrid], i / 2);

//         add(regGrid, gridTypeSize);
//         dec(rGridRest);
//     }

//     L(lLoop2End);
//     vperm2f128(vWCoord, vWCoord, vWCoord, 0x1);
//     vperm2f128(vHCoord, vHCoord, vHCoord, 0x1);

//     L(lEnd);
// }

// template <>
// void RandomUniform<x64::sse41>::getTailCoordinates(const Vmm& vHCoord, const Vmm& vWCoord) {
//     Xbyak::Label lRest, lHShuf, lGridShift, lEnd;
//     auto rAux = getReg64();

//     mov(rAux, r64_work_amount);
//     sal(rAux, 0x1); // Multiply by gridShape[3] == 2
//     cmp(r64_work_amount, dataElPerVec / 2);
//     jl(lRest, T_NEAR);
//     {
//         // Here is movups + pshufd instead of pshufd due to
//         // pshufd does not work with not aligned address.
//         movups(vWCoord, ptr[regGrid]);
//         pshufd(vWCoord, vWCoord, 0B11011000);
//         shufpd(vHCoord, vWCoord, 0B00000010);

//         add(regGrid, vlen);
//         sub(rAux, dataElPerVec);
//         cmp(rAux, 0);
//         jle(lHShuf, T_NEAR);

//         auto vAux = getVmm();
//         load(vAux, ptr[regGrid], rAux, dataTypeSize);
//         pshufd(vAux, vAux, 0B11011000);
//         shufpd(vWCoord, vAux, 0x0);        // Extract X component
//         shufpd(vHCoord, vAux, 0B00000011); // Extract Y component

//         jmp(lGridShift, T_NEAR);
//         L(lHShuf);
//         shufpd(vHCoord, vHCoord, 0B00000001); // Extract Y component
//         jmp(lEnd, T_NEAR);
//     }
//     L(lRest);
//     {
//         load(vWCoord, ptr[regGrid], rAux, dataTypeSize);
//         pshufd(vWCoord, vWCoord, 0B11011000); // Extract X component
//         shufpd(vHCoord, vWCoord, 0B00000010); // Extract Y component
//         shufpd(vHCoord, vHCoord, 0B00000001);
//     }

//     L(lGridShift);
//     if (dataTypeSize > 1)
//         sal(rAux, dataTypeShift); // Multiply by source data type size.
//     add(regGrid, rAux);

//     L(lEnd);
// }

// template <x64::cpu_isa_t isa> // Works for AVX512, AVX2, AVX, SSE41
// void RandomUniform<isa>::denormalizeRawCoordinates(const Vmm& vWCoord, const Vmm& vHCoord) {
//     if (jcp.alignCorners) {
//         if (vWDenormCoefF.isInitialized()) {
//             uni_vfmadd132ps(vWCoord, vWDenormCoefF, vWDenormCoefF);
//         } else {
//             auto rAux = getReg64();
//             auto vAux = getVmm();
//             mov(rAux, ptr[regParams + GET_OFF(wDenormCoefF)]);
//             uni_vmovups(vAux, ptr[rAux]);
//             uni_vfmadd132ps(vWCoord, vAux, vAux);
//         }

//         if (vHDenormCoefF.isInitialized()) {
//             uni_vfmadd132ps(vHCoord, vHDenormCoefF, vHDenormCoefF);
//         } else {
//             auto rAux = getReg64();
//             auto vAux = getVmm();
//             mov(rAux, ptr[regParams + GET_OFF(hDenormCoefF)]);
//             uni_vmovups(vAux, ptr[rAux]);
//             uni_vfmadd132ps(vHCoord, vAux, vAux);
//         }
//     } else {
//         Vmm vHalfTmp;
//         RegistersPool::Reg<Vmm> halfHolder;
//         if (vHalfF.isInitialized()) {
//             vHalfTmp = vHalfF;
//         } else {
//             auto rAux = getReg64();
//             halfHolder = getVmm();
//             vHalfTmp = halfHolder;
//             static const float halfValues[x64::cpu_isa_traits<x64::avx512_core>::vlen / sizeof(float)] =
//                     { 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f };
//             mov(rAux, reinterpret_cast<uintptr_t>(halfValues));
//             uni_vmovups(vHalfTmp, ptr[rAux]);
//         }

//         if (vSrcWidthF.isInitialized()) {
//             uni_vfmadd132ps(vWCoord, vSrcWidthF, vSrcWidthF);
//         } else {
//             auto rAux = getReg64();
//             auto vAux = getVmm();
//             mov(rAux, ptr[regParams + GET_OFF(srcWidthF)]);
//             uni_vpbroadcastd(vAux, ptr[rAux]);
//             uni_vfmadd132ps(vWCoord, vAux, vAux);
//         }
//         uni_vfmsub132ps(vWCoord, vHalfTmp, vHalfTmp);

//         if (vSrcHeightF.isInitialized()) {
//             uni_vfmadd132ps(vHCoord, vSrcHeightF, vSrcHeightF);
//         } else {
//             auto rAux = getReg64();
//             auto vAux = getVmm();
//             mov(rAux, ptr[regParams + GET_OFF(srcHeightF)]);
//             uni_vpbroadcastd(vAux, ptr[rAux]);
//             uni_vfmadd132ps(vHCoord, vAux, vAux);
//         }
//         uni_vfmsub132ps(vHCoord, vHalfTmp, vHalfTmp);
//     }
// }

// template <>
// void RandomUniform<x64::avx512_core>::zerosPaddingW(const Vmask& kDst, const Vmm& vCoord) {
//     vcmpps(kDst, vCoord, vSrcWidthF, CMP_LT_PS);    // vCoord < vUpperBound
//     vcmpps(kDst | kDst, vZeros, vCoord, CMP_LE_PS); // vCoord >= vZeros
// }

// template <>
// void RandomUniform<x64::avx512_core>::zerosPaddingH(const Vmask& kDst, const Vmm& vCoord, const Vmask& kMaskW) {
//     vcmpps(kDst | kMaskW, vCoord, vSrcHeightF, CMP_LT_PS); // vCoord < vUpperBound
//     vcmpps(kDst | kDst, vZeros, vCoord, CMP_LE_PS);        // vCoord >= vZeros
// }

// template <>
// void RandomUniform<x64::avx512_core>::zerosPadding(const Vmask& kDst, const Vmm& vHCoord, const Vmm& vWCoord) {
//     zerosPaddingW(kDst, vWCoord);
//     zerosPaddingH(kDst, vHCoord, kDst);
// }

// template <>
// void RandomUniform<x64::sse41>::zerosPaddingW(const Vmask& kDst, const Vmm& vWCoord) {
//     auto vAux = getVmm();

//     if (vSrcWidthF.isInitialized()) {
//         uni_vcmpps(vAux, vWCoord, vSrcWidthF, CMP_LT_PS); // vWCoord < vSrcWidthF
//     } else {
//         auto rAux = getReg64();
//         mov(rAux, ptr[regParams + GET_OFF(srcWidthF)]);
//         uni_vcmpps(vAux, vWCoord, ptr[rAux], CMP_LT_PS);  // vWCoord < vSrcWidthF
//     }

//     uni_vpxor(kDst, kDst, kDst);
//     uni_vcmpps(kDst, kDst, vWCoord, CMP_LE_PS);           // vWCoord >= vZeros
//     uni_vpand(kDst, kDst, vAux);                    // vZeros <= vWCoord < vSrcWidthF
// }

// template <>
// void RandomUniform<x64::sse41>::zerosPaddingH(const Vmask& kDst, const Vmm& vHCoord, const Vmask& kMaskW) {
//     auto vAux = getVmm();

//     if (vSrcHeightF.isInitialized()) {
//         uni_vcmpps(vAux, vHCoord, vSrcHeightF, CMP_LT_PS); // vHCoord < vSrcHeightF
//     } else {
//         auto rAux = getReg64();
//         mov(rAux, ptr[regParams + GET_OFF(srcHeightF)]);
//         uni_vcmpps(vAux, vHCoord, ptr[rAux], CMP_LT_PS);   // vHCoord < vSrcHeightF
//     }

//     uni_vmovups(kDst, kMaskW);
//     uni_vpand(kDst, kDst, vAux); // vHCoord < vSrcHeightF && vZeros <= vWCoord < vSrcWidthF
//     uni_vpxor(vAux, vAux, vAux);
//     uni_vcmpps(vAux, vAux, vHCoord, CMP_LE_PS); // vHCoord >= vZeros
//     uni_vpand(kDst, kDst, vAux); // vZeros <= vHCoord < vSrcHeightF && vZeros <= vWCoord < vSrcWidthF
// }

// template <>
// void RandomUniform<x64::sse41>::zerosPadding(const Vmask& kDst, const Vmm& vHCoord, const Vmm& vWCoord) {
//     zerosPaddingW(kDst, vWCoord);
//     zerosPaddingH(kDst, vHCoord, kDst);
// }

// template <x64::cpu_isa_t isa> // Works for AVX2, AVX
// void RandomUniform<isa>::zerosPaddingW(const Vmask& kDst, const Vmm& vCoord) {
//     auto vAux = getVmm();
//     Vmm vZerosTmp;
//     RegistersPool::Reg<Vmm> zerosHolder;
//     if (vZeros.isInitialized()) {
//         vZerosTmp = vZeros;
//     } else {
//         zerosHolder = getVmm();
//         vZerosTmp = zerosHolder;
//         uni_vpxor(vZerosTmp, vZerosTmp, vZerosTmp);
//     }

//     if (vSrcWidthF.isInitialized()) {
//         uni_vcmpps(vAux, vCoord, vSrcWidthF, CMP_LT_PS); // vWCoord < vSrcWidthF
//     } else {
//         auto rAux = getReg64();
//         mov(rAux, ptr[regParams + GET_OFF(srcWidthF)]);
//         uni_vcmpps(vAux, vCoord, ptr[rAux], CMP_LT_PS);  // vWCoord < vSrcWidthF
//     }

//     uni_vcmpps(kDst, vZerosTmp, vCoord, CMP_LE_PS);      // vWCoord >= vZeros
//     uni_vandps(kDst, kDst, vAux);                  // vZeros <= vWCoord < vSrcWidthF
// }

// template <x64::cpu_isa_t isa> // Works for AVX2, AVX
// void RandomUniform<isa>::zerosPaddingH(const Vmask& kDst, const Vmm& vCoord, const Vmask& kMaskW) {
//     auto vAux = getVmm();
//     Vmm vZerosTmp;
//     RegistersPool::Reg<Vmm> zerosHolder;
//     if (vZeros.isInitialized()) {
//         vZerosTmp = vZeros;
//     } else {
//         zerosHolder = getVmm();
//         vZerosTmp = zerosHolder;
//         uni_vpxor(vZerosTmp, vZerosTmp, vZerosTmp);
//     }

//     if (vSrcHeightF.isInitialized()) {
//         uni_vcmpps(vAux, vCoord, vSrcHeightF, CMP_LT_PS); // vHCoord < vSrcHeightF
//     } else {
//         auto rAux = getReg64();
//         mov(rAux, ptr[regParams + GET_OFF(srcHeightF)]);
//         uni_vcmpps(vAux, vCoord, ptr[rAux], CMP_LT_PS);   // vHCoord < vSrcHeightF
//     }

//     uni_vandps(kDst, kMaskW, vAux);
//     uni_vcmpps(vAux, vZerosTmp, vCoord, CMP_LE_PS);       // vHCoord >= vZeros
//     uni_vandps(kDst, kDst, vAux);
// }

// template <x64::cpu_isa_t isa> // Works for AVX2, AVX
// void RandomUniform<isa>::zerosPadding(const Vmask& kDst, const Vmm& vHCoord, const Vmm& vWCoord) {
//     bool releaseZeroVec = false;
//     if (!vZeros.isInitialized()) {
//         releaseZeroVec = true;
//         vZeros = getVmm();
//         uni_vpxor(vZeros, vZeros, vZeros);
//     }

//     zerosPaddingW(kDst, vWCoord);
//     zerosPaddingH(kDst, vHCoord, kDst);

//     if (releaseZeroVec) {
//         vZeros.release();
//     }
// }

// template <>
// void RandomUniform<x64::avx512_core>::borderPadding(const Vmm& vCoordDst, const Vmm& vCoordOrigin, const coord dim) {
//     vrangeps(vCoordDst, vCoordOrigin, dim == coord::w ? vSrcWidthSub1F : vSrcHeightSub1F, 0x0); // vWCoord >= vSrcWidthF
//     vrangeps(vCoordDst, vCoordDst, vZeros, 0x1); // vWCoord < vZeros
// }

// template <x64::cpu_isa_t isa> // Works for AVX2, AVX, SSE41
// void RandomUniform<isa>::borderPadding(const Vmm& vCoordDst, const Vmm& vCoordOrigin, const coord dim) {
//     auto rAux = getReg64();
//     auto vAux = getVmm();
//     RegistersPool::Reg<Vmm> vAux1;

//     Vmm vSub1F;
//     if (dim == coord::w) {
//         if (vSrcWidthSub1F.isInitialized()) {
//             vSub1F = vSrcWidthSub1F;
//         } else {
//             vAux1 = getVmm();
//             vSub1F = vAux1;
//             mov(rAux, ptr[regParams + GET_OFF(srcWidthSub1F)]);
//             uni_vmovups(vSub1F, ptr[rAux]);
//         }
//     } else if (dim == coord::h) {
//         if (vSrcHeightSub1F.isInitialized()) {
//             vSub1F = vSrcHeightSub1F;
//         } else {
//             vAux1 = getVmm();
//             vSub1F = vAux1;
//             mov(rAux, ptr[regParams + GET_OFF(srcHeightSub1F)]);
//             uni_vmovups(vSub1F, ptr[rAux]);
//         }
//     }

//     uni_vcmpps(vAux, vCoordOrigin, vSub1F, CMP_LE_PS);  // vCoord <= vUpperBound
//     uni_vandps(vCoordDst, vCoordOrigin, vAux);
//     uni_vandnps(vAux, vAux, vSub1F);
//     uni_vaddps(vCoordDst, vCoordDst, vAux);

//     if (vZeros.isInitialized()) {
//         uni_vcmpps(vAux, vCoordDst, vZeros, 0x6); // vCoord >= vZeros
//     } else {
//         if (isa == x64::sse41) {
//             if (!vAux1.isInitialized()) {
//                 vAux1 = getVmm();
//                 vSub1F = vAux1;
//             }
//             uni_vpxor(vSub1F, vSub1F, vSub1F);
//             uni_vcmpps(vAux, vCoordDst, vSub1F, 0x6); // vCoord >= vZeros
//         } else {
//             uni_vpxor(vAux, vAux, vAux);
//             uni_vcmpps(vAux, vCoordDst, vAux, 0x6);   // vCoord >= vZeros
//         }
//     }
//     uni_vandps(vCoordDst, vCoordDst, vAux);
// }

// template <>
// void RandomUniform<x64::avx512_core>::reflectionPadding(const Vmm& vCoordDst, const Vmm& vCoordOrigin, const coord dim) {
//     auto vAux = getVmm();
//     auto kAux = getMask();
//     const auto& vSrcDimMul2Sub1F = dim == coord::w ? vSrcWidthMul2Sub1F : vSrcHeightMul2Sub1F;

//     if (jcp.alignCorners) {
//         // abs(x) % D21
//         uni_vandps(vCoordDst, vCoordOrigin, vAbsMask); // abs(x)
//         uni_vdivps(vAux, vCoordDst, vSrcDimMul2Sub1F);
//         uni_vroundps(vAux, vAux, 0x3);                       // Truncation
//         uni_vfnmadd231ps(vCoordDst, vAux, vSrcDimMul2Sub1F); // abs(x) % D21

//         // Check that the result does not exceed the divisor.
//         vcmpps(kAux, vSrcDimMul2Sub1F, vCoordDst, CMP_LE_PS);
//         uni_vmovups(vCoordDst | kAux, vZeros);
//         vrangeps(vCoordDst, vCoordDst, vZeros, 0x1);
//     } else {
//         const auto& vSrcDimMul2F = dim == coord::w ? vSrcWidthMul2F : vSrcHeightMul2F;
//         // (x % D2 + D2) % D2
//         if (vCoordDst.getIdx() != vCoordOrigin.getIdx())
//             uni_vmovups(vCoordDst, vCoordOrigin);
//         uni_vdivps(vAux, vCoordDst, vSrcDimMul2F);
//         uni_vroundps(vAux, vAux, 0x3);                   // Truncation
//         uni_vfnmadd231ps(vCoordDst, vAux, vSrcDimMul2F); // x % D2
//         uni_vaddps(vCoordDst, vCoordDst, vSrcDimMul2F);  // x % D2 + D2
//         uni_vdivps(vAux, vCoordDst, vSrcDimMul2F);
//         uni_vroundps(vAux, vAux, 0x3);                   // Truncation
//         uni_vfnmadd231ps(vCoordDst, vAux, vSrcDimMul2F); // (x % D2 + D2) % D2

//         // Check that the result does not exceed the divisor.
//         vcmpps(kAux, vSrcDimMul2F, vCoordDst, CMP_LE_PS);
//         uni_vmovups(vCoordDst | kAux, vZeros);
//         vrangeps(vCoordDst, vCoordDst, vZeros, 0x1);
//     }

//     uni_vsubps(vAux, vSrcDimMul2Sub1F, vCoordDst);
//     vcmpps(kAux, dim == coord::w ? vSrcWidthF : vSrcHeightF, vCoordDst, CMP_LE_PS); // vCoordDst >= vSrcDimF
//     uni_vmovups(vCoordDst | kAux, vAux);
// }

// template <x64::cpu_isa_t isa> // Works for AVX2, AVX, SSE41
// void RandomUniform<isa>::reflectionPadding(const Vmm& vCoordDst, const Vmm& vCoordOrigin, const coord dim) {
//     auto rAux  = getReg64();
//     auto vAux0 = getVmm();
//     auto vAux1 = getVmm();

//     // D2  = Dim * 2
//     // D21 = (Dim - 1) * 2
//     if (jcp.alignCorners) {
//         // x' = abs(x) % D21 - D21
//         static const unsigned absMask[8] = { 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff };
//         if (isa ==x64::sse41) {
//             static const unsigned *absPtr = absMask + (reinterpret_cast<int64_t>(absMask) % 16) / sizeof(unsigned);
//             mov(rAux, reinterpret_cast<uintptr_t>(absPtr));
//         } else {
//             mov(rAux, reinterpret_cast<uintptr_t>(absMask));
//         }
//         uni_vandps(vCoordDst, vCoordOrigin, ptr[rAux]); // abs(x)

//         Vmm vMul2Sub1;
//         if (dim == coord::w) {
//             if (vSrcWidthMul2Sub1F.isInitialized()) {
//                 vMul2Sub1 = vSrcWidthMul2Sub1F;
//             } else {
//                 vMul2Sub1 = vAux1;
//                 mov(rAux, ptr[regParams + GET_OFF(srcWidthMul2Sub1F)]);
//                 uni_vmovups(vAux1, ptr[rAux]);
//             }
//         } else if (coord::h) {
//             if (vSrcHeightMul2Sub1F.isInitialized()) {
//                 vMul2Sub1 = vSrcHeightMul2Sub1F;
//             } else {
//                 vMul2Sub1 = vAux1;
//                 mov(rAux, ptr[regParams + GET_OFF(srcHeightMul2Sub1F)]);
//                 uni_vmovups(vAux1, ptr[rAux]);
//             }
//         }
//         uni_vdivps(vAux0, vCoordDst, vMul2Sub1);
//         uni_vroundps(vAux0, vAux0, 0x3);               // Truncation
//         uni_vfnmadd231ps(vCoordDst, vAux0, vMul2Sub1); // abs(x) % D21

//         // Check that the result does not exceed the divisor.
//         uni_vcmpps(vAux0, vCoordDst, vMul2Sub1, CMP_LT_PS);
//         uni_vandps(vCoordDst, vCoordDst, vAux0);
//         uni_vxorps(vAux0, vAux0, vAux0);
//         uni_vcmpps(vAux0, vAux0, vCoordDst, CMP_LE_PS);
//         uni_vandps(vCoordDst, vCoordDst, vAux0);

//         uni_vsubps(vAux0, vCoordDst, vMul2Sub1);       // abs(x) % D21 - D21
//     } else {
//         // x' = (x % D2 + D2) % D2 - D21
//         if (vCoordDst.getIdx() != vCoordOrigin.getIdx())
//             uni_vmovups(vCoordDst, vCoordOrigin);
//         Vmm vMul2;
//         if (dim == coord::w) {
//             if (vSrcWidthMul2F.isInitialized()) {
//                 vMul2 = vSrcWidthMul2F;
//             } else {
//                 vMul2 = vAux1;
//                 mov(rAux, ptr[regParams + GET_OFF(srcWidthMul2F)]);
//                 uni_vmovups(vAux1, ptr[rAux]);
//             }
//         } else if (coord::h) {
//             if (vSrcHeightMul2F.isInitialized()) {
//                 vMul2 = vSrcHeightMul2F;
//             } else {
//                 vMul2 = vAux1;
//                 mov(rAux, ptr[regParams + GET_OFF(srcHeightMul2F)]);
//                 uni_vmovups(vAux1, ptr[rAux]);
//             }
//         }
//         uni_vdivps(vAux0, vCoordOrigin, vMul2);
//         uni_vroundps(vAux0, vAux0, 0x3);           // Truncation
//         uni_vfnmadd231ps(vCoordDst, vAux0, vMul2); // x % D2
//         uni_vaddps(vCoordDst, vCoordDst, vMul2);   // x % D2 + D2
//         uni_vdivps(vAux0, vCoordDst, vMul2);
//         uni_vroundps(vAux0, vAux0, 0x3);           // Truncation
//         uni_vfnmadd231ps(vCoordDst, vAux0, vMul2); // (x % D2 + D2) % D2

//         // Check that the result does not exceed the divisor.
//         uni_vcmpps(vAux0, vCoordDst, vMul2, CMP_LT_PS);
//         uni_vandps(vCoordDst, vCoordDst, vAux0);
//         uni_vxorps(vAux0, vAux0, vAux0);
//         uni_vcmpps(vAux0, vAux0, vCoordDst, CMP_LE_PS);
//         uni_vandps(vCoordDst, vCoordDst, vAux0);

//         if (dim == coord::w) {
//             if (vSrcWidthMul2Sub1F.isInitialized()) {
//                 uni_vsubps(vAux0, vCoordDst, vSrcWidthMul2Sub1F);
//             } else {
//                 mov(rAux, ptr[regParams + GET_OFF(srcWidthMul2Sub1F)]);
//                 uni_vsubps(vAux0, vCoordDst, ptr[rAux]);
//             }
//         } else if (coord::h) {
//             if (vSrcHeightMul2Sub1F.isInitialized()) {
//                 uni_vsubps(vAux0, vCoordDst, vSrcHeightMul2Sub1F);
//             } else {
//                 mov(rAux, ptr[regParams + GET_OFF(srcHeightMul2Sub1F)]);
//                 uni_vsubps(vAux0, vCoordDst, ptr[rAux]);
//             }
//         }
//     }

//     if (dim == coord::w) {
//         if (vSrcWidthF.isInitialized()) {
//             uni_vcmpps(vAux1, vCoordDst, vSrcWidthF, CMP_LT_PS);  // vCoordDst < vUpperBound
//         } else {
//             mov(rAux, ptr[regParams + GET_OFF(srcWidthF)]);
//             uni_vcmpps(vAux1, vCoordDst, ptr[rAux], CMP_LT_PS);   // vCoordDst < vUpperBound
//         }
//     } else {
//         if (vSrcHeightF.isInitialized()) {
//             uni_vcmpps(vAux1, vCoordDst, vSrcHeightF, CMP_LT_PS); // vCoordDst < vUpperBound
//         } else {
//             mov(rAux, ptr[regParams + GET_OFF(srcHeightF)]);
//             uni_vcmpps(vAux1, vCoordDst, ptr[rAux], CMP_LT_PS);   // vCoordDst < vUpperBound
//         }
//     }

//     uni_vandps(vCoordDst, vCoordDst, vAux1);
//     uni_vandnps(vAux1, vAux1, vAux0);
//     uni_vsubps(vCoordDst, vCoordDst, vAux1); // set -x' for vCoordDst >= Dim
// }

// template <>
// void RandomUniform<x64::avx512_core>::bicubicCoefficients(const Vmm& vCoef, const Vmm& vDDim, const uint8_t idx) {
//     if (idx == 0) {
//         uni_vmovups(vCoef, vDDim);
//         uni_vfnmadd132ps(vCoef, vOnesF, vConst_2_00);
//         uni_vfmadd231ps(vCoef, vDDim, vDDim);
//         uni_vmulps(vCoef, vCoef, vDDim);
//         uni_vmulps(vCoef, vCoef, vConst_0_75);
//     } else if (idx == 1) {
//         uni_vmovups(vCoef, vDDim);
//         vfmsub132ps(vCoef, vConst_2_25, vConst_1_25);
//         uni_vmulps(vCoef, vCoef, vDDim);
//         uni_vfmadd132ps(vCoef, vOnesF, vDDim);
//     } else if (idx == 2) {
//         uni_vmovups(vCoef, vDDim);
//         uni_vfnmadd132ps(vCoef, vConst_1_50, vConst_1_25);
//         uni_vfmsub132ps(vCoef, vConst_0_75, vDDim);
//         uni_vmulps(vCoef, vCoef, vDDim);
//     } else if (idx == 3) {
//         uni_vmulps(vCoef, vConst_0_75, vDDim);
//         uni_vmulps(vCoef, vCoef, vDDim);
//         uni_vfnmadd132ps(vCoef, vCoef, vDDim);
//     }
// }

// template <>
// void RandomUniform<x64::avx2>::bicubicCoefficients(const Vmm& vCoef, const Vmm& vDDim, const uint8_t idx) {
//     static const size_t elPerVec = x64::cpu_isa_traits<x64::avx2>::vlen / sizeof(float);;
//     static const float const_0_75[elPerVec] = { -0.75f, -0.75f, -0.75f, -0.75f, -0.75f, -0.75f, -0.75f, -0.75f };
//     static const float const_1_25[elPerVec] = { 1.25f, 1.25f, 1.25f, 1.25f, 1.25f, 1.25f, 1.25f, 1.25f };
//     static const float const_1_50[elPerVec] = { 1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f };
//     static const float const_2_00[elPerVec] = { 2.f, 2.f, 2.f, 2.f, 2.f, 2.f, 2.f, 2.f };
//     static const float const_2_25[elPerVec] = { 2.25f, 2.25f, 2.25f, 2.25f, 2.25f, 2.25f, 2.25f, 2.25f };

//     auto rAux = getReg64();

//     if (idx == 0) {
//         uni_vmovups(vCoef, vDDim);
//         mov(rAux, reinterpret_cast<uintptr_t>(const_2_00));
//         uni_vfnmadd132ps(vCoef, vOnesF, ptr[rAux]);
//         uni_vfmadd231ps(vCoef, vDDim, vDDim);
//         uni_vmulps(vCoef, vCoef, vDDim);
//         mov(rAux, reinterpret_cast<uintptr_t>(const_0_75));
//         uni_vmulps(vCoef, vCoef, ptr[rAux]);
//     } else if (idx == 1) {
//         mov(rAux, reinterpret_cast<uintptr_t>(const_1_25));
//         uni_vmulps(vCoef, vDDim, ptr[rAux]);
//         mov(rAux, reinterpret_cast<uintptr_t>(const_2_25));
//         uni_vsubps(vCoef, vCoef, ptr[rAux]);
//         uni_vmulps(vCoef, vCoef, vDDim);
//         uni_vfmadd132ps(vCoef, vOnesF, vDDim);
//     } else if (idx == 2) {
//         mov(rAux, reinterpret_cast<uintptr_t>(const_1_25));
//         uni_vmulps(vCoef, vDDim, ptr[rAux]);
//         mov(rAux, reinterpret_cast<uintptr_t>(const_1_50));
//         uni_vsubps(vCoef, vCoef, ptr[rAux]);
//         mov(rAux, reinterpret_cast<uintptr_t>(const_0_75));
//         vfnmsub213ps(vCoef, vDDim, ptr[rAux]);
//         uni_vmulps(vCoef, vCoef, vDDim);
//     } else if (idx == 3) {
//         mov(rAux, reinterpret_cast<uintptr_t>(const_0_75));
//         uni_vmulps(vCoef, vDDim, ptr[rAux]);
//         uni_vmulps(vCoef, vCoef, vDDim);
//         uni_vfnmadd132ps(vCoef, vCoef, vDDim);
//     }
// }

// template <>
// void RandomUniform<x64::avx>::bicubicCoefficients(const Vmm& vCoef, const Vmm& vDDim, const uint8_t idx) {
//     static const size_t elPerVec = x64::cpu_isa_traits<x64::avx>::vlen / sizeof(float);
//     static const float const_0_75[elPerVec] = { -0.75f, -0.75f, -0.75f, -0.75f, -0.75f, -0.75f, -0.75f, -0.75f };
//     static const float const_1_25[elPerVec] = { 1.25f, 1.25f, 1.25f, 1.25f, 1.25f, 1.25f, 1.25f, 1.25f };
//     static const float const_1_50[elPerVec] = { 1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f };
//     static const float const_2_00[elPerVec] = { 2.f, 2.f, 2.f, 2.f, 2.f, 2.f, 2.f, 2.f };
//     static const float const_2_25[elPerVec] = { 2.25f, 2.25f, 2.25f, 2.25f, 2.25f, 2.25f, 2.25f, 2.25f };

//     auto rAux = getReg64();
//     auto vAux = getVmm();

//     if (idx == 0) {
//         uni_vmovups(vCoef, vDDim);
//         mov(rAux, reinterpret_cast<uintptr_t>(const_2_00));
//         uni_vfnmadd132ps(vCoef, vOnesF, ptr[rAux]);
//         uni_vmulps(vAux, vDDim, vDDim);
//         uni_vaddps(vCoef, vCoef, vAux);
//         uni_vmulps(vCoef, vCoef, vDDim);
//         mov(rAux, reinterpret_cast<uintptr_t>(const_0_75));
//         uni_vmulps(vCoef, vCoef, ptr[rAux]);
//     } else if (idx == 1) {
//         mov(rAux, reinterpret_cast<uintptr_t>(const_1_25));
//         uni_vmulps(vCoef, vDDim, ptr[rAux]);
//         mov(rAux, reinterpret_cast<uintptr_t>(const_2_25));
//         uni_vsubps(vCoef, vCoef, ptr[rAux]);
//         uni_vmulps(vCoef, vCoef, vDDim);
//         uni_vfmadd132ps(vCoef, vOnesF, vDDim);
//     } else if (idx == 2) {
//         mov(rAux, reinterpret_cast<uintptr_t>(const_1_25));
//         uni_vmulps(vAux, vDDim, ptr[rAux]);
//         mov(rAux, reinterpret_cast<uintptr_t>(const_1_50));
//         uni_vmovups(vCoef, ptr[rAux]);
//         uni_vsubps(vCoef, vCoef, vAux);
//         uni_vmulps(vCoef, vCoef, vDDim);
//         mov(rAux, reinterpret_cast<uintptr_t>(const_0_75));
//         uni_vsubps(vCoef, vCoef, ptr[rAux]);
//         uni_vmulps(vCoef, vCoef, vDDim);
//     } else if (idx == 3) {
//         mov(rAux, reinterpret_cast<uintptr_t>(const_0_75));
//         uni_vmulps(vCoef, vDDim, ptr[rAux]);
//         uni_vmulps(vCoef, vCoef, vDDim);
//         uni_vmulps(vAux, vCoef, vDDim);
//         uni_vsubps(vCoef, vCoef, vAux);
//     }
// }

// template <>
// void RandomUniform<x64::sse41>::bicubicCoefficients(const Vmm& vCoef, const Vmm& vDDim, const uint8_t idx) {
//     static const size_t elToAllocate = 2 * x64::cpu_isa_traits<x64::sse41>::vlen / sizeof(float);
//     // Allocation with a margin for address alignment.
//     static const float c_0_75[elToAllocate] = { -0.75f, -0.75f, -0.75f, -0.75f, -0.75f, -0.75f, -0.75f, -0.75f };
//     static const float c_1_25[elToAllocate] = { 1.25f, 1.25f, 1.25f, 1.25f, 1.25f, 1.25f, 1.25f, 1.25f };
//     static const float c_1_50[elToAllocate] = { 1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f };
//     static const float c_2_00[elToAllocate] = { 2.f, 2.f, 2.f, 2.f, 2.f, 2.f, 2.f, 2.f };
//     static const float c_2_25[elToAllocate] = { 2.25f, 2.25f, 2.25f, 2.25f, 2.25f, 2.25f, 2.25f, 2.25f };
//     // Address alignment for XMM.
//     static const float* const_0_75 = c_0_75 + (reinterpret_cast<int64_t>(c_0_75) % 16) / sizeof(float);
//     static const float* const_1_25 = c_1_25 + (reinterpret_cast<int64_t>(c_1_25) % 16) / sizeof(float);
//     static const float* const_1_50 = c_1_50 + (reinterpret_cast<int64_t>(c_1_50) % 16) / sizeof(float);
//     static const float* const_2_00 = c_2_00 + (reinterpret_cast<int64_t>(c_2_00) % 16) / sizeof(float);
//     static const float* const_2_25 = c_2_25 + (reinterpret_cast<int64_t>(c_2_25) % 16) / sizeof(float);

//     auto rAux = getReg64();
//     auto vAux = getVmm();

//     if (idx == 0) {
//         uni_vmovups(vAux, vDDim);
//         mov(rAux, reinterpret_cast<uintptr_t>(const_2_00));
//         uni_vmulps(vAux, vAux, ptr[rAux]);
//         uni_vsubps(vAux, vAux, vOnesF);
//         uni_vmovups(vCoef, vDDim);
//         uni_vmulps(vCoef, vCoef, vCoef);
//         uni_vsubps(vCoef, vCoef, vAux);
//         uni_vmulps(vCoef, vCoef, vDDim);
//         mov(rAux, reinterpret_cast<uintptr_t>(const_0_75));
//         uni_vmulps(vCoef, vCoef, ptr[rAux]);
//     } else if (idx == 1) {
//         uni_vmovups(vCoef, vDDim);
//         mov(rAux, reinterpret_cast<uintptr_t>(const_1_25));
//         uni_vmulps(vCoef, vCoef, ptr[rAux]);
//         mov(rAux, reinterpret_cast<uintptr_t>(const_2_25));
//         uni_vsubps(vCoef, vCoef, ptr[rAux]);
//         uni_vmulps(vCoef, vCoef, vDDim);
//         uni_vfmadd132ps(vCoef, vOnesF, vDDim);
//     } else if (idx == 2) {
//         uni_vmovups(vAux, vDDim);
//         uni_vmulps(vAux, vDDim, vDDim);
//         mov(rAux, reinterpret_cast<uintptr_t>(const_1_25));
//         uni_vmulps(vAux, vAux, ptr[rAux]);
//         mov(rAux, reinterpret_cast<uintptr_t>(const_0_75));
//         uni_vaddps(vAux, vAux, ptr[rAux]);
//         uni_vmovups(vCoef, vDDim);
//         mov(rAux, reinterpret_cast<uintptr_t>(const_1_50));
//         uni_vmulps(vCoef, vCoef, ptr[rAux]);
//         uni_vsubps(vCoef, vCoef, vAux);
//         uni_vmulps(vCoef, vCoef, vDDim);
//     } else if (idx == 3) {
//         uni_vmovups(vCoef, vDDim);
//         mov(rAux, reinterpret_cast<uintptr_t>(const_0_75));
//         uni_vmulps(vCoef, vCoef, ptr[rAux]);
//         uni_vmulps(vCoef, vCoef, vDDim);
//         uni_vmovups(vAux, vCoef);
//         uni_vmulps(vAux, vAux, vDDim);
//         uni_vsubps(vCoef, vCoef, vAux);
//     }
// }

// template <x64::cpu_isa_t isa> // Works for AVX512, AVX2, AVX, SSE41
// void RandomUniform<isa>::nearestInterpolation(const Vmm& vWCoord, const Vmm& vHCoord, bool tail) {
//     const auto& vSrcShift = vWCoord;
//     const auto& vAux      = vHCoord;
//     auto kGatherMask      = getMask();
//     auto kAuxMask         = getMask();

//     uni_vroundps(vWCoord, vWCoord, 0x0); // Round near
//     uni_vroundps(vHCoord, vHCoord, 0x0); // Round near

//     bool useMask = false, zeroFill = false;
//     if (jcp.paddingMode == GridSamplePaddingMode::ZEROS) {
//         useMask = zeroFill = true;
//         zerosPadding(kGatherMask, vHCoord, vWCoord);
//     } else if (jcp.paddingMode == GridSamplePaddingMode::BORDER) {
//         borderPadding(vWCoord, vWCoord, coord::w);
//         borderPadding(vHCoord, vHCoord, coord::h);
//     } else if (jcp.paddingMode == GridSamplePaddingMode::REFLECTION) {
//         reflectionPadding(vWCoord, vWCoord, coord::w);
//         reflectionPadding(vHCoord, vHCoord, coord::h);
//     }

//     hwShiftPs2dq(vSrcShift, vHCoord, vWCoord, vSrcWidthF);

//     // PER CHANNEL LOOP
//     Xbyak::Label lChannelLoopBegin, lChannelLoopEnd;
//     RegistersPool::Reg<Xbyak::Reg64> rChannel;
//     auto rSrcTmp = getReg64();
//     auto rDstTmp = getReg64();
//     mov(rSrcTmp, regSrc);
//     mov(rDstTmp, r64_dst);

//     for (uint64_t ch = 0; ch < jcp.cannelNum; ch++) {
//         if (jcp.dynamicChannel) {
//             rChannel = getReg64();
//             mov(rChannel, ptr[regParams + GET_OFF(channelsNum)]);

//             L(lChannelLoopBegin);
//             cmp(rChannel, 0);
//             jle(lChannelLoopEnd, T_NEAR);
//         }

//         if (jcp.paddingMode == GridSamplePaddingMode::ZEROS) {
//             if (isa == x64::avx512_core && tail)
//                 uni_kandd(kAuxMask, kTailMask, kGatherMask);
//             else
//                 uni_kmovd(kAuxMask, kGatherMask);
//         }

//         if (!tail) {
//             gatherdd(vAux, rSrcTmp, vSrcShift, kAuxMask, useMask, zeroFill);
//             uni_vmovups(ptr[rDstTmp], vAux);
//         } else {
//             if (isa == x64::avx512_core) {
//                 if (jcp.paddingMode != GridSamplePaddingMode::ZEROS) {
//                     uni_kmovd(kAuxMask, kTailMask);
//                 }
//                 gatherdd(vAux, rSrcTmp, vSrcShift, kAuxMask, tail, zeroFill);
//                 uni_vmovups(ptr[rDstTmp] | Xbyak::Opmask(kTailMask.getIdx()), vAux);
//             } else {
//                 memMovDD(rDstTmp, rSrcTmp, Vmm(kAuxMask.getIdx()), vSrcShift, r64_work_amount, useMask, zeroFill);
//             }
//         }

//         add(rSrcTmp, regSrcChannelStepB);
//         add(rDstTmp, regDstChannelStepB);

//         if (jcp.dynamicChannel) {
//             dec(rChannel);
//             jmp(lChannelLoopBegin, T_NEAR);
//             L(lChannelLoopEnd);
//         }
//     }
// }

// template <>
// void RandomUniform<x64::avx512_core>::bilinearInterpolation(const Vmm& vWCoord, const Vmm& vHCoord, bool tail) {
//     const auto& vDX = vWCoord;
//     const auto& vDY = vHCoord;
//     auto shift00    = getVmm();
//     auto shift01    = getVmm();
//     auto shift10    = getVmm();
//     auto shift11    = getVmm();
//     auto vAux       = getVmm();
//     RegistersPool::Reg<Vmask> kMask00, kMask01, kMask10, kMask11;

//     uni_vroundps(shift00, vWCoord, 0x1); // Round floor
//     uni_vroundps(shift01, vHCoord, 0x1); // Round floor
//     uni_vsubps(vDX, vWCoord, shift00);
//     uni_vsubps(vDY, vHCoord, shift01);
//     uni_vaddps(shift10, shift00, vOnesF);
//     uni_vaddps(shift11, shift01, vOnesF);

//     bool useMask = false, zeroFill = false;
//     if (jcp.paddingMode == GridSamplePaddingMode::ZEROS) {
//         useMask = zeroFill = true;
//         kMask00 = getMask();
//         kMask01 = getMask();
//         kMask10 = getMask();
//         kMask11 = getMask();

//         zerosPadding(kMask00, shift01, shift00); // (y; x)
//         zerosPadding(kMask01, shift01, shift10); // (y; x + 1)
//         zerosPadding(kMask11, shift11, shift10); // (y + 1; x + 1)
//         zerosPadding(kMask10, shift11, shift00); // (y + 1; x)

//         hwShiftPs2dq(shift00, shift01, shift00, vSrcWidthF);
//         uni_vpaddd(shift01, shift00, vDataTypeSizeB);
//         uni_vpaddd(shift10, shift00, vSrcWidthB);
//         uni_vpaddd(shift11, shift10, vDataTypeSizeB);
//     } else if (jcp.paddingMode == GridSamplePaddingMode::BORDER) {
//         borderPadding(shift00, shift00, coord::w);
//         borderPadding(shift01, shift01, coord::h);
//         borderPadding(shift10, shift10, coord::w);
//         borderPadding(shift11, shift11, coord::h);
//     } else if (jcp.paddingMode == GridSamplePaddingMode::REFLECTION) {
//         reflectionPadding(shift00, shift00, coord::w);
//         reflectionPadding(shift01, shift01, coord::h);
//         reflectionPadding(shift10, shift10, coord::w);
//         reflectionPadding(shift11, shift11, coord::h);
//     }
//     if (jcp.paddingMode == GridSamplePaddingMode::BORDER || jcp.paddingMode == GridSamplePaddingMode::REFLECTION) {
//         // W * y + x
//         hwShiftPs2dq(vAux, shift11, shift00, vSrcWidthF);
//         hwShiftPs2dq(shift00, shift01, shift00, vSrcWidthF);
//         hwShiftPs2dq(shift01, shift01, shift10, vSrcWidthF);
//         hwShiftPs2dq(shift11, shift11, shift10, vSrcWidthF);
//         uni_vmovups(shift10, vAux);
//     }

//     auto kAuxMask = getMask();
//     auto vQ0 = getVmm();
//     auto vQ1 = getVmm();

//     // PER CHANNEL LOOP
//     Xbyak::Label lChannelLoopBegin, lChannelLoopEnd;
//     RegistersPool::Reg<Xbyak::Reg64> rChannel;
//     auto rSrcTmp  = getReg64();
//     auto rDstTmp  = getReg64();
//     mov(rSrcTmp, regSrc);
//     mov(rDstTmp, r64_dst);

//     for (uint64_t ch = 0; ch < jcp.cannelNum; ch++) {
//         if (jcp.dynamicChannel) {
//             rChannel = getReg64();
//             mov(rChannel, 0);

//             L(lChannelLoopBegin);
//             cmp(rChannel, regChannelNum);
//             jge(lChannelLoopEnd, T_NEAR);
//         }

//         // (y; x)
//         if (jcp.paddingMode == GridSamplePaddingMode::ZEROS) {
//             kmovw(kAuxMask, kMask00);
//         }
//         gatherdd(vQ0, rSrcTmp, shift00, kAuxMask, useMask, zeroFill); // v00 -> vQ0
//         if (jcp.inDataPrc == InferenceEngine::Precision::I32) {
//             uni_vcvtdq2ps(vQ0, vQ0);
//         }
//         uni_vfmsub213ps(vQ0, vDX, vQ0); // q0 = -(v00 - dx * v00)

//         // (y; x + 1)
//         if (jcp.paddingMode == GridSamplePaddingMode::ZEROS) {
//             kmovw(kAuxMask, kMask01);
//         }
//         gatherdd(vAux, rSrcTmp, shift01, kAuxMask, useMask, zeroFill);
//         if (jcp.inDataPrc == InferenceEngine::Precision::I32) {
//             uni_vcvtdq2ps(vAux, vAux);
//         }
//         uni_vfmsub231ps(vQ0, vAux, vDX); // q0 = -q0 + dx * v01

//         // (y + 1; x + 1)
//         if (jcp.paddingMode == GridSamplePaddingMode::ZEROS) {
//             kmovw(kAuxMask, kMask11);
//         }
//         gatherdd(vAux, rSrcTmp, shift11, kAuxMask, useMask, zeroFill);
//         if (jcp.inDataPrc == InferenceEngine::Precision::I32) {
//             uni_vcvtdq2ps(vAux, vAux);
//         }

//         // (y + 1; x)
//         if (jcp.paddingMode == GridSamplePaddingMode::ZEROS) {
//             kmovw(kAuxMask, kMask10);
//         }
//         gatherdd(vQ1, rSrcTmp, shift10, kAuxMask, useMask, zeroFill);
//         if (jcp.inDataPrc == InferenceEngine::Precision::I32) {
//             uni_vcvtdq2ps(vQ1, vQ1);
//         }

//         uni_vfmsub213ps(vQ1, vDX, vQ1);  // q1 = -(v10 - dx * v10)
//         uni_vfmsub231ps(vQ1, vAux, vDX); // q1 = -q1 + dx * v11
//         // Res = q0 + dy * (q1 - q0)
//         uni_vsubps(vQ1, vQ1, vQ0);
//         uni_vfmadd132ps(vQ1, vQ0, vDY);

//         if (jcp.inDataPrc == InferenceEngine::Precision::I32) {
//             uni_vroundps(vQ1, vQ1, 0x3); // Truncation
//             uni_vcvtps2dq(vQ1, vQ1);
//         }

//         if (!tail) {
//             uni_vmovups(ptr[rDstTmp], vQ1);
//         } else {
//             uni_vmovups(ptr[rDstTmp] | kTailMask, vQ1);
//         }
//         add(rSrcTmp, regSrcChannelStepB);
//         add(rDstTmp, regDstChannelStepB);

//         if (jcp.dynamicChannel) {
//             inc(rChannel);
//             jmp(lChannelLoopBegin, T_NEAR);
//             L(lChannelLoopEnd);
//         }
//     }
// }

// template <x64::cpu_isa_t isa> // Works for AVX2, AVX, SSE41
// void RandomUniform<isa>::bilinearInterpolation(const Vmm& vWCoord, const Vmm& vHCoord, bool tail) {
//     auto vWRound = getVmm();
//     auto vHRound = getVmm();
//     auto& vDX    = vWCoord;
//     auto& vDY    = vHCoord;
//     auto vAux    = getVmm();
//     Vmm shift00, shift01, shift10, shift11;
//     RegistersPool::Reg<Vmm> shift10Holder, shift11Holder;
//     // For ZEROS padding only.
//     RegistersPool::Reg<Vmm> vMask00, vMask01, vMask10, vMask11;

//     uni_vroundps(vWRound, vWCoord, 0x1); // Round floor
//     uni_vroundps(vHRound, vHCoord, 0x1); // Round floor
//     uni_vsubps(vDX, vDX, vWRound);
//     uni_vsubps(vDY, vDY, vHRound);

//     if (jcp.paddingMode != GridSamplePaddingMode::ZEROS) {
//         shift00 = vWRound;
//         shift01 = vHRound;
//         shift10Holder = getVmm();
//         shift10 = shift10Holder;
//         shift11Holder = getVmm();
//         shift11 = shift11Holder;

//         uni_vaddps(shift10, vWRound, vOnesF);
//         uni_vaddps(shift11, vHRound, vOnesF);
//     }

//     bool useMask = false, zeroFill = false;
//     if (jcp.paddingMode == GridSamplePaddingMode::ZEROS) {
//         useMask = zeroFill = true;
//         {
//             auto rAux = getReg64();
//             static const float onesArr[8] = { 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f };
//             if (isa ==x64::sse41) {
//                 static const float *onesPtr = onesArr + (reinterpret_cast<int64_t>(onesArr) % 16) / sizeof(float);
//                 mov(rAux, reinterpret_cast<uintptr_t>(onesPtr));
//             } else {
//                 mov(rAux, reinterpret_cast<uintptr_t>(onesArr));
//             }
//             uni_vmovups(vAux, ptr[rAux]);
//         }
//         shift00 = vWRound;
//         shift10 = vHRound;
//         vMask00 = getVmm();
//         vMask01 = getVmm();
//         vMask10 = getVmm();
//         vMask11 = getVmm();

//         uni_vaddps(vMask00, vWRound, vAux);
//         uni_vaddps(vAux, vAux, vHRound);

//         zerosPadding(vMask01, vHRound, vMask00); // (y; x + 1)
//         zerosPadding(vMask10, vAux, vWRound);    // (y + 1; x)
//         zerosPadding(vMask11, vAux, vMask00);    // (y + 1; x + 1)
//         zerosPadding(vMask00, vHRound, vWRound); // (y; x)

//         hwShiftPs2dq(shift00, vHRound, vWRound, vSrcWidthF);
//     } else if (jcp.paddingMode == GridSamplePaddingMode::BORDER) {
//         borderPadding(vWRound, vWRound, coord::w);
//         borderPadding(vHRound, vHRound, coord::h);
//         borderPadding(shift10, shift10, coord::w);
//         borderPadding(shift11, shift11, coord::h);
//     } else if (jcp.paddingMode == GridSamplePaddingMode::REFLECTION) {
//         reflectionPadding(vWRound, vWRound, coord::w);
//         reflectionPadding(vHRound, vHRound, coord::h);
//         reflectionPadding(shift10, shift10, coord::w);
//         reflectionPadding(shift11, shift11, coord::h);
//     }
//     if (one_of(jcp.paddingMode, GridSamplePaddingMode::BORDER, GridSamplePaddingMode::REFLECTION)) {
//         // W * y + x
//         hwShiftPs2dq(vAux, shift11, vWRound, vSrcWidthF);
//         hwShiftPs2dq(vWRound, vHRound, vWRound, vSrcWidthF);
//         hwShiftPs2dq(vHRound, vHRound, shift10, vSrcWidthF);
//         hwShiftPs2dq(shift11, shift11, shift10, vSrcWidthF);
//         uni_vmovups(shift10, vAux);
//     }

//     auto vGatherMask = getVmm();
//     auto vQ0         = getVmm();
//     auto vQ1         = getVmm();

//     // PER CHANNEL LOOP
//     Xbyak::Label lChannelLoopBegin, lChannelLoopEnd;
//     RegistersPool::Reg<Xbyak::Reg64> rChannel;
//     auto rSrcTmp   = getReg64();
//     auto rDstTmp   = getReg64();
//     auto rTypeSize = getReg64();
//     mov(rSrcTmp,   regSrc);
//     mov(rDstTmp,   r64_dst);
//     mov(rTypeSize, ptr[regParams + GET_OFF(dataTypeSize)]);

//     for (uint64_t ch = 0; ch < jcp.cannelNum; ch++) {
//         if (jcp.dynamicChannel) {
//             rChannel = getReg64();
//             mov(rChannel, ptr[regParams + GET_OFF(channelsNum)]);

//             L(lChannelLoopBegin);
//             cmp(rChannel, 0);
//             jle(lChannelLoopEnd, T_NEAR);
//         }

//         // (y; x)
//         if (jcp.paddingMode == GridSamplePaddingMode::ZEROS && isa == x64::avx2) {
//             uni_vmovups(vGatherMask, vMask00);
//         }
//         gatherdd(vQ0, rSrcTmp, shift00, (isa == x64::avx2 || !vMask00.isInitialized()) ? vGatherMask : vMask00, useMask, zeroFill); // v00 -> vQ0
//         if (jcp.inDataPrc == InferenceEngine::Precision::I32) {
//             uni_vcvtdq2ps(vQ0, vQ0);
//         }
//         if (isa == x64::avx2) {
//             uni_vfmsub213ps(vQ0, vDX, vQ0); // q0 = -(v00 - dx * v00)
//         } else {
//             uni_vmulps(vGatherMask, vQ0, vDX);
//             uni_vsubps(vQ0, vQ0, vGatherMask);
//         }

//         // (y; x + 1)
//         if (jcp.paddingMode == GridSamplePaddingMode::ZEROS) {
//             uni_vpaddd(shift10, shift00, ptr[rTypeSize]);
//             if (isa == x64::avx2)
//                 uni_vmovups(vGatherMask, vMask01);
//         }
//         gatherdd(vAux, rSrcTmp, jcp.paddingMode != GridSamplePaddingMode::ZEROS ? shift01 : shift10,
//                  (isa == x64::avx2 || !vMask01.isInitialized()) ? vGatherMask : vMask01, useMask, zeroFill);
//         if (jcp.inDataPrc == InferenceEngine::Precision::I32) {
//             uni_vcvtdq2ps(vAux, vAux);
//         }
//         if (isa == x64::avx2) {
//             uni_vfmsub231ps(vQ0, vAux, vDX); // q0 = -q0 + dx * v01
//         } else {
//             uni_vmulps(vAux, vAux, vDX);
//             uni_vaddps(vQ0, vQ0, vAux);
//         }

//         // (y + 1; x + 1)
//         if (jcp.paddingMode == GridSamplePaddingMode::ZEROS) {
//             {
//                 auto rSrcWidth = getReg64();
//                 mov(rSrcWidth, ptr[regParams + GET_OFF(srcWidthB)]);
//                 uni_vpaddd(shift10, shift10, ptr[rSrcWidth]);
//             }
//             if (isa == x64::avx2)
//                 uni_vmovups(vGatherMask, vMask11);
//         }
//         gatherdd(vAux, rSrcTmp, jcp.paddingMode != GridSamplePaddingMode::ZEROS ? shift11 : shift10,
//                  (isa == x64::avx2 || !vMask11.isInitialized()) ? vGatherMask : vMask11, useMask, zeroFill);
//         if (jcp.inDataPrc == InferenceEngine::Precision::I32) {
//             uni_vcvtdq2ps(vAux, vAux);
//         }

//         // (y + 1; x)
//         if (jcp.paddingMode == GridSamplePaddingMode::ZEROS) {
//             uni_vpsubd(shift10, shift10, ptr[rTypeSize]);
//             if (isa == x64::avx2)
//                 uni_vmovups(vGatherMask, vMask10);
//         }
//         gatherdd(vQ1, rSrcTmp, shift10, (isa == x64::avx2 || !vMask10.isInitialized()) ? vGatherMask : vMask10, useMask, zeroFill);
//         if (jcp.inDataPrc == InferenceEngine::Precision::I32) {
//             uni_vcvtdq2ps(vQ1, vQ1);
//         }

//         // q1 = -(v10 - dx * v10)
//         if (isa == x64::avx2) {
//             uni_vfmsub213ps(vQ1, vDX, vQ1);
//         } else {
//             uni_vmulps(vGatherMask, vQ1, vDX);
//             if (isa == x64::avx) {
//                 uni_vsubps(vQ1, vGatherMask, vQ1);
//             } else {
//                 uni_vsubps(vGatherMask, vGatherMask, vQ1);
//                 uni_vmovups(vQ1, vGatherMask);
//             }
//         }
//         uni_vfmsub231ps(vQ1, vAux, vDX); // q1 = -q1 + dx * v11
//         // Res = q0 + dy * (q1 - q0)
//         uni_vsubps(vQ1, vQ1, vQ0);
//         uni_vfmadd132ps(vQ1, vQ0, vDY);

//         if (jcp.inDataPrc == InferenceEngine::Precision::I32) {
//             uni_vroundps(vQ1, vQ1, 0x3); // Truncation
//             uni_vcvtps2dq(vQ1, vQ1);
//         }

//         if (!tail) {
//             uni_vmovups(ptr[rDstTmp], vQ1);
//         } else {
//             store(ptr[rDstTmp], vQ1, r64_work_amount, dataTypeSize);
//         }

//         add(rSrcTmp, regSrcChannelStepB);
//         add(rDstTmp, regDstChannelStepB);

//         if (jcp.dynamicChannel) {
//             dec(rChannel);
//             jmp(lChannelLoopBegin, T_NEAR);
//             L(lChannelLoopEnd);
//         }
//     }
// }

// template <>
// void RandomUniform<x64::avx512_core>::bicubicInterpolation(const Vmm& vWCoord, const Vmm& vHCoord, bool tail) {
//     auto vHTop      = getVmm();
//     auto vWLeft     = getVmm();
//     auto vDX        = getVmm();
//     auto vDY        = getVmm();
//     auto vXDotProd  = getVmm();
//     auto& vYDotProd = vDX;
//     auto vSrcShift0 = getVmm();
//     auto vSrcShift  = getVmm();
//     auto vAux       = getVmm();
//     auto kAuxMask   = getMask();
//     RegistersPool::Reg<Vmask> kMaskH;
//     std::vector<RegistersPool::Reg<Vmask>> wMasks;

//     uni_vroundps(vHTop, vHCoord, 0x1);  // Round floor
//     uni_vroundps(vWLeft, vWCoord, 0x1); // Round floor
//     uni_vsubps(vDY, vHCoord, vHTop);
//     uni_vsubps(vDX, vWCoord, vWLeft);
//     uni_vsubps(vHTop, vHTop, vOnesF);
//     uni_vsubps(vWLeft, vWLeft, vOnesF);

//     RegistersPool::Reg<Vmm> vCX[4] = {getVmm(), getVmm(), getVmm(), getVmm() };
//     for (int i = 0; i < 4; i++) {
//         bicubicCoefficients(vCX[i], vDX, i);
//     }

//     bool useMask = false, zeroFill = false;
//     if (jcp.paddingMode == GridSamplePaddingMode::ZEROS) {
//         useMask = zeroFill = true;
//         wMasks.resize(4);
//         for (auto& mask : wMasks) {
//             mask = getMask();
//         }
//         zerosPaddingW(wMasks[0], vWLeft);
//         uni_vaddps(vWCoord, vWLeft, vOnesF);
//         zerosPaddingW(wMasks[1], vWCoord);
//         uni_vaddps(vWCoord, vWCoord, vOnesF);
//         zerosPaddingW(wMasks[2], vWCoord);
//         uni_vaddps(vWCoord, vWCoord, vOnesF);
//         zerosPaddingW(wMasks[3], vWCoord);
//         kMaskH = getMask();
//     }

//     // PER CHANNEL LOOP
//     Xbyak::Label lChannelLoopBegin, lChannelLoopEnd;
//     RegistersPool::Reg<Xbyak::Reg64> rChannel;
//     auto rSrcTmp  = getReg64();
//     auto rDstTmp  = getReg64();
//     mov(rSrcTmp, regSrc);
//     mov(rDstTmp, r64_dst);

//     for (size_t ch = 0; ch < jcp.cannelNum; ch++) {
//         if (jcp.dynamicChannel) {
//             rChannel = getReg64();
//             mov(rChannel, 0);

//             L(lChannelLoopBegin);
//             cmp(rChannel, regChannelNum);
//             jge(lChannelLoopEnd, T_NEAR);
//         }

//         uni_vmovups(vHCoord, vHTop);
//         uni_vpxor(vYDotProd, vYDotProd, vYDotProd);
//         for (int h = 0; h < 4; h++) {
//             // (y - 1 + h; x - 1)
//             if (jcp.paddingMode == GridSamplePaddingMode::ZEROS) {
//                 Xbyak::Opmask maskH = kMaskH;
//                 vcmpps(kMaskH, vHCoord, vSrcHeightF, CMP_LT_PS);
//                 vcmpps(maskH | maskH, vZeros, vHCoord, CMP_LE_PS);
//                 kandw(kAuxMask, kMaskH, wMasks[0]);
//                 uni_vmulps(vSrcShift0, vHCoord, vSrcWidthF);
//                 uni_vmovups(vWCoord, vWLeft);
//                 uni_vaddps(vSrcShift, vSrcShift0, vWCoord);
//             } else if (jcp.paddingMode == GridSamplePaddingMode::BORDER) {
//                 borderPadding(vSrcShift0, vHCoord, coord::h);
//                 uni_vmulps(vSrcShift0, vSrcShift0, vSrcWidthF);
//                 uni_vmovups(vWCoord, vWLeft);
//                 borderPadding(vSrcShift, vWCoord, coord::w);
//                 uni_vaddps(vSrcShift, vSrcShift0, vSrcShift);
//             } else if (jcp.paddingMode == GridSamplePaddingMode::REFLECTION) {
//                 reflectionPadding(vSrcShift0, vHCoord, coord::h);
//                 uni_vmulps(vSrcShift0, vSrcShift0, vSrcWidthF);
//                 uni_vmovups(vWCoord, vWLeft);
//                 reflectionPadding(vSrcShift, vWCoord, coord::w);
//                 uni_vaddps(vSrcShift, vSrcShift0, vSrcShift);
//             }
//             uni_vcvtps2dq(vSrcShift, vSrcShift);
//             if (dataTypeSize > 1)
//                 uni_vpslld(vSrcShift, vSrcShift, dataTypeShift);
//             gatherdd(vAux, rSrcTmp, vSrcShift, kAuxMask, useMask, zeroFill);
//             if (jcp.inDataPrc == InferenceEngine::Precision::I32) {
//                 uni_vcvtdq2ps(vAux, vAux);
//             }
//             uni_vmulps(vXDotProd, vAux, vCX[0]);

//             // (y - 1 + h; x)
//             // (y - 1 + h; x + 1)
//             // (y - 1 + h; x + 2)
//             for (int w = 1; w < 4; w++) {
//                 uni_vaddps(vWCoord, vWCoord, vOnesF);
//                 if (jcp.paddingMode == GridSamplePaddingMode::ZEROS) {
//                     uni_vaddps(vSrcShift, vSrcShift0, vWCoord);
//                     kandw(kAuxMask, kMaskH, wMasks[w]);
//                 } else if (jcp.paddingMode == GridSamplePaddingMode::BORDER) {
//                     borderPadding(vSrcShift, vWCoord, coord::w);
//                     uni_vaddps(vSrcShift, vSrcShift0, vSrcShift);
//                 } else if (jcp.paddingMode == GridSamplePaddingMode::REFLECTION) {
//                     reflectionPadding(vSrcShift, vWCoord, coord::w);
//                     uni_vaddps(vSrcShift, vSrcShift0, vSrcShift);
//                 }
//                 uni_vcvtps2dq(vSrcShift, vSrcShift);
//                 if (dataTypeSize > 1)
//                     uni_vpslld(vSrcShift, vSrcShift, dataTypeShift);
//                 gatherdd(vAux, rSrcTmp, vSrcShift, kAuxMask, useMask, zeroFill);
//                 if (jcp.inDataPrc == InferenceEngine::Precision::I32) {
//                     uni_vcvtdq2ps(vAux, vAux);
//                 }
//                 uni_vfmadd231ps(vXDotProd, vAux, vCX[w]);
//             }

//             if (h != 3) {
//                 uni_vaddps(vHCoord, vHCoord, vOnesF);
//             }

//             bicubicCoefficients(vAux, vDY, h);
//             uni_vfmadd231ps(vYDotProd, vXDotProd, vAux);
//         }

//         if (jcp.inDataPrc == InferenceEngine::Precision::I32) {
//             uni_vroundps(vYDotProd, vYDotProd, 0x3); // Truncation
//             uni_vcvtps2dq(vYDotProd, vYDotProd);
//         }

//         if (!tail) {
//             uni_vmovups(ptr[rDstTmp], vYDotProd);
//         } else {
//             uni_vmovups(ptr[rDstTmp] | kTailMask, vYDotProd);
//         }
//         add(rSrcTmp, regSrcChannelStepB);
//         add(rDstTmp, regDstChannelStepB);

//         if (jcp.dynamicChannel) {
//             inc(rChannel);
//             jmp(lChannelLoopBegin, T_NEAR);
//             L(lChannelLoopEnd);
//         }
//     }
// }

// template <x64::cpu_isa_t isa> // Works for AVX2, AVX, SSE41
// void RandomUniform<isa>::bicubicInterpolation(const Vmm& vWCoord, const Vmm& vHCoord, bool tail) {
//     auto vHTop  = getVmm();
//     auto vWLeft = getVmm();
//     auto vDX    = getVmm();
//     auto vDY    = getVmm();

//     uni_vroundps(vHTop,  vHCoord, 0x1); // Round floor
//     uni_vroundps(vWLeft, vWCoord, 0x1); // Round floor
//     uni_vsubps(vDY, vHCoord, vHTop);
//     uni_vsubps(vDX, vWCoord, vWLeft);
//     uni_vsubps(vHTop, vHTop, vOnesF);
//     uni_vsubps(vWLeft, vWLeft, vOnesF);

//     auto rBuff = getReg64();
//     mov(rBuff, ptr[regParams + GET_OFF(buffer)]);

//     bool useMask = false, zeroFill = false;

//     if (jcp.paddingMode == GridSamplePaddingMode::BORDER) {
//         auto rAux = getReg64();

//         if (!vSrcWidthSub1F.isInitialized()) {
//             vSrcWidthSub1F = getVmm();
//             mov(rAux, ptr[regParams + GET_OFF(srcWidthSub1F)]);
//             uni_vmovups(vSrcWidthSub1F, ptr[rAux]);
//         }

//         auto vW0 = getVmm(), vW1 = getVmm();
//         Vmm vW[4] = { vW0, vW1, vHCoord, vWCoord };
//         for (int w = 0; w < 4; w++) {
//             borderPadding(vW[w], vWLeft, coord::w);
//             if (w < 3) {
//                 uni_vaddps(vWLeft, vWLeft, vOnesF);
//             }
//         }
//         vWLeft.release();
//         vSrcWidthSub1F.release();

//         if (!vSrcHeightSub1F.isInitialized()) {
//             vSrcHeightSub1F = getVmm();
//             mov(rAux, ptr[regParams + GET_OFF(srcHeightSub1F)]);
//             uni_vmovups(vSrcHeightSub1F, ptr[rAux]);
//         }
//         auto vH  = getVmm();

//         size_t bufShift = 0lu;
//         for (int h = 0; h < 4; h++) {
//             borderPadding(vH, vHTop, coord::h);
//             uni_vmulps(vH, vH, vSrcWidthF);
//             auto vShift = getVmm();
//             for (int w = 0; w < 4; w++) {
//                 uni_vaddps(vShift, vH, vW[w]);
//                 dataTypeShiftPs2Dq(vShift, vShift);
//                 uni_vmovups(ptr[rBuff + bufShift], vShift);
//                 bufShift += vlen;
//             }
//             if (h < 3) {
//                 uni_vaddps(vHTop, vHTop, vOnesF);
//             }
//         }
//         vHTop.release();
//         vSrcHeightSub1F.release();
//     } else if (jcp.paddingMode == GridSamplePaddingMode::REFLECTION) {
//         auto rAux = getReg64();
//         if (!jcp.alignCorners && !vSrcWidthMul2F.isInitialized()) {
//             vSrcWidthMul2F = getVmm();
//             mov(rAux, ptr[regParams + GET_OFF(srcWidthMul2F)]);
//             uni_vmovups(vSrcWidthMul2F, ptr[rAux]);
//         }
//         if (!vSrcWidthMul2Sub1F.isInitialized()) {
//             vSrcWidthMul2Sub1F = getVmm();
//             mov(rAux, ptr[regParams + GET_OFF(srcWidthMul2Sub1F)]);
//             uni_vmovups(vSrcWidthMul2Sub1F, ptr[rAux]);
//         }

//         auto vW0 = getVmm(), vW1 = getVmm();
//         Vmm vW[4] = { vW0, vW1, vHCoord, vWCoord };
//         for (int w = 0; w < 4; w++) {
//             reflectionPadding(vW[w], vWLeft, coord::w);
//             if (w < 3) {
//                 uni_vaddps(vWLeft, vWLeft, vOnesF);
//             }
//         }
//         vWLeft.release();
//         vSrcWidthMul2F.release();
//         vSrcWidthMul2Sub1F.release();

//         if (!jcp.alignCorners && !vSrcHeightMul2F.isInitialized()) {
//             vSrcHeightMul2F = getVmm();
//             mov(rAux, ptr[regParams + GET_OFF(srcHeightMul2F)]);
//             uni_vmovups(vSrcHeightMul2F, ptr[rAux]);
//         }
//         if (!vSrcHeightMul2Sub1F.isInitialized()) {
//             vSrcHeightMul2Sub1F = getVmm();
//             mov(rAux, ptr[regParams + GET_OFF(srcHeightMul2Sub1F)]);
//             uni_vmovups(vSrcHeightMul2Sub1F, ptr[rAux]);
//         }
//         auto vH  = getVmm();

//         size_t bufShift = 0lu;
//         for (int h = 0; h < 4; h++) {
//             reflectionPadding(vH, vHTop, coord::h);
//             uni_vmulps(vH, vH, vSrcWidthF);
//             auto vShift = getVmm();
//             for (int w = 0; w < 4; w++) {
//                 uni_vaddps(vShift, vH, vW[w]);
//                 dataTypeShiftPs2Dq(vShift, vShift);
//                 uni_vmovups(ptr[rBuff + bufShift], vShift);
//                 bufShift += vlen;
//             }
//             if (h < 3) {
//                 uni_vaddps(vHTop, vHTop, vOnesF);
//             }
//         }
//         vHTop.release();
//         vSrcHeightMul2F.release();
//         vSrcHeightMul2Sub1F.release();
//     } else if (jcp.paddingMode == GridSamplePaddingMode::ZEROS) {
//         useMask = zeroFill = true;

//         RegistersPool::Reg<Vmm> vWMask[4] = { getVmm(), getVmm(), getVmm(), getVmm() };
//         for (int w = 0; w < 4; w++) {
//             if (w == 0) {
//                 zerosPaddingW(vWMask[w], vWLeft);
//                 uni_vaddps(vWCoord, vWLeft, vOnesF);
//             } else {
//                 zerosPaddingW(vWMask[w], vWCoord);
//                 if (w < 3) {
//                     uni_vaddps(vWCoord, vWCoord, vOnesF);
//                 }
//             }
//         }

//         size_t bufShift = 0lu;
//         auto vShift = vWCoord, vMaskH = vHCoord;
//         if (!vDataTypeSizeB.isInitialized()) {
//             auto rAux = getReg64();
//             vDataTypeSizeB = getVmm();
//             mov(rAux, ptr[regParams + GET_OFF(dataTypeSize)]);
//             uni_vmovups(vDataTypeSizeB, ptr[rAux]);
//         }

//         for (int h = 0; h < 4; h++) {
//             if (isa == x64::avx2) {
//                 uni_vmovups(vShift, vHTop);
//                 uni_vfmadd132ps(vShift, vWLeft, vSrcWidthF);
//             } else {
//                 uni_vmulps(vShift, vHTop, vSrcWidthF);
//                 uni_vaddps(vShift, vShift, vWLeft);
//             }
//             dataTypeShiftPs2Dq(vShift, vShift);
//             for (int w = 0; w < 4; w++) {
//                 uni_vmovups(ptr[rBuff + bufShift], vShift);
//                 if (w < 3) {
//                     uni_vpaddd(vShift, vShift, vDataTypeSizeB);
//                 }

//                 zerosPaddingH(vMaskH, vHTop, vWMask[w]);
//                 uni_vmovups(ptr[rBuff + bufShift + 16 * vlen], vMaskH);
//                 bufShift += vlen;
//             }
//             if (h < 3) {
//                 uni_vaddps(vHTop, vHTop, vOnesF);
//             }
//         }
//         vHTop.release();
//         vWLeft.release();
//         vDataTypeSizeB.release();
//     }

//     RegistersPool::Reg<Vmm> vCX[4] = { getVmm(), getVmm(), getVmm(), getVmm() };
//     for (int w = 0; w < 4; w++) {
//         bicubicCoefficients(vCX[w], vDX, w);
//     }
//     auto vCY0 = getVmm(), vCY1 = getVmm();
//     Vmm vCY[4] = { vCY0, vCY1, vHCoord, vWCoord };
//     for (int h = 0; h < 4; h++) {
//         bicubicCoefficients(vCY[h], vDY, h);
//     }

//     const auto& vXDotProd = vDX;
//     const auto& vYDotProd = vDY;
//     auto vSrcShift   = getVmm();
//     auto kGatherMask = getVmm();
//     auto vAux        = getVmm();

//     // PER CHANNEL LOOP
//     Xbyak::Label lChannelLoopBegin, lChannelLoopEnd;
//     RegistersPool::Reg<Xbyak::Reg64> rChannel;
//     auto rSrcTmp = getReg64();
//     auto rDstTmp = getReg64();
//     mov(rSrcTmp, regSrc);
//     mov(rDstTmp, r64_dst);

//     for (uint64_t ch = 0; ch < jcp.cannelNum; ch++) {
//         if (jcp.dynamicChannel) {
//             rChannel = getReg64();
//             mov(rChannel, ptr[regParams + GET_OFF(channelsNum)]);

//             L(lChannelLoopBegin);
//             cmp(rChannel, 0);
//             jle(lChannelLoopEnd, T_NEAR);
//         }

//         uni_vpxor(vYDotProd, vYDotProd, vYDotProd);
//         for (int h = 0; h < 4; h++) {
//             size_t bufShift = h * 4 * vlen;
//             // (y - 1 + h; x - 1)
//             if (jcp.paddingMode == GridSamplePaddingMode::ZEROS) {
//                 uni_vmovups(kGatherMask, ptr[rBuff + bufShift + 16 * vlen]);
//             }
//             uni_vmovups(vSrcShift, ptr[rBuff + bufShift]);
//             bufShift += vlen;

//             gatherdd(vAux, rSrcTmp, vSrcShift, kGatherMask, useMask, zeroFill);
//             if (jcp.inDataPrc == InferenceEngine::Precision::I32) {
//                 uni_vcvtdq2ps(vAux, vAux);
//             }
//             uni_vmulps(vXDotProd, vAux, vCX[0]);

//             // (y - 1 + h; x)
//             // (y - 1 + h; x + 1)
//             // (y - 1 + h; x + 2)
//             for (int w = 1; w < 4; w++) {
//                 if (jcp.paddingMode == GridSamplePaddingMode::ZEROS) {
//                     uni_vmovups(kGatherMask, ptr[rBuff + bufShift + 16 * vlen]);
//                 }
//                 uni_vmovups(vSrcShift, ptr[rBuff + bufShift]);
//                 bufShift += vlen;

//                 gatherdd(vAux, rSrcTmp, vSrcShift, kGatherMask, useMask, zeroFill);
//                 if (jcp.inDataPrc == InferenceEngine::Precision::I32) {
//                     uni_vcvtdq2ps(vAux, vAux);
//                 }
//                 uni_vfmadd231ps(vXDotProd, vAux, vCX[w]);
//             }
//             uni_vfmadd231ps(vYDotProd, vXDotProd, vCY[h]);
//         }

//         if (jcp.inDataPrc == InferenceEngine::Precision::I32) {
//             uni_vroundps(vYDotProd, vYDotProd, 0x3); // Truncation
//             uni_vcvtps2dq(vYDotProd, vYDotProd);
//         }

//         if (!tail) {
//             uni_vmovups(ptr[rDstTmp], vYDotProd);
//         } else {
//             store(ptr[rDstTmp], vYDotProd, r64_work_amount, dataTypeSize);
//         }
//         add(rSrcTmp, regSrcChannelStepB);
//         add(rDstTmp, regDstChannelStepB);

//         if (jcp.dynamicChannel) {
//             dec(rChannel);
//             jmp(lChannelLoopBegin, T_NEAR);
//             L(lChannelLoopEnd);
//         }
//     }
// }

// template <x64::cpu_isa_t isa>
// void RandomUniform<isa>::dataTypeShiftPs2Dq(const Vmm& vDst, const Vmm& vSrc) {
//     if (dataTypeSize == 1)
//         return;

//     if (isa == x64::avx) { // vpslld works just with XMM for AVX, so use vmulps for YMM
//         auto rAux = getReg64();
//         static const float val = dataTypeSize;
//         static const float dataTypeSizeArr[8] = {val, val, val, val, val, val, val, val};
//         mov(rAux, reinterpret_cast<uintptr_t>(dataTypeSizeArr));
//         uni_vmulps(vDst, vSrc, ptr[rAux]);
//         uni_vcvtps2dq(vDst, vDst);
//     } else {
//         uni_vcvtps2dq(vDst, vSrc);
//         if (dataTypeSize > 1)
//             uni_vpslld(vDst, vDst, dataTypeShift); // multiply by source data type size.
//     }
// }

// template <x64::cpu_isa_t isa>
// void RandomUniform<isa>::hwShiftPs2dq(const Vmm& vDst, const Vmm& vHCoord, const Vmm& vWCoord, const Vmm& vWidth) {
//     if (vDst.getIdx() == vWCoord.getIdx()) {
//         if (one_of(isa, x64::avx512_core, x64::avx2)) {
//             uni_vfmadd231ps(vDst, vHCoord, vWidth);
//         } else {
//             auto vTmp = getVmm();
//             uni_vmulps(vTmp, vHCoord, vWidth);
//             uni_vaddps(vDst, vWCoord, vTmp);
//         }
//     } else if (vDst.getIdx() == vHCoord.getIdx()) {
//         uni_vfmadd132ps(vDst, vWCoord, vWidth);
//     } else if (vDst.getIdx() == vWidth.getIdx()) {
//         uni_vfmadd132ps(vDst, vWCoord, vHCoord);
//     } else {
//         if (one_of(isa, x64::avx2, x64::avx512_core)) {
//             uni_vmovups(vDst, vWCoord);
//             uni_vfmadd231ps(vDst, vHCoord, vWidth);
//         } else {
//             uni_vmulps(vDst, vHCoord, vWidth);
//             uni_vaddps(vDst, vDst, vWCoord);
//         }
//     }

//     if (isa == x64::avx) { // vpslld works just with XMM for AVX, so use vmulps for YMM
//         if (dataTypeSize > 1) {
//             auto rAux = getReg64();
//             const float val = dataTypeSize;
//             static const float dataTypeSizeArr[8] = {val, val, val, val, val, val, val, val};
//             mov(rAux, reinterpret_cast<uintptr_t>(dataTypeSizeArr));
//             uni_vmulps(vDst, vDst, ptr[rAux]);
//         }
//         uni_vcvtps2dq(vDst, vDst);
//     } else {
//         uni_vcvtps2dq(vDst, vDst);
//         if (dataTypeSize > 1)
//             uni_vpslld(vDst, vDst, dataTypeShift); // multiply by source data type size.
//     }
// }

template class RandomUniform<x64::avx512_core>;
template class RandomUniform<x64::avx2>;
template class RandomUniform<x64::sse41>;

}   // namespace kernel
}   // namespace intel_cpu
}   // namespace ov
