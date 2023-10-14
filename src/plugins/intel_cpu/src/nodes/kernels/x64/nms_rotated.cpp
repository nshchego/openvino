// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "nms_rotated.hpp"

using namespace dnnl::impl::cpu;

namespace ov {
namespace intel_cpu {
namespace kernel {

#define GET_OFF(field) offsetof(NmsRotatedCallArgs, field)

template <x64::cpu_isa_t isa>
NmsRotated<isa>::NmsRotated(const NmsRotatedCompileParams& jcp) :
        JitKernel(jit_name(), jcp, isa) {
}

template <x64::cpu_isa_t isa>
void NmsRotated<isa>::generate() {
    OPENVINO_THROW("Not ready");
    this->preamble();
    registersPool = RegistersPool::create(isa, {rax, rcx, rsp, rdi, k0});

    r64_dst = getReg64();
    r64_boxes = getReg64();
    r64_work_amount = getReg64();

    mov(r64_dst, ptr[regParams + GET_OFF(dst_ptr)]);
    mov(r64_boxes, ptr[regParams + GET_OFF(boxes_ptr)]);
    mov(r64_work_amount, ptr[regParams + GET_OFF(work_amount)]);

    initVectors();
    process();

    registersPool.reset();
    this->postamble();
}

template <>
void NmsRotated<x64::avx512_core>::initVectors() {
    const auto r64_aux = getReg64();
    const auto r32_aux = Xbyak::Reg32(r64_aux.getIdx());
    const auto r16_aux = Xbyak::Reg16(r64_aux.getIdx());

    v_max_mul_n_64 = getVmm();
    v_max_mul_c_64 = getVmm();
    v_add_low_k    = getVmm();
    v_add_up_k     = getVmm();
    v_convert_0    = getVmm();
    v_convert_1    = getVmm();
    v_n_inc        = getVmm();
    v_range        = getVmm();
    v_min          = getVmm();
    v_key_64       = getVmm();
    v_counter_64   = getVmm();
    v_n_64         = getVmm();
    v_res_perm     = getVmm();

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
        vpbroadcastd(v_range, ptr[r64_aux]);

        mov(r64_aux, ptr[regParams + GET_OFF(min_ptr)]);
        vpbroadcastd(v_min, ptr[r64_aux]);

        uni_vsubps(v_range, v_range, v_min);
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
        const auto ymm_range = Xbyak::Ymm(v_range.getIdx());

        mov(r64_aux, ptr[regParams + GET_OFF(max_ptr)]);
        vpbroadcastd(v_range, ptr[r64_aux]);

        mov(r64_aux, ptr[regParams + GET_OFF(min_ptr)]);
        vpbroadcastd(v_min, ptr[r64_aux]);

        uni_vpsubd(v_range, v_range, v_min);
        uni_vcvtdq2pd(v_range, ymm_range);
    } else {
        OPENVINO_THROW("NmsRotated kernel does not support precision ", m_jcp.out_data_type, " for ", x64::get_isa_info());
    }

    // Initialize inputs.
    // mov(r64_aux, ptr[regParams + GET_OFF(key_ptr)]);
    // vpbroadcastq(v_key_64, ptr[r64_aux]);

    // mov(r64_aux, ptr[regParams + GET_OFF(counter_ptr)]);
    // vpbroadcastq(v_counter_64, ptr[r64_aux]);

    mov(r64_aux, ptr[regParams + GET_OFF(n_ptr)]);
    vpbroadcastq(v_n_64, ptr[r64_aux]);
    if (m_jcp.out_data_type.size() <= 4) {
        static const uint64_t n_inc_arr[8]  = { 0, 1, 2, 3, 4, 5, 6, 7 };
        mov(r64_aux, reinterpret_cast<uintptr_t>(n_inc_arr));
    } else {
        static const uint64_t n_inc_arr[8]  = { 0, 1, 2, 3, 4, 5, 6, 7 }; // TODO: i64
        mov(r64_aux, reinterpret_cast<uintptr_t>(n_inc_arr));
    }
    uni_vpaddq(v_n_64, v_n_64, ptr[r64_aux]);

    // Initialize auxiliary vectors.
    static const uint32_t res_perm_mask[16] = { 0b00000000, 0b00010000, 0b00001000, 0b00011000, 0b00000010, 0b00010010, 0b00001010, 0b00011010,
                                                0b00000100, 0b00010100, 0b00001100, 0b00011100, 0b00000110, 0b00010110, 0b00001110, 0b00011110 };
    mov(r64_aux, reinterpret_cast<uintptr_t>(res_perm_mask));
    uni_vmovups(v_res_perm, ptr[r64_aux]);
}

template <x64::cpu_isa_t isa> // Works for AVX2, SSE41
void NmsRotated<isa>::initVectors() {
    const auto r64_aux = getReg64();
    const auto r16_aux = Xbyak::Reg16(r64_aux.getIdx());

    v_max_mul_n_64 = getVmm();
    v_max_mul_c_64 = getVmm();
    v_add_low_k    = getVmm();
    v_add_up_k     = getVmm();
    v_range        = getVmm();
    v_key_64       = getVmm();
    v_counter_64   = getVmm();
    v_n_64         = getVmm();

    r64_n_inc      = getReg64();
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
    INIT_ARR(max_mul_n_64, 0xd2511f53, r64_aux, uint64_t);
    uni_vmovups(v_max_mul_n_64, ptr[r64_aux]);

    INIT_ARR(max_mul_c_64, 0xcd9e8d57, r64_aux, uint64_t);
    uni_vmovups(v_max_mul_c_64, ptr[r64_aux]);

    INIT_ARR(add_low_k, 0x9e3779b9, r64_aux, uint32_t);
    uni_vmovups(v_add_low_k, ptr[r64_aux]);

    INIT_ARR(add_up_k, 0xbb67ae85, r64_aux, uint32_t);
    uni_vmovups(v_add_up_k, ptr[r64_aux]);

    INIT_ARR(n_inc_step, isa == x64::avx2 ? 4 : 2, r64_n_inc, uint64_t);

    if (m_jcp.out_data_type == element::f32) {
        r64_convert_0  = getReg64();
        r64_convert_1  = getReg64();

        INIT_ARR(convert_0, 0x3f800000, r64_convert_0, uint32_t);
        INIT_ARR(convert_1, 0x007fffff, r64_convert_1, uint32_t);

        mov(r64_aux, ptr[regParams + GET_OFF(max_ptr)]);
        uni_vpbroadcastd(v_range, ptr[r64_aux]);

        auto v_aux = getVmm();
        mov(r64_aux, ptr[regParams + GET_OFF(min_ptr)]);
        uni_vpbroadcastd(v_aux, ptr[r64_aux]);
        static uint32_t min_arr[8];
        mov(r64_min, reinterpret_cast<uintptr_t>(min_arr));
        uni_vmovups(ptr[r64_min], v_aux);

        uni_vsubps(v_range, v_range, v_aux);
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
        r64_f64_pow_52 = getReg64();
        const auto v_aux = getVmm();
        const auto xmm_range = Xbyak::Xmm(v_range.getIdx());

        INIT_ARR(f64_pow_52, 0x4330000000000000, r64_f64_pow_52, uint64_t);

        mov(r64_aux, ptr[regParams + GET_OFF(max_ptr)]);
        uni_vpbroadcastd(v_range, ptr[r64_aux]);

        mov(r64_aux, ptr[regParams + GET_OFF(min_ptr)]);
        uni_vpbroadcastd(v_aux, ptr[r64_aux]);
        static uint32_t min_arr[8];
        mov(r64_min, reinterpret_cast<uintptr_t>(min_arr));
        uni_vmovups(ptr[r64_min], v_aux);

        uni_vpsubd(v_range, v_range, v_aux);
        uni_vcvtdq2pd(v_range, xmm_range);
    } else {
        OPENVINO_THROW("NmsRotated kernel does not support precision ", m_jcp.out_data_type, " for ", x64::get_isa_info());
    }

    // Initialize inputs.
    // mov(r64_aux, ptr[regParams + GET_OFF(key_ptr)]);
    // uni_vpbroadcastq(v_key_64, ptr[r64_aux]);

    // mov(r64_aux, ptr[regParams + GET_OFF(counter_ptr)]);
    // uni_vpbroadcastq(v_counter_64, ptr[r64_aux]);

    mov(r64_aux, ptr[regParams + GET_OFF(n_ptr)]);
    uni_vpbroadcastq(v_n_64, ptr[r64_aux]);

    if (m_jcp.out_data_type.size() <= 4) {
        if (isa == x64::avx2) {
            static const uint64_t n_inc_arr[4]  = { 0, 1, 2, 3 };
            mov(r64_aux, reinterpret_cast<uintptr_t>(n_inc_arr));
        } else {
            static uint64_t n_inc_arr[4];
            static uint64_t* n_inc_arr_aligned = n_inc_arr + (reinterpret_cast<int64_t>(n_inc_arr) % 16) / sizeof(uint64_t);
            n_inc_arr_aligned[0] = 0;
            n_inc_arr_aligned[1] = 1;
            mov(r64_aux, reinterpret_cast<uintptr_t>(n_inc_arr_aligned));
        }
    } else {
        static const uint64_t n_inc_arr[4]  = { 0, 1, 2, 3 }; // TODO: i64
        mov(r64_aux, reinterpret_cast<uintptr_t>(n_inc_arr));
    }

    uni_vpaddq(v_n_64, v_n_64, ptr[r64_aux]);

#undef INIT_ARR
}

template <x64::cpu_isa_t isa>
void NmsRotated<isa>::process() {
    auto v_dst_0 = getVmm();
    auto v_dst_1 = getVmm();
    std::vector<Vmm> v_res{ v_dst_0, v_dst_1 };

    auto step = vlen;
    if (one_of(m_jcp.out_data_type.size(), 2lu, 4lu)) {
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
        if (one_of(m_jcp.out_data_type.size(), 4lu, 8lu)) {
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
void NmsRotated<isa>::getRotatedVertices(const Vmm& vmm_boxes, const Vmm& vmm_vertices) {
    auto v_aux = getVmm();
    auto ymm_aux = Xbyak::Ymm(v_aux.getIdx());
    uni_vpbroadcastq(ymm_aux, ptr[r64_boxes]);
}

template <x64::cpu_isa_t isa>
void NmsRotated<isa>::calculateRound(const Vmm& vmm_k_0, const Vmm& vmm_k_1, const Vmm& vmm_c_0, const Vmm& vmm_c_1,
                                        const Vmm& vmm_n_0, const Vmm& vmm_n_1, const Vmm& vmm_aux_0, const Vmm& vmm_aux_1) {
    uni_vpmuludq(vmm_aux_0, vmm_n_0, v_max_mul_n_64);  // {p0,p1,p0,p1} = {n0,_,n0,_} * {m0,_,m0,_}
    uni_vpmuludq(vmm_aux_1, vmm_c_0, v_max_mul_c_64);  // {r0,r1,r0,r1} = {c0,_,c0,_} * {m0,_,m0,_}

    uni_vpshufd(vmm_c_0, vmm_aux_0, 0b10110001);       // {p1,p0,p1,p0} = shuf {p0,p1,p0,p1}
    uni_vxorps(vmm_c_0, vmm_c_0, vmm_c_1);             // {c0,_,c0,_} = {p1,_,p1,_} ^ {c1,_,c1,_}
    uni_vxorps(vmm_c_0, vmm_c_0, vmm_k_1);             // {c0,_,c0,_} = {c0,_,c0,_} ^ {k1,_,k1,_}

    uni_vpshufd(vmm_n_0, vmm_aux_1, 0b10110001);       // {r1,r0,r1,r0} = shuf {r0,r1,r0,r1}
    uni_vxorps(vmm_n_0, vmm_n_0, vmm_n_1);             // {n0,_,n0,_} = {r1,_,r1,_} ^ {n1,_,n1,_}
    uni_vxorps(vmm_n_0, vmm_n_0, vmm_k_0);             // {n0,_,n0,_} = {n0,_,n0,_} ^ {k0,_,k0,_}
}

template <x64::cpu_isa_t isa>
void NmsRotated<isa>::runPhilox(const std::vector<Vmm>& vmm_dst, const Vmm& vmm_key, const Vmm& vmm_counter, const Vmm& vmm_n) {
    auto vmm_k_0 = getVmm();
    auto vmm_k_1 = getVmm();
    auto vmm_n_0 = getVmm();
    auto vmm_n_1 = vmm_dst[0];
    auto vmm_c_0 = getVmm();
    auto vmm_c_1 = getVmm();
    auto vmm_aux_0 = getVmm();
    auto vmm_aux_1 = vmm_dst[1];

    uni_vmovups(vmm_k_0, vmm_key);                        // {k0,k1,k0,k1} -> {k0,_,k0,_}
    uni_vpshufd(vmm_k_1, vmm_key, 0b10110001);            // {k0,k1,k0,k1} -> {k1,_,k1,_}

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
        vpermt2d(vmm_n_0, v_res_perm, vmm_n_1);             // {n0,n1,n0,n1} = perm {n0,_,n0,_} {n1,_,n1,_}
        vpermt2d(vmm_c_0, v_res_perm, vmm_c_1);             // {c0,c1,c0,c1} = perm {c0,_,c0,_} {c1,_,c1,_}
        vshufpd(vmm_dst[0], vmm_n_0, vmm_c_0, 0b00000000);  // {n0,n1,c0,c1} = shuf {n0,n1,n0,n1} {c0,c1,c0,c1}
        vshufpd(vmm_dst[1], vmm_n_0, vmm_c_0, 0b11111111);  // {n0,n1,c0,c1} = shuf {n0,n1,n0,n1} {c0,c1,c0,c1}
    } else if (isa == x64::avx2) {
        auto ymm_dst_0 = Xbyak::Ymm(vmm_dst[0].getIdx());
        auto ymm_dst_1 = Xbyak::Ymm(vmm_dst[1].getIdx());
        auto ymm_c_0 = Xbyak::Ymm(vmm_c_0.getIdx());

        uni_vshufps(vmm_n_0, vmm_n_0, vmm_n_1, 0b10001000);   // {n0,n0,n1,n1} = shuf {n0,_,n0,_} {n1,_,n1,_}
        uni_vshufps(vmm_c_0, vmm_c_0, vmm_c_1, 0b10001000);   // {c0,c0,c1,c1} = shuf {c0,_,c0,_} {c1,_,c1,_}
        uni_vshufps(ymm_dst_1, vmm_n_0, vmm_c_0, 0b10001000); // {n0,n1,c0,c1} = shuf {n0,n0,n1,n1} {c0,c0,c1,c1}
        uni_vshufps(vmm_c_0, vmm_n_0, vmm_c_0, 0b11011101);   // {n0,n1,c0,c1} = shuf {n0,n0,n1,n1} {c0,c0,c1,c1}
        vperm2f128(ymm_dst_0, ymm_dst_1, ymm_c_0, 0b00100000);
        vperm2f128(ymm_dst_1, ymm_dst_1, ymm_c_0, 0b00110001);
    } else {
        uni_vshufps(vmm_n_0, vmm_n_0, vmm_n_1, 0b10001000);
        uni_vshufps(vmm_c_0, vmm_c_0, vmm_c_1, 0b10001000);
        uni_vshufps(vmm_dst[0], vmm_n_0, vmm_c_0, 0b10001000);
        uni_vshufps(vmm_dst[1], vmm_n_0, vmm_c_0, 0b11011101);
    }
}

template <x64::cpu_isa_t isa>
void NmsRotated<isa>::raiseKey(const Vmm& vmm_k_0, const Vmm& vmm_k_1) {
    uni_vpaddd(vmm_k_0, vmm_k_0, v_add_low_k); // {k0,_,k0,_} + {l0,_,l0,_}
    uni_vpaddd(vmm_k_1, vmm_k_1, v_add_up_k);  // {k1,_,k1,_} + {u0,_,u0,_}
}

template <>
void NmsRotated<x64::avx512_core>::convert(const std::vector<Vmm>& v_dst, const std::vector<Vmm>& v_src) {
    if (m_jcp.out_data_type.size() == 4) {
        for (size_t i = 0lu; i < v_src.size(); i++) {
            auto vmm_src = v_src[i];
            auto vmm_dst = v_dst[i];

            if (m_jcp.out_data_type == element::f32) {
                uni_vandps(vmm_dst, vmm_src, v_convert_1);
                uni_vorps(vmm_dst, vmm_dst, v_convert_0);
                uni_vsubps(vmm_dst, vmm_dst, v_convert_0);
                vfmadd132ps(vmm_dst, v_min, v_range);
            } else if (m_jcp.out_data_type == element::i32) {
                // x % (max - min) + min
                const auto v_aux_0 = getVmm();
                const auto v_aux_1 = getVmm();
                const auto ymm_src = Xbyak::Ymm(vmm_src.getIdx());
                const auto ymm_dst = Xbyak::Ymm(vmm_dst.getIdx());
                const auto ymm_aux_1 = Xbyak::Ymm(v_aux_1.getIdx());

                // Divide in the f64 due to the f32 loses accuracy here.
                vcvtudq2pd(v_aux_0, ymm_src);
                uni_vdivpd(v_aux_1, v_aux_0, v_range);
                uni_vroundpd(v_aux_1, v_aux_1, 3);
                vfnmadd132pd(v_aux_1, v_aux_0, v_range);

                vextractf64x4(ymm_dst, vmm_src, 1);
                vcvtudq2pd(v_aux_0, ymm_dst);
                uni_vcvtpd2dq(ymm_dst, v_aux_1);
                uni_vdivpd(v_aux_1, v_aux_0, v_range);
                uni_vroundpd(v_aux_1, v_aux_1, 3);
                vfnmadd132pd(v_aux_1, v_aux_0, v_range);
                uni_vcvtpd2dq(ymm_aux_1, v_aux_1);
                vshuff64x2(vmm_dst, vmm_dst, v_aux_1, 0b01000100);

                uni_vpaddd(vmm_dst, vmm_dst, v_min);
            } else {
                OPENVINO_THROW("NmsRotated kernel does not support precision ", m_jcp.out_data_type, " for ", x64::get_isa_info());
            }
        }
    } else if (m_jcp.out_data_type.size() == 2) {
        if (m_jcp.out_data_type == element::f16 && x64::mayiuse(x64::avx512_core_fp16)) {
            auto ymm_dst_0 = Xbyak::Ymm(v_dst[0].getIdx());
            auto ymm_dst_1 = Xbyak::Ymm(v_dst[1].getIdx());

            vpmovusdw(ymm_dst_0, v_dst[0]);
            vpmovusdw(ymm_dst_1, v_dst[1]);
            vshuff64x2(v_dst[0], v_dst[0], v_dst[1], 0b01000100);

            uni_vandps(v_dst[0], v_dst[0], v_convert_1);
            uni_vorps(v_dst[0], v_dst[0], v_convert_0);
            vsubph(v_dst[0], v_dst[0], v_convert_0);
            vfmadd132ph(v_dst[0], v_min, v_range);
        } else if (m_jcp.out_data_type == element::bf16 && x64::mayiuse(x64::avx512_core_bf16)) {
            auto ymm_convert_0 = Xbyak::Ymm(v_convert_0.getIdx());
            auto ymm_convert_1 = Xbyak::Ymm(v_convert_1.getIdx());

            for (const auto& vmm_dst : v_dst) {
                auto ymm_dst = Xbyak::Ymm(vmm_dst.getIdx());

                vpmovusdw(ymm_dst, vmm_dst);
                uni_vandps(ymm_dst, ymm_dst, ymm_convert_1);
                uni_vorps(ymm_dst, ymm_dst, ymm_convert_0);

                vpmovzxwd(vmm_dst, ymm_dst);
                uni_vpslld(vmm_dst, vmm_dst, 16);

                uni_vsubps(vmm_dst, vmm_dst, v_convert_0);
                vfmadd132ps(vmm_dst, v_min, v_range);
            }

            vcvtne2ps2bf16(v_dst[0], v_dst[0], v_dst[1]);
        } else {
            OPENVINO_THROW("NmsRotated kernel does not support precision ", m_jcp.out_data_type, " for ", x64::get_isa_info());
        }
    } else if (m_jcp.out_data_type.size() == 8) {
        if (m_jcp.out_data_type == element::i64) {
            // TODO: in scope of i64 enabling.
        }
        OPENVINO_THROW("NmsRotated kernel does not support precision ", m_jcp.out_data_type, " for ", x64::get_isa_info());
    } else {
        OPENVINO_THROW("NmsRotated kernel does not support precision ", m_jcp.out_data_type, " for ", x64::get_isa_info());
    }
}

template <x64::cpu_isa_t isa> // Works for AVX2, SSE41
void NmsRotated<isa>::convert(const std::vector<Vmm>& v_dst, const std::vector<Vmm>& v_src) {
    if (m_jcp.out_data_type.size() == 4) {
        for (size_t i = 0lu; i < v_src.size(); i++) {
            auto vmm_src = v_src[i];
            auto vmm_dst = v_dst[i];

            if (m_jcp.out_data_type == element::f32) {
                uni_vandps(vmm_dst, vmm_src, ptr[r64_convert_1]);
                uni_vorps(vmm_dst, vmm_dst, ptr[r64_convert_0]);
                uni_vsubps(vmm_dst, vmm_dst, ptr[r64_convert_0]);
                if (isa == x64::avx2) {
                    vfmadd213ps(vmm_dst, v_range, ptr[r64_min]);
                } else {
                    uni_vmulps(vmm_dst, vmm_dst, v_range);
                    uni_vaddps(vmm_dst, vmm_dst, ptr[r64_min]);
                }
            } else if (m_jcp.out_data_type == element::i32) {
                // x % (max - min) + min
                const auto v_aux_0 = getVmm();
                const auto v_aux_1 = getVmm();
                const auto xmm_dst = Xbyak::Xmm(vmm_dst.getIdx());
                const auto ymm_dst = Xbyak::Ymm(vmm_dst.getIdx());
                const auto xmm_aux_1 = Xbyak::Xmm(v_aux_1.getIdx());

                // Convert u32->f64. TODO: move to convert emitter after i64 enabling.
                uni_vpmovzxdq(v_aux_0, xmm_dst);
                uni_vorps(v_aux_0, v_aux_0, ptr[r64_f64_pow_52]);
                uni_vsubpd(v_aux_0, v_aux_0, ptr[r64_f64_pow_52]);

                // Divide in the f64 due to the f32 loses accuracy here.
                uni_vdivpd(v_aux_1, v_aux_0, v_range);
                uni_vroundpd(v_aux_1, v_aux_1, 3);
                if (isa == x64::avx2) {
                    vfnmadd132pd(v_aux_1, v_aux_0, v_range);
                } else {
                    uni_vmulpd(v_aux_1, v_aux_1, v_range);
                    uni_vsubpd(v_aux_0, v_aux_0, v_aux_1);
                    uni_vmovups(v_aux_1, v_aux_0);
                }

                if (isa == x64::avx2) {
                    vperm2f128(ymm_dst, ymm_dst, ymm_dst, 0b00000001);
                } else {
                    uni_vshufpd(vmm_dst, vmm_dst, vmm_dst, 0b00000001);
                }
                // Convert u32->f64. TODO: move to convert emitter after i64 enabling.
                uni_vpmovzxdq(v_aux_0, xmm_dst);
                uni_vorps(v_aux_0, v_aux_0, ptr[r64_f64_pow_52]);
                uni_vsubpd(v_aux_0, v_aux_0, ptr[r64_f64_pow_52]);

                uni_vcvtpd2dq(xmm_dst, v_aux_1);
                uni_vdivpd(v_aux_1, v_aux_0, v_range);
                uni_vroundpd(v_aux_1, v_aux_1, 3);
                if (isa == x64::avx2) {
                    vfnmadd132pd(v_aux_1, v_aux_0, v_range);
                } else {
                    uni_vmulpd(v_aux_1, v_aux_1, v_range);
                    uni_vsubpd(v_aux_0, v_aux_0, v_aux_1);
                    uni_vmovups(v_aux_1, v_aux_0);
                }
                uni_vcvtpd2dq(xmm_aux_1, v_aux_1);
                if (isa == x64::avx2) {
                    vperm2f128(ymm_dst, ymm_dst, v_aux_1, 0b00100000);
                } else {
                    uni_vshufpd(vmm_dst, vmm_dst, v_aux_1, 0b00000000);
                }

                uni_vpaddd(vmm_dst, vmm_dst, ptr[r64_min]);
            } else {
                OPENVINO_THROW("NmsRotated kernel does not support precision ", m_jcp.out_data_type, " for ", x64::get_isa_info());
            }
        }
    } else if (m_jcp.out_data_type.size() == 8) {
        if (m_jcp.out_data_type == element::i64) {
            // TODO: in scope of i64 enabling.
        }
        OPENVINO_THROW("NmsRotated kernel does not support precision ", m_jcp.out_data_type, " for ", x64::get_isa_info());
    } else {
        OPENVINO_THROW("NmsRotated kernel does not support precision ", m_jcp.out_data_type, " for ", x64::get_isa_info());
    }
}

template <>
void NmsRotated<x64::avx512_core>::tail(const std::vector<Vmm>& vmm_dst) {
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

template <>
void NmsRotated<x64::avx2>::tail(const std::vector<Vmm>& vmm_dst) {
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

template <x64::cpu_isa_t isa>
void NmsRotated<isa>::tail(const std::vector<Vmm>& vmm_dst) {
    Xbyak::Label l_0, l_end;
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
    store(ptr[r64_dst], vmm_dst[1], r64_work_amount, m_jcp.out_data_type.size());
    jmp(l_end, T_NEAR);

    L(l_0);
    store(ptr[r64_dst], vmm_dst[0], r64_work_amount, m_jcp.out_data_type.size());

    L(l_end);
}

template class NmsRotated<x64::avx512_core>;
template class NmsRotated<x64::avx2>;
template class NmsRotated<x64::sse41>;

}   // namespace kernel
}   // namespace intel_cpu
}   // namespace ov
