; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc < %s -mtriple=i386-linux-generic -verify-machineinstrs -mattr=sse | FileCheck %s --check-prefix=X86 --check-prefix=X86-SSE  --check-prefix=X86-SSE1
; RUN: llc < %s -mtriple=i386-linux-generic -verify-machineinstrs -mattr=sse2 | FileCheck %s --check-prefix=X86 --check-prefix=X86-SSE  --check-prefix=X86-SSE2
; RUN: llc < %s -mtriple=i386-linux-generic -verify-machineinstrs -mattr=avx | FileCheck %s --check-prefix=X86 --check-prefix=X86-AVX --check-prefix=X86-AVX1
; RUN: llc < %s -mtriple=i386-linux-generic -verify-machineinstrs -mattr=avx512f | FileCheck %s --check-prefix=X86 --check-prefix=X86-AVX --check-prefix=X86-AVX512
; RUN: llc < %s -mtriple=i386-linux-generic -verify-machineinstrs | FileCheck %s --check-prefix=X86 --check-prefix=X86-NOSSE
; RUN: llc < %s -mtriple=x86_64-linux-generic -verify-machineinstrs -mattr=sse2 | FileCheck %s --check-prefix=X64 --check-prefix=X64-SSE
; RUN: llc < %s -mtriple=x86_64-linux-generic -verify-machineinstrs -mattr=avx | FileCheck %s --check-prefix=X64 --check-prefix=X64-AVX  --check-prefix=X64-AVX1
; RUN: llc < %s -mtriple=x86_64-linux-generic -verify-machineinstrs -mattr=avx512f | FileCheck %s --check-prefix=X64 --check-prefix=X64-AVX  --check-prefix=X64-AVX512

; Note: This test is testing that the lowering for atomics matches what we
; currently emit for non-atomics + the atomic restriction.  The presence of
; particular lowering detail in these tests should not be read as requiring
; that detail for correctness unless it's related to the atomicity itself.
; (Specifically, there were reviewer questions about the lowering for halfs
;  and their calling convention which remain unresolved.)

define void @store_half(half* %fptr, half %v) {
; X86-LABEL: store_half:
; X86:       # %bb.0:
; X86-NEXT:    movzwl {{[0-9]+}}(%esp), %eax
; X86-NEXT:    movl {{[0-9]+}}(%esp), %ecx
; X86-NEXT:    movw %ax, (%ecx)
; X86-NEXT:    retl
;
; X64-LABEL: store_half:
; X64:       # %bb.0:
; X64-NEXT:    movw %si, (%rdi)
; X64-NEXT:    retq
  store atomic half %v, half* %fptr unordered, align 2
  ret void
}

define void @store_float(float* %fptr, float %v) {
; X86-LABEL: store_float:
; X86:       # %bb.0:
; X86-NEXT:    movl {{[0-9]+}}(%esp), %eax
; X86-NEXT:    movl {{[0-9]+}}(%esp), %ecx
; X86-NEXT:    movl %ecx, (%eax)
; X86-NEXT:    retl
;
; X64-SSE-LABEL: store_float:
; X64-SSE:       # %bb.0:
; X64-SSE-NEXT:    movss %xmm0, (%rdi)
; X64-SSE-NEXT:    retq
;
; X64-AVX-LABEL: store_float:
; X64-AVX:       # %bb.0:
; X64-AVX-NEXT:    vmovss %xmm0, (%rdi)
; X64-AVX-NEXT:    retq
  store atomic float %v, float* %fptr unordered, align 4
  ret void
}

define void @store_double(double* %fptr, double %v) {
; X86-SSE1-LABEL: store_double:
; X86-SSE1:       # %bb.0:
; X86-SSE1-NEXT:    pushl %ebx
; X86-SSE1-NEXT:    .cfi_def_cfa_offset 8
; X86-SSE1-NEXT:    pushl %esi
; X86-SSE1-NEXT:    .cfi_def_cfa_offset 12
; X86-SSE1-NEXT:    .cfi_offset %esi, -12
; X86-SSE1-NEXT:    .cfi_offset %ebx, -8
; X86-SSE1-NEXT:    movl {{[0-9]+}}(%esp), %esi
; X86-SSE1-NEXT:    movl {{[0-9]+}}(%esp), %ebx
; X86-SSE1-NEXT:    movl {{[0-9]+}}(%esp), %ecx
; X86-SSE1-NEXT:    movl (%esi), %eax
; X86-SSE1-NEXT:    movl 4(%esi), %edx
; X86-SSE1-NEXT:    .p2align 4, 0x90
; X86-SSE1-NEXT:  .LBB2_1: # %atomicrmw.start
; X86-SSE1-NEXT:    # =>This Inner Loop Header: Depth=1
; X86-SSE1-NEXT:    lock cmpxchg8b (%esi)
; X86-SSE1-NEXT:    jne .LBB2_1
; X86-SSE1-NEXT:  # %bb.2: # %atomicrmw.end
; X86-SSE1-NEXT:    popl %esi
; X86-SSE1-NEXT:    .cfi_def_cfa_offset 8
; X86-SSE1-NEXT:    popl %ebx
; X86-SSE1-NEXT:    .cfi_def_cfa_offset 4
; X86-SSE1-NEXT:    retl
;
; X86-SSE2-LABEL: store_double:
; X86-SSE2:       # %bb.0:
; X86-SSE2-NEXT:    movl {{[0-9]+}}(%esp), %eax
; X86-SSE2-NEXT:    movsd {{.*#+}} xmm0 = mem[0],zero
; X86-SSE2-NEXT:    movlps %xmm0, (%eax)
; X86-SSE2-NEXT:    retl
;
; X86-AVX-LABEL: store_double:
; X86-AVX:       # %bb.0:
; X86-AVX-NEXT:    movl {{[0-9]+}}(%esp), %eax
; X86-AVX-NEXT:    vmovsd {{.*#+}} xmm0 = mem[0],zero
; X86-AVX-NEXT:    vmovlps %xmm0, (%eax)
; X86-AVX-NEXT:    retl
;
; X86-NOSSE-LABEL: store_double:
; X86-NOSSE:       # %bb.0:
; X86-NOSSE-NEXT:    pushl %ebx
; X86-NOSSE-NEXT:    .cfi_def_cfa_offset 8
; X86-NOSSE-NEXT:    pushl %esi
; X86-NOSSE-NEXT:    .cfi_def_cfa_offset 12
; X86-NOSSE-NEXT:    .cfi_offset %esi, -12
; X86-NOSSE-NEXT:    .cfi_offset %ebx, -8
; X86-NOSSE-NEXT:    movl {{[0-9]+}}(%esp), %esi
; X86-NOSSE-NEXT:    movl {{[0-9]+}}(%esp), %ebx
; X86-NOSSE-NEXT:    movl {{[0-9]+}}(%esp), %ecx
; X86-NOSSE-NEXT:    movl (%esi), %eax
; X86-NOSSE-NEXT:    movl 4(%esi), %edx
; X86-NOSSE-NEXT:    .p2align 4, 0x90
; X86-NOSSE-NEXT:  .LBB2_1: # %atomicrmw.start
; X86-NOSSE-NEXT:    # =>This Inner Loop Header: Depth=1
; X86-NOSSE-NEXT:    lock cmpxchg8b (%esi)
; X86-NOSSE-NEXT:    jne .LBB2_1
; X86-NOSSE-NEXT:  # %bb.2: # %atomicrmw.end
; X86-NOSSE-NEXT:    popl %esi
; X86-NOSSE-NEXT:    .cfi_def_cfa_offset 8
; X86-NOSSE-NEXT:    popl %ebx
; X86-NOSSE-NEXT:    .cfi_def_cfa_offset 4
; X86-NOSSE-NEXT:    retl
;
; X64-SSE-LABEL: store_double:
; X64-SSE:       # %bb.0:
; X64-SSE-NEXT:    movsd %xmm0, (%rdi)
; X64-SSE-NEXT:    retq
;
; X64-AVX-LABEL: store_double:
; X64-AVX:       # %bb.0:
; X64-AVX-NEXT:    vmovsd %xmm0, (%rdi)
; X64-AVX-NEXT:    retq
  store atomic double %v, double* %fptr unordered, align 8
  ret void
}

define void @store_fp128(fp128* %fptr, fp128 %v) {
; X86-SSE-LABEL: store_fp128:
; X86-SSE:       # %bb.0:
; X86-SSE-NEXT:    subl $36, %esp
; X86-SSE-NEXT:    .cfi_adjust_cfa_offset 36
; X86-SSE-NEXT:    leal {{[0-9]+}}(%esp), %eax
; X86-SSE-NEXT:    pushl {{[0-9]+}}(%esp)
; X86-SSE-NEXT:    .cfi_adjust_cfa_offset 4
; X86-SSE-NEXT:    pushl {{[0-9]+}}(%esp)
; X86-SSE-NEXT:    .cfi_adjust_cfa_offset 4
; X86-SSE-NEXT:    pushl {{[0-9]+}}(%esp)
; X86-SSE-NEXT:    .cfi_adjust_cfa_offset 4
; X86-SSE-NEXT:    pushl {{[0-9]+}}(%esp)
; X86-SSE-NEXT:    .cfi_adjust_cfa_offset 4
; X86-SSE-NEXT:    pushl {{[0-9]+}}(%esp)
; X86-SSE-NEXT:    .cfi_adjust_cfa_offset 4
; X86-SSE-NEXT:    pushl %eax
; X86-SSE-NEXT:    .cfi_adjust_cfa_offset 4
; X86-SSE-NEXT:    calll __sync_lock_test_and_set_16
; X86-SSE-NEXT:    .cfi_adjust_cfa_offset -4
; X86-SSE-NEXT:    addl $56, %esp
; X86-SSE-NEXT:    .cfi_adjust_cfa_offset -56
; X86-SSE-NEXT:    retl
;
; X86-AVX-LABEL: store_fp128:
; X86-AVX:       # %bb.0:
; X86-AVX-NEXT:    subl $44, %esp
; X86-AVX-NEXT:    .cfi_def_cfa_offset 48
; X86-AVX-NEXT:    vmovaps {{[0-9]+}}(%esp), %xmm0
; X86-AVX-NEXT:    movl {{[0-9]+}}(%esp), %eax
; X86-AVX-NEXT:    movl %eax, {{[0-9]+}}(%esp)
; X86-AVX-NEXT:    vmovups %xmm0, {{[0-9]+}}(%esp)
; X86-AVX-NEXT:    leal {{[0-9]+}}(%esp), %eax
; X86-AVX-NEXT:    movl %eax, (%esp)
; X86-AVX-NEXT:    calll __sync_lock_test_and_set_16
; X86-AVX-NEXT:    addl $40, %esp
; X86-AVX-NEXT:    .cfi_def_cfa_offset 4
; X86-AVX-NEXT:    retl
;
; X86-NOSSE-LABEL: store_fp128:
; X86-NOSSE:       # %bb.0:
; X86-NOSSE-NEXT:    subl $36, %esp
; X86-NOSSE-NEXT:    .cfi_adjust_cfa_offset 36
; X86-NOSSE-NEXT:    leal {{[0-9]+}}(%esp), %eax
; X86-NOSSE-NEXT:    pushl {{[0-9]+}}(%esp)
; X86-NOSSE-NEXT:    .cfi_adjust_cfa_offset 4
; X86-NOSSE-NEXT:    pushl {{[0-9]+}}(%esp)
; X86-NOSSE-NEXT:    .cfi_adjust_cfa_offset 4
; X86-NOSSE-NEXT:    pushl {{[0-9]+}}(%esp)
; X86-NOSSE-NEXT:    .cfi_adjust_cfa_offset 4
; X86-NOSSE-NEXT:    pushl {{[0-9]+}}(%esp)
; X86-NOSSE-NEXT:    .cfi_adjust_cfa_offset 4
; X86-NOSSE-NEXT:    pushl {{[0-9]+}}(%esp)
; X86-NOSSE-NEXT:    .cfi_adjust_cfa_offset 4
; X86-NOSSE-NEXT:    pushl %eax
; X86-NOSSE-NEXT:    .cfi_adjust_cfa_offset 4
; X86-NOSSE-NEXT:    calll __sync_lock_test_and_set_16
; X86-NOSSE-NEXT:    .cfi_adjust_cfa_offset -4
; X86-NOSSE-NEXT:    addl $56, %esp
; X86-NOSSE-NEXT:    .cfi_adjust_cfa_offset -56
; X86-NOSSE-NEXT:    retl
;
; X64-SSE-LABEL: store_fp128:
; X64-SSE:       # %bb.0:
; X64-SSE-NEXT:    subq $24, %rsp
; X64-SSE-NEXT:    .cfi_def_cfa_offset 32
; X64-SSE-NEXT:    movaps %xmm0, (%rsp)
; X64-SSE-NEXT:    movq (%rsp), %rsi
; X64-SSE-NEXT:    movq {{[0-9]+}}(%rsp), %rdx
; X64-SSE-NEXT:    callq __sync_lock_test_and_set_16
; X64-SSE-NEXT:    addq $24, %rsp
; X64-SSE-NEXT:    .cfi_def_cfa_offset 8
; X64-SSE-NEXT:    retq
;
; X64-AVX-LABEL: store_fp128:
; X64-AVX:       # %bb.0:
; X64-AVX-NEXT:    subq $24, %rsp
; X64-AVX-NEXT:    .cfi_def_cfa_offset 32
; X64-AVX-NEXT:    vmovaps %xmm0, (%rsp)
; X64-AVX-NEXT:    movq (%rsp), %rsi
; X64-AVX-NEXT:    movq {{[0-9]+}}(%rsp), %rdx
; X64-AVX-NEXT:    callq __sync_lock_test_and_set_16
; X64-AVX-NEXT:    addq $24, %rsp
; X64-AVX-NEXT:    .cfi_def_cfa_offset 8
; X64-AVX-NEXT:    retq
  store atomic fp128 %v, fp128* %fptr unordered, align 16
  ret void
}

define half @load_half(half* %fptr) {
; X86-LABEL: load_half:
; X86:       # %bb.0:
; X86-NEXT:    movl {{[0-9]+}}(%esp), %eax
; X86-NEXT:    movzwl (%eax), %eax
; X86-NEXT:    retl
;
; X64-LABEL: load_half:
; X64:       # %bb.0:
; X64-NEXT:    movzwl (%rdi), %eax
; X64-NEXT:    retq
  %v = load atomic half, half* %fptr unordered, align 2
  ret half %v
}

define float @load_float(float* %fptr) {
; X86-SSE1-LABEL: load_float:
; X86-SSE1:       # %bb.0:
; X86-SSE1-NEXT:    pushl %eax
; X86-SSE1-NEXT:    .cfi_def_cfa_offset 8
; X86-SSE1-NEXT:    movl {{[0-9]+}}(%esp), %eax
; X86-SSE1-NEXT:    movl (%eax), %eax
; X86-SSE1-NEXT:    movl %eax, (%esp)
; X86-SSE1-NEXT:    flds (%esp)
; X86-SSE1-NEXT:    popl %eax
; X86-SSE1-NEXT:    .cfi_def_cfa_offset 4
; X86-SSE1-NEXT:    retl
;
; X86-SSE2-LABEL: load_float:
; X86-SSE2:       # %bb.0:
; X86-SSE2-NEXT:    pushl %eax
; X86-SSE2-NEXT:    .cfi_def_cfa_offset 8
; X86-SSE2-NEXT:    movl {{[0-9]+}}(%esp), %eax
; X86-SSE2-NEXT:    movss {{.*#+}} xmm0 = mem[0],zero,zero,zero
; X86-SSE2-NEXT:    movss %xmm0, (%esp)
; X86-SSE2-NEXT:    flds (%esp)
; X86-SSE2-NEXT:    popl %eax
; X86-SSE2-NEXT:    .cfi_def_cfa_offset 4
; X86-SSE2-NEXT:    retl
;
; X86-AVX-LABEL: load_float:
; X86-AVX:       # %bb.0:
; X86-AVX-NEXT:    pushl %eax
; X86-AVX-NEXT:    .cfi_def_cfa_offset 8
; X86-AVX-NEXT:    movl {{[0-9]+}}(%esp), %eax
; X86-AVX-NEXT:    vmovss {{.*#+}} xmm0 = mem[0],zero,zero,zero
; X86-AVX-NEXT:    vmovss %xmm0, (%esp)
; X86-AVX-NEXT:    flds (%esp)
; X86-AVX-NEXT:    popl %eax
; X86-AVX-NEXT:    .cfi_def_cfa_offset 4
; X86-AVX-NEXT:    retl
;
; X86-NOSSE-LABEL: load_float:
; X86-NOSSE:       # %bb.0:
; X86-NOSSE-NEXT:    pushl %eax
; X86-NOSSE-NEXT:    .cfi_def_cfa_offset 8
; X86-NOSSE-NEXT:    movl {{[0-9]+}}(%esp), %eax
; X86-NOSSE-NEXT:    movl (%eax), %eax
; X86-NOSSE-NEXT:    movl %eax, (%esp)
; X86-NOSSE-NEXT:    flds (%esp)
; X86-NOSSE-NEXT:    popl %eax
; X86-NOSSE-NEXT:    .cfi_def_cfa_offset 4
; X86-NOSSE-NEXT:    retl
;
; X64-SSE-LABEL: load_float:
; X64-SSE:       # %bb.0:
; X64-SSE-NEXT:    movss {{.*#+}} xmm0 = mem[0],zero,zero,zero
; X64-SSE-NEXT:    retq
;
; X64-AVX-LABEL: load_float:
; X64-AVX:       # %bb.0:
; X64-AVX-NEXT:    vmovss {{.*#+}} xmm0 = mem[0],zero,zero,zero
; X64-AVX-NEXT:    retq
  %v = load atomic float, float* %fptr unordered, align 4
  ret float %v
}

define double @load_double(double* %fptr) {
; X86-SSE1-LABEL: load_double:
; X86-SSE1:       # %bb.0:
; X86-SSE1-NEXT:    subl $20, %esp
; X86-SSE1-NEXT:    .cfi_def_cfa_offset 24
; X86-SSE1-NEXT:    movl {{[0-9]+}}(%esp), %eax
; X86-SSE1-NEXT:    fildll (%eax)
; X86-SSE1-NEXT:    fistpll {{[0-9]+}}(%esp)
; X86-SSE1-NEXT:    movl {{[0-9]+}}(%esp), %eax
; X86-SSE1-NEXT:    movl {{[0-9]+}}(%esp), %ecx
; X86-SSE1-NEXT:    movl %ecx, {{[0-9]+}}(%esp)
; X86-SSE1-NEXT:    movl %eax, (%esp)
; X86-SSE1-NEXT:    fldl (%esp)
; X86-SSE1-NEXT:    addl $20, %esp
; X86-SSE1-NEXT:    .cfi_def_cfa_offset 4
; X86-SSE1-NEXT:    retl
;
; X86-SSE2-LABEL: load_double:
; X86-SSE2:       # %bb.0:
; X86-SSE2-NEXT:    subl $12, %esp
; X86-SSE2-NEXT:    .cfi_def_cfa_offset 16
; X86-SSE2-NEXT:    movl {{[0-9]+}}(%esp), %eax
; X86-SSE2-NEXT:    movsd {{.*#+}} xmm0 = mem[0],zero
; X86-SSE2-NEXT:    movlps %xmm0, (%esp)
; X86-SSE2-NEXT:    fldl (%esp)
; X86-SSE2-NEXT:    addl $12, %esp
; X86-SSE2-NEXT:    .cfi_def_cfa_offset 4
; X86-SSE2-NEXT:    retl
;
; X86-AVX-LABEL: load_double:
; X86-AVX:       # %bb.0:
; X86-AVX-NEXT:    subl $12, %esp
; X86-AVX-NEXT:    .cfi_def_cfa_offset 16
; X86-AVX-NEXT:    movl {{[0-9]+}}(%esp), %eax
; X86-AVX-NEXT:    vmovsd {{.*#+}} xmm0 = mem[0],zero
; X86-AVX-NEXT:    vmovlps %xmm0, (%esp)
; X86-AVX-NEXT:    fldl (%esp)
; X86-AVX-NEXT:    addl $12, %esp
; X86-AVX-NEXT:    .cfi_def_cfa_offset 4
; X86-AVX-NEXT:    retl
;
; X86-NOSSE-LABEL: load_double:
; X86-NOSSE:       # %bb.0:
; X86-NOSSE-NEXT:    subl $20, %esp
; X86-NOSSE-NEXT:    .cfi_def_cfa_offset 24
; X86-NOSSE-NEXT:    movl {{[0-9]+}}(%esp), %eax
; X86-NOSSE-NEXT:    fildll (%eax)
; X86-NOSSE-NEXT:    fistpll {{[0-9]+}}(%esp)
; X86-NOSSE-NEXT:    movl {{[0-9]+}}(%esp), %eax
; X86-NOSSE-NEXT:    movl {{[0-9]+}}(%esp), %ecx
; X86-NOSSE-NEXT:    movl %ecx, {{[0-9]+}}(%esp)
; X86-NOSSE-NEXT:    movl %eax, (%esp)
; X86-NOSSE-NEXT:    fldl (%esp)
; X86-NOSSE-NEXT:    addl $20, %esp
; X86-NOSSE-NEXT:    .cfi_def_cfa_offset 4
; X86-NOSSE-NEXT:    retl
;
; X64-SSE-LABEL: load_double:
; X64-SSE:       # %bb.0:
; X64-SSE-NEXT:    movsd {{.*#+}} xmm0 = mem[0],zero
; X64-SSE-NEXT:    retq
;
; X64-AVX-LABEL: load_double:
; X64-AVX:       # %bb.0:
; X64-AVX-NEXT:    vmovsd {{.*#+}} xmm0 = mem[0],zero
; X64-AVX-NEXT:    retq
  %v = load atomic double, double* %fptr unordered, align 8
  ret double %v
}

define fp128 @load_fp128(fp128* %fptr) {
; X86-SSE-LABEL: load_fp128:
; X86-SSE:       # %bb.0:
; X86-SSE-NEXT:    pushl %edi
; X86-SSE-NEXT:    .cfi_def_cfa_offset 8
; X86-SSE-NEXT:    pushl %esi
; X86-SSE-NEXT:    .cfi_def_cfa_offset 12
; X86-SSE-NEXT:    subl $20, %esp
; X86-SSE-NEXT:    .cfi_def_cfa_offset 32
; X86-SSE-NEXT:    .cfi_offset %esi, -12
; X86-SSE-NEXT:    .cfi_offset %edi, -8
; X86-SSE-NEXT:    movl {{[0-9]+}}(%esp), %esi
; X86-SSE-NEXT:    subl $8, %esp
; X86-SSE-NEXT:    .cfi_adjust_cfa_offset 8
; X86-SSE-NEXT:    leal {{[0-9]+}}(%esp), %eax
; X86-SSE-NEXT:    pushl $0
; X86-SSE-NEXT:    .cfi_adjust_cfa_offset 4
; X86-SSE-NEXT:    pushl $0
; X86-SSE-NEXT:    .cfi_adjust_cfa_offset 4
; X86-SSE-NEXT:    pushl $0
; X86-SSE-NEXT:    .cfi_adjust_cfa_offset 4
; X86-SSE-NEXT:    pushl $0
; X86-SSE-NEXT:    .cfi_adjust_cfa_offset 4
; X86-SSE-NEXT:    pushl $0
; X86-SSE-NEXT:    .cfi_adjust_cfa_offset 4
; X86-SSE-NEXT:    pushl $0
; X86-SSE-NEXT:    .cfi_adjust_cfa_offset 4
; X86-SSE-NEXT:    pushl $0
; X86-SSE-NEXT:    .cfi_adjust_cfa_offset 4
; X86-SSE-NEXT:    pushl $0
; X86-SSE-NEXT:    .cfi_adjust_cfa_offset 4
; X86-SSE-NEXT:    pushl {{[0-9]+}}(%esp)
; X86-SSE-NEXT:    .cfi_adjust_cfa_offset 4
; X86-SSE-NEXT:    pushl %eax
; X86-SSE-NEXT:    .cfi_adjust_cfa_offset 4
; X86-SSE-NEXT:    calll __sync_val_compare_and_swap_16
; X86-SSE-NEXT:    .cfi_adjust_cfa_offset -4
; X86-SSE-NEXT:    addl $44, %esp
; X86-SSE-NEXT:    .cfi_adjust_cfa_offset -44
; X86-SSE-NEXT:    movl (%esp), %eax
; X86-SSE-NEXT:    movl {{[0-9]+}}(%esp), %ecx
; X86-SSE-NEXT:    movl {{[0-9]+}}(%esp), %edx
; X86-SSE-NEXT:    movl {{[0-9]+}}(%esp), %edi
; X86-SSE-NEXT:    movl %edi, 8(%esi)
; X86-SSE-NEXT:    movl %edx, 12(%esi)
; X86-SSE-NEXT:    movl %eax, (%esi)
; X86-SSE-NEXT:    movl %ecx, 4(%esi)
; X86-SSE-NEXT:    movl %esi, %eax
; X86-SSE-NEXT:    addl $20, %esp
; X86-SSE-NEXT:    .cfi_def_cfa_offset 12
; X86-SSE-NEXT:    popl %esi
; X86-SSE-NEXT:    .cfi_def_cfa_offset 8
; X86-SSE-NEXT:    popl %edi
; X86-SSE-NEXT:    .cfi_def_cfa_offset 4
; X86-SSE-NEXT:    retl $4
;
; X86-AVX-LABEL: load_fp128:
; X86-AVX:       # %bb.0:
; X86-AVX-NEXT:    pushl %esi
; X86-AVX-NEXT:    .cfi_def_cfa_offset 8
; X86-AVX-NEXT:    subl $56, %esp
; X86-AVX-NEXT:    .cfi_def_cfa_offset 64
; X86-AVX-NEXT:    .cfi_offset %esi, -8
; X86-AVX-NEXT:    movl {{[0-9]+}}(%esp), %esi
; X86-AVX-NEXT:    movl {{[0-9]+}}(%esp), %eax
; X86-AVX-NEXT:    vxorps %xmm0, %xmm0, %xmm0
; X86-AVX-NEXT:    vmovups %ymm0, {{[0-9]+}}(%esp)
; X86-AVX-NEXT:    movl %eax, {{[0-9]+}}(%esp)
; X86-AVX-NEXT:    leal {{[0-9]+}}(%esp), %eax
; X86-AVX-NEXT:    movl %eax, (%esp)
; X86-AVX-NEXT:    vzeroupper
; X86-AVX-NEXT:    calll __sync_val_compare_and_swap_16
; X86-AVX-NEXT:    subl $4, %esp
; X86-AVX-NEXT:    vmovups {{[0-9]+}}(%esp), %xmm0
; X86-AVX-NEXT:    vmovaps %xmm0, (%esi)
; X86-AVX-NEXT:    movl %esi, %eax
; X86-AVX-NEXT:    addl $56, %esp
; X86-AVX-NEXT:    .cfi_def_cfa_offset 8
; X86-AVX-NEXT:    popl %esi
; X86-AVX-NEXT:    .cfi_def_cfa_offset 4
; X86-AVX-NEXT:    retl $4
;
; X86-NOSSE-LABEL: load_fp128:
; X86-NOSSE:       # %bb.0:
; X86-NOSSE-NEXT:    pushl %edi
; X86-NOSSE-NEXT:    .cfi_def_cfa_offset 8
; X86-NOSSE-NEXT:    pushl %esi
; X86-NOSSE-NEXT:    .cfi_def_cfa_offset 12
; X86-NOSSE-NEXT:    subl $20, %esp
; X86-NOSSE-NEXT:    .cfi_def_cfa_offset 32
; X86-NOSSE-NEXT:    .cfi_offset %esi, -12
; X86-NOSSE-NEXT:    .cfi_offset %edi, -8
; X86-NOSSE-NEXT:    movl {{[0-9]+}}(%esp), %esi
; X86-NOSSE-NEXT:    subl $8, %esp
; X86-NOSSE-NEXT:    .cfi_adjust_cfa_offset 8
; X86-NOSSE-NEXT:    leal {{[0-9]+}}(%esp), %eax
; X86-NOSSE-NEXT:    pushl $0
; X86-NOSSE-NEXT:    .cfi_adjust_cfa_offset 4
; X86-NOSSE-NEXT:    pushl $0
; X86-NOSSE-NEXT:    .cfi_adjust_cfa_offset 4
; X86-NOSSE-NEXT:    pushl $0
; X86-NOSSE-NEXT:    .cfi_adjust_cfa_offset 4
; X86-NOSSE-NEXT:    pushl $0
; X86-NOSSE-NEXT:    .cfi_adjust_cfa_offset 4
; X86-NOSSE-NEXT:    pushl $0
; X86-NOSSE-NEXT:    .cfi_adjust_cfa_offset 4
; X86-NOSSE-NEXT:    pushl $0
; X86-NOSSE-NEXT:    .cfi_adjust_cfa_offset 4
; X86-NOSSE-NEXT:    pushl $0
; X86-NOSSE-NEXT:    .cfi_adjust_cfa_offset 4
; X86-NOSSE-NEXT:    pushl $0
; X86-NOSSE-NEXT:    .cfi_adjust_cfa_offset 4
; X86-NOSSE-NEXT:    pushl {{[0-9]+}}(%esp)
; X86-NOSSE-NEXT:    .cfi_adjust_cfa_offset 4
; X86-NOSSE-NEXT:    pushl %eax
; X86-NOSSE-NEXT:    .cfi_adjust_cfa_offset 4
; X86-NOSSE-NEXT:    calll __sync_val_compare_and_swap_16
; X86-NOSSE-NEXT:    .cfi_adjust_cfa_offset -4
; X86-NOSSE-NEXT:    addl $44, %esp
; X86-NOSSE-NEXT:    .cfi_adjust_cfa_offset -44
; X86-NOSSE-NEXT:    movl (%esp), %eax
; X86-NOSSE-NEXT:    movl {{[0-9]+}}(%esp), %ecx
; X86-NOSSE-NEXT:    movl {{[0-9]+}}(%esp), %edx
; X86-NOSSE-NEXT:    movl {{[0-9]+}}(%esp), %edi
; X86-NOSSE-NEXT:    movl %edi, 8(%esi)
; X86-NOSSE-NEXT:    movl %edx, 12(%esi)
; X86-NOSSE-NEXT:    movl %eax, (%esi)
; X86-NOSSE-NEXT:    movl %ecx, 4(%esi)
; X86-NOSSE-NEXT:    movl %esi, %eax
; X86-NOSSE-NEXT:    addl $20, %esp
; X86-NOSSE-NEXT:    .cfi_def_cfa_offset 12
; X86-NOSSE-NEXT:    popl %esi
; X86-NOSSE-NEXT:    .cfi_def_cfa_offset 8
; X86-NOSSE-NEXT:    popl %edi
; X86-NOSSE-NEXT:    .cfi_def_cfa_offset 4
; X86-NOSSE-NEXT:    retl $4
;
; X64-SSE-LABEL: load_fp128:
; X64-SSE:       # %bb.0:
; X64-SSE-NEXT:    subq $24, %rsp
; X64-SSE-NEXT:    .cfi_def_cfa_offset 32
; X64-SSE-NEXT:    xorl %esi, %esi
; X64-SSE-NEXT:    xorl %edx, %edx
; X64-SSE-NEXT:    xorl %ecx, %ecx
; X64-SSE-NEXT:    xorl %r8d, %r8d
; X64-SSE-NEXT:    callq __sync_val_compare_and_swap_16
; X64-SSE-NEXT:    movq %rdx, {{[0-9]+}}(%rsp)
; X64-SSE-NEXT:    movq %rax, (%rsp)
; X64-SSE-NEXT:    movaps (%rsp), %xmm0
; X64-SSE-NEXT:    addq $24, %rsp
; X64-SSE-NEXT:    .cfi_def_cfa_offset 8
; X64-SSE-NEXT:    retq
;
; X64-AVX-LABEL: load_fp128:
; X64-AVX:       # %bb.0:
; X64-AVX-NEXT:    subq $24, %rsp
; X64-AVX-NEXT:    .cfi_def_cfa_offset 32
; X64-AVX-NEXT:    xorl %esi, %esi
; X64-AVX-NEXT:    xorl %edx, %edx
; X64-AVX-NEXT:    xorl %ecx, %ecx
; X64-AVX-NEXT:    xorl %r8d, %r8d
; X64-AVX-NEXT:    callq __sync_val_compare_and_swap_16
; X64-AVX-NEXT:    movq %rdx, {{[0-9]+}}(%rsp)
; X64-AVX-NEXT:    movq %rax, (%rsp)
; X64-AVX-NEXT:    vmovaps (%rsp), %xmm0
; X64-AVX-NEXT:    addq $24, %rsp
; X64-AVX-NEXT:    .cfi_def_cfa_offset 8
; X64-AVX-NEXT:    retq
  %v = load atomic fp128, fp128* %fptr unordered, align 16
  ret fp128 %v
}


; sanity check the seq_cst lowering since that's the
; interesting one from an ordering perspective on x86.

define void @store_float_seq_cst(float* %fptr, float %v) {
; X86-LABEL: store_float_seq_cst:
; X86:       # %bb.0:
; X86-NEXT:    movl {{[0-9]+}}(%esp), %eax
; X86-NEXT:    movl {{[0-9]+}}(%esp), %ecx
; X86-NEXT:    xchgl %ecx, (%eax)
; X86-NEXT:    retl
;
; X64-SSE-LABEL: store_float_seq_cst:
; X64-SSE:       # %bb.0:
; X64-SSE-NEXT:    movd %xmm0, %eax
; X64-SSE-NEXT:    xchgl %eax, (%rdi)
; X64-SSE-NEXT:    retq
;
; X64-AVX-LABEL: store_float_seq_cst:
; X64-AVX:       # %bb.0:
; X64-AVX-NEXT:    vmovd %xmm0, %eax
; X64-AVX-NEXT:    xchgl %eax, (%rdi)
; X64-AVX-NEXT:    retq
  store atomic float %v, float* %fptr seq_cst, align 4
  ret void
}

define void @store_double_seq_cst(double* %fptr, double %v) {
; X86-SSE1-LABEL: store_double_seq_cst:
; X86-SSE1:       # %bb.0:
; X86-SSE1-NEXT:    pushl %ebx
; X86-SSE1-NEXT:    .cfi_def_cfa_offset 8
; X86-SSE1-NEXT:    pushl %esi
; X86-SSE1-NEXT:    .cfi_def_cfa_offset 12
; X86-SSE1-NEXT:    .cfi_offset %esi, -12
; X86-SSE1-NEXT:    .cfi_offset %ebx, -8
; X86-SSE1-NEXT:    movl {{[0-9]+}}(%esp), %esi
; X86-SSE1-NEXT:    movl {{[0-9]+}}(%esp), %ebx
; X86-SSE1-NEXT:    movl {{[0-9]+}}(%esp), %ecx
; X86-SSE1-NEXT:    movl (%esi), %eax
; X86-SSE1-NEXT:    movl 4(%esi), %edx
; X86-SSE1-NEXT:    .p2align 4, 0x90
; X86-SSE1-NEXT:  .LBB9_1: # %atomicrmw.start
; X86-SSE1-NEXT:    # =>This Inner Loop Header: Depth=1
; X86-SSE1-NEXT:    lock cmpxchg8b (%esi)
; X86-SSE1-NEXT:    jne .LBB9_1
; X86-SSE1-NEXT:  # %bb.2: # %atomicrmw.end
; X86-SSE1-NEXT:    popl %esi
; X86-SSE1-NEXT:    .cfi_def_cfa_offset 8
; X86-SSE1-NEXT:    popl %ebx
; X86-SSE1-NEXT:    .cfi_def_cfa_offset 4
; X86-SSE1-NEXT:    retl
;
; X86-SSE2-LABEL: store_double_seq_cst:
; X86-SSE2:       # %bb.0:
; X86-SSE2-NEXT:    movl {{[0-9]+}}(%esp), %eax
; X86-SSE2-NEXT:    movsd {{.*#+}} xmm0 = mem[0],zero
; X86-SSE2-NEXT:    movlps %xmm0, (%eax)
; X86-SSE2-NEXT:    lock orl $0, (%esp)
; X86-SSE2-NEXT:    retl
;
; X86-AVX-LABEL: store_double_seq_cst:
; X86-AVX:       # %bb.0:
; X86-AVX-NEXT:    movl {{[0-9]+}}(%esp), %eax
; X86-AVX-NEXT:    vmovsd {{.*#+}} xmm0 = mem[0],zero
; X86-AVX-NEXT:    vmovlps %xmm0, (%eax)
; X86-AVX-NEXT:    lock orl $0, (%esp)
; X86-AVX-NEXT:    retl
;
; X86-NOSSE-LABEL: store_double_seq_cst:
; X86-NOSSE:       # %bb.0:
; X86-NOSSE-NEXT:    pushl %ebx
; X86-NOSSE-NEXT:    .cfi_def_cfa_offset 8
; X86-NOSSE-NEXT:    pushl %esi
; X86-NOSSE-NEXT:    .cfi_def_cfa_offset 12
; X86-NOSSE-NEXT:    .cfi_offset %esi, -12
; X86-NOSSE-NEXT:    .cfi_offset %ebx, -8
; X86-NOSSE-NEXT:    movl {{[0-9]+}}(%esp), %esi
; X86-NOSSE-NEXT:    movl {{[0-9]+}}(%esp), %ebx
; X86-NOSSE-NEXT:    movl {{[0-9]+}}(%esp), %ecx
; X86-NOSSE-NEXT:    movl (%esi), %eax
; X86-NOSSE-NEXT:    movl 4(%esi), %edx
; X86-NOSSE-NEXT:    .p2align 4, 0x90
; X86-NOSSE-NEXT:  .LBB9_1: # %atomicrmw.start
; X86-NOSSE-NEXT:    # =>This Inner Loop Header: Depth=1
; X86-NOSSE-NEXT:    lock cmpxchg8b (%esi)
; X86-NOSSE-NEXT:    jne .LBB9_1
; X86-NOSSE-NEXT:  # %bb.2: # %atomicrmw.end
; X86-NOSSE-NEXT:    popl %esi
; X86-NOSSE-NEXT:    .cfi_def_cfa_offset 8
; X86-NOSSE-NEXT:    popl %ebx
; X86-NOSSE-NEXT:    .cfi_def_cfa_offset 4
; X86-NOSSE-NEXT:    retl
;
; X64-SSE-LABEL: store_double_seq_cst:
; X64-SSE:       # %bb.0:
; X64-SSE-NEXT:    movq %xmm0, %rax
; X64-SSE-NEXT:    xchgq %rax, (%rdi)
; X64-SSE-NEXT:    retq
;
; X64-AVX-LABEL: store_double_seq_cst:
; X64-AVX:       # %bb.0:
; X64-AVX-NEXT:    vmovq %xmm0, %rax
; X64-AVX-NEXT:    xchgq %rax, (%rdi)
; X64-AVX-NEXT:    retq
  store atomic double %v, double* %fptr seq_cst, align 8
  ret void
}

define float @load_float_seq_cst(float* %fptr) {
; X86-SSE1-LABEL: load_float_seq_cst:
; X86-SSE1:       # %bb.0:
; X86-SSE1-NEXT:    pushl %eax
; X86-SSE1-NEXT:    .cfi_def_cfa_offset 8
; X86-SSE1-NEXT:    movl {{[0-9]+}}(%esp), %eax
; X86-SSE1-NEXT:    movl (%eax), %eax
; X86-SSE1-NEXT:    movl %eax, (%esp)
; X86-SSE1-NEXT:    flds (%esp)
; X86-SSE1-NEXT:    popl %eax
; X86-SSE1-NEXT:    .cfi_def_cfa_offset 4
; X86-SSE1-NEXT:    retl
;
; X86-SSE2-LABEL: load_float_seq_cst:
; X86-SSE2:       # %bb.0:
; X86-SSE2-NEXT:    pushl %eax
; X86-SSE2-NEXT:    .cfi_def_cfa_offset 8
; X86-SSE2-NEXT:    movl {{[0-9]+}}(%esp), %eax
; X86-SSE2-NEXT:    movss {{.*#+}} xmm0 = mem[0],zero,zero,zero
; X86-SSE2-NEXT:    movss %xmm0, (%esp)
; X86-SSE2-NEXT:    flds (%esp)
; X86-SSE2-NEXT:    popl %eax
; X86-SSE2-NEXT:    .cfi_def_cfa_offset 4
; X86-SSE2-NEXT:    retl
;
; X86-AVX-LABEL: load_float_seq_cst:
; X86-AVX:       # %bb.0:
; X86-AVX-NEXT:    pushl %eax
; X86-AVX-NEXT:    .cfi_def_cfa_offset 8
; X86-AVX-NEXT:    movl {{[0-9]+}}(%esp), %eax
; X86-AVX-NEXT:    vmovss {{.*#+}} xmm0 = mem[0],zero,zero,zero
; X86-AVX-NEXT:    vmovss %xmm0, (%esp)
; X86-AVX-NEXT:    flds (%esp)
; X86-AVX-NEXT:    popl %eax
; X86-AVX-NEXT:    .cfi_def_cfa_offset 4
; X86-AVX-NEXT:    retl
;
; X86-NOSSE-LABEL: load_float_seq_cst:
; X86-NOSSE:       # %bb.0:
; X86-NOSSE-NEXT:    pushl %eax
; X86-NOSSE-NEXT:    .cfi_def_cfa_offset 8
; X86-NOSSE-NEXT:    movl {{[0-9]+}}(%esp), %eax
; X86-NOSSE-NEXT:    movl (%eax), %eax
; X86-NOSSE-NEXT:    movl %eax, (%esp)
; X86-NOSSE-NEXT:    flds (%esp)
; X86-NOSSE-NEXT:    popl %eax
; X86-NOSSE-NEXT:    .cfi_def_cfa_offset 4
; X86-NOSSE-NEXT:    retl
;
; X64-SSE-LABEL: load_float_seq_cst:
; X64-SSE:       # %bb.0:
; X64-SSE-NEXT:    movss {{.*#+}} xmm0 = mem[0],zero,zero,zero
; X64-SSE-NEXT:    retq
;
; X64-AVX-LABEL: load_float_seq_cst:
; X64-AVX:       # %bb.0:
; X64-AVX-NEXT:    vmovss {{.*#+}} xmm0 = mem[0],zero,zero,zero
; X64-AVX-NEXT:    retq
  %v = load atomic float, float* %fptr seq_cst, align 4
  ret float %v
}

define double @load_double_seq_cst(double* %fptr) {
; X86-SSE1-LABEL: load_double_seq_cst:
; X86-SSE1:       # %bb.0:
; X86-SSE1-NEXT:    subl $20, %esp
; X86-SSE1-NEXT:    .cfi_def_cfa_offset 24
; X86-SSE1-NEXT:    movl {{[0-9]+}}(%esp), %eax
; X86-SSE1-NEXT:    fildll (%eax)
; X86-SSE1-NEXT:    fistpll {{[0-9]+}}(%esp)
; X86-SSE1-NEXT:    movl {{[0-9]+}}(%esp), %eax
; X86-SSE1-NEXT:    movl {{[0-9]+}}(%esp), %ecx
; X86-SSE1-NEXT:    movl %ecx, {{[0-9]+}}(%esp)
; X86-SSE1-NEXT:    movl %eax, (%esp)
; X86-SSE1-NEXT:    fldl (%esp)
; X86-SSE1-NEXT:    addl $20, %esp
; X86-SSE1-NEXT:    .cfi_def_cfa_offset 4
; X86-SSE1-NEXT:    retl
;
; X86-SSE2-LABEL: load_double_seq_cst:
; X86-SSE2:       # %bb.0:
; X86-SSE2-NEXT:    subl $12, %esp
; X86-SSE2-NEXT:    .cfi_def_cfa_offset 16
; X86-SSE2-NEXT:    movl {{[0-9]+}}(%esp), %eax
; X86-SSE2-NEXT:    movsd {{.*#+}} xmm0 = mem[0],zero
; X86-SSE2-NEXT:    movlps %xmm0, (%esp)
; X86-SSE2-NEXT:    fldl (%esp)
; X86-SSE2-NEXT:    addl $12, %esp
; X86-SSE2-NEXT:    .cfi_def_cfa_offset 4
; X86-SSE2-NEXT:    retl
;
; X86-AVX-LABEL: load_double_seq_cst:
; X86-AVX:       # %bb.0:
; X86-AVX-NEXT:    subl $12, %esp
; X86-AVX-NEXT:    .cfi_def_cfa_offset 16
; X86-AVX-NEXT:    movl {{[0-9]+}}(%esp), %eax
; X86-AVX-NEXT:    vmovsd {{.*#+}} xmm0 = mem[0],zero
; X86-AVX-NEXT:    vmovlps %xmm0, (%esp)
; X86-AVX-NEXT:    fldl (%esp)
; X86-AVX-NEXT:    addl $12, %esp
; X86-AVX-NEXT:    .cfi_def_cfa_offset 4
; X86-AVX-NEXT:    retl
;
; X86-NOSSE-LABEL: load_double_seq_cst:
; X86-NOSSE:       # %bb.0:
; X86-NOSSE-NEXT:    subl $20, %esp
; X86-NOSSE-NEXT:    .cfi_def_cfa_offset 24
; X86-NOSSE-NEXT:    movl {{[0-9]+}}(%esp), %eax
; X86-NOSSE-NEXT:    fildll (%eax)
; X86-NOSSE-NEXT:    fistpll {{[0-9]+}}(%esp)
; X86-NOSSE-NEXT:    movl {{[0-9]+}}(%esp), %eax
; X86-NOSSE-NEXT:    movl {{[0-9]+}}(%esp), %ecx
; X86-NOSSE-NEXT:    movl %ecx, {{[0-9]+}}(%esp)
; X86-NOSSE-NEXT:    movl %eax, (%esp)
; X86-NOSSE-NEXT:    fldl (%esp)
; X86-NOSSE-NEXT:    addl $20, %esp
; X86-NOSSE-NEXT:    .cfi_def_cfa_offset 4
; X86-NOSSE-NEXT:    retl
;
; X64-SSE-LABEL: load_double_seq_cst:
; X64-SSE:       # %bb.0:
; X64-SSE-NEXT:    movsd {{.*#+}} xmm0 = mem[0],zero
; X64-SSE-NEXT:    retq
;
; X64-AVX-LABEL: load_double_seq_cst:
; X64-AVX:       # %bb.0:
; X64-AVX-NEXT:    vmovsd {{.*#+}} xmm0 = mem[0],zero
; X64-AVX-NEXT:    retq
  %v = load atomic double, double* %fptr seq_cst, align 8
  ret double %v
}