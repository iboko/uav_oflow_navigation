// Bare-metal Cortex-M7 harness for QEMU mps2-an500: runs the SAME
// workload as the host benchmark and 'ofnav bench' and reports executed
// instructions (QEMU -icount: virtual time is proportional to retired
// instructions). QEMU is not cycle accurate: cycles are estimated from the
// instruction count by docs/ESKF15_FMU_V6X_RU.md. Output via semihosting.
#include "ofnav/eskf15_workload.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <initializer_list>

extern "C" {
extern std::uint32_t __bss_start__, __bss_end__, _estack, __stack_bottom;
void __libc_init_array();
int main();
void Reset_Handler();
void Default_Handler() { for (;;) {} }
// -nostartfiles: no crti/crtn, __libc_init_array still calls these.
void _init() {}
void _fini() {}
__attribute__((section(".isr_vector"), used))
void (*const g_vectors[16])() = {
    reinterpret_cast<void (*)()>(&_estack), Reset_Handler, Default_Handler, Default_Handler,
    Default_Handler, Default_Handler, Default_Handler, nullptr, nullptr, nullptr, nullptr,
    Default_Handler, Default_Handler, nullptr, Default_Handler, Default_Handler};
}

namespace {

volatile std::uint32_t &reg(std::uintptr_t a) { return *reinterpret_cast<volatile std::uint32_t *>(a); }

int semihost(int op, const void *arg) {
    register int r0 __asm__("r0") = op;
    register const void *r1 __asm__("r1") = arg;
    __asm__ volatile("bkpt 0xAB" : "+r"(r0) : "r"(r1) : "memory");
    return r0;
}
void put(const char *s) { semihost(0x04, s); }  // SYS_WRITE0

// SysTick (24-bit, counts down) accumulated into a 64-bit tick count.
std::uint32_t g_last = 0;
std::uint64_t g_total = 0;
std::uint64_t sysTicks() {
    const std::uint32_t now = reg(0xE000E018) & 0xFFFFFFU;
    g_total += (g_last - now) & 0xFFFFFFU;
    g_last = now;
    return g_total;
}
void startSysTick() {
    reg(0xE000E014) = 0xFFFFFFU;  // reload
    reg(0xE000E018) = 0U;         // current
    reg(0xE000E010) = 5U;         // enable, processor clock, no interrupt
    g_last = reg(0xE000E018) & 0xFFFFFFU;
}
// Known instruction count: 2 instructions (subs, bne) per iteration.
__attribute__((noinline)) void spin(std::uint32_t n) {
    __asm__ volatile("1: subs %0, %0, #1\n bne 1b" : "+r"(n) : : "cc");
}

ofnav15::WorkloadBuffers g_buffers;

__attribute__((noinline)) void paintStack() {
    // Paint from the stack bottom up to 64 bytes below the current SP.
    std::uint32_t *p = &__stack_bottom;
    std::uint32_t *sp;
    __asm__ volatile("mov %0, sp" : "=r"(sp));
    while (p < sp - 16) { *p++ = 0xA5A5A5A5U; }
}
std::uint32_t stackUsed() {
    std::uint32_t *p = &__stack_bottom;
    while (*p == 0xA5A5A5A5U) { ++p; }
    return static_cast<std::uint32_t>(reinterpret_cast<std::uintptr_t>(&_estack) -
                                      reinterpret_cast<std::uintptr_t>(p));
}

} // namespace

extern "C" void Reset_Handler() {
    reg(0xE000ED88) |= (0xFU << 20);  // CPACR: enable CP10/CP11 (FPU)
    __asm__ volatile("dsb\n isb");
    for (std::uint32_t *p = &__bss_start__; p < &__bss_end__; ++p) { *p = 0U; }
    __libc_init_array();
    const int rc = main();
    const std::uint32_t exit_args[2] = {0x20026U, static_cast<std::uint32_t>(rc)};
    semihost(0x20, exit_args);  // SYS_EXIT_EXTENDED
    for (;;) {}
}

int main() {
    startSysTick();
    // Calibrate: instructions per SysTick tick.
    const std::uint64_t c0 = sysTicks();
    spin(2000000U);
    const std::uint64_t c1 = sysTicks();
    const double insn_per_tick = 4000000.0 / static_cast<double>(c1 - c0);
    bool healthy = true;
    for (const std::uint64_t period : {10000ULL, 20000ULL}) {
        paintStack();
        const std::uint32_t base_used = stackUsed();
        const ofnav15::WorkloadResult r = ofnav15::runWorkload(g_buffers, &sysTicks, 30, period);
        const std::uint32_t used = stackUsed();
        healthy = healthy && r.healthy && r.failed_events == 0U;
        char line[640];
        std::snprintf(line, sizeof(line),
            "{\"type\":\"QEMU_CORTEX_M7_ESKF15_INSTRUCTION_COUNT\",\"imu_rate_hz\":%lu,"
            "\"insn_per_tick\":%.3f,"
            "\"imu_samples\":%lu,\"frames\":%lu,\"processed\":%lu,\"failed\":%lu,\"healthy\":%s,"
            "\"imu_insn_mean\":%lu,\"imu_insn_worst\":%lu,"
            "\"frame_insn_mean\":%lu,\"frame_insn_worst\":%lu,\"max_replay\":%u,"
            "\"stack_used_bytes\":%lu,\"stack_used_before_workload\":%lu,"
            "\"sizeof_workload_buffers\":%u}\n",
            static_cast<unsigned long>(1000000ULL / period), insn_per_tick,
            static_cast<unsigned long>(r.imu_samples),
            static_cast<unsigned long>(r.frames), static_cast<unsigned long>(r.processed),
            static_cast<unsigned long>(r.failed_events), r.healthy ? "true" : "false",
            static_cast<unsigned long>(insn_per_tick * r.imu_total / r.imu_samples),
            static_cast<unsigned long>(insn_per_tick * r.imu_worst),
            static_cast<unsigned long>(insn_per_tick * r.frame_total / r.frames),
            static_cast<unsigned long>(insn_per_tick * r.frame_worst),
            static_cast<unsigned>(r.max_replay_length),
            static_cast<unsigned long>(used), static_cast<unsigned long>(base_used),
            static_cast<unsigned>(sizeof(g_buffers)));
        put(line);
    }
    return healthy ? 0 : 1;
}
