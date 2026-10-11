/* QEMU mps2-an500 (Cortex-M7): 4 MiB SSRAM at 0x00000000 holds the whole
   image; QEMU loads the ELF sections directly. */
MEMORY { RAM (rwx) : ORIGIN = 0x00000000, LENGTH = 4M }
ENTRY(Reset_Handler)
SECTIONS {
  .isr_vector : { KEEP(*(.isr_vector)) } > RAM
  .text : { *(.text*) *(.rodata*) KEEP(*(.init)) KEEP(*(.fini)) } > RAM
  .preinit_array : { PROVIDE_HIDDEN(__preinit_array_start = .); KEEP(*(.preinit_array*)) PROVIDE_HIDDEN(__preinit_array_end = .); } > RAM
  .init_array : { PROVIDE_HIDDEN(__init_array_start = .); KEEP(*(SORT(.init_array.*))) KEEP(*(.init_array*)) PROVIDE_HIDDEN(__init_array_end = .); } > RAM
  .fini_array : { PROVIDE_HIDDEN(__fini_array_start = .); KEEP(*(.fini_array*)) PROVIDE_HIDDEN(__fini_array_end = .); } > RAM
  .ARM.exidx : { *(.ARM.exidx*) } > RAM
  .data : { *(.data*) } > RAM
  .bss (NOLOAD) : { __bss_start__ = .; *(.bss*) *(COMMON) __bss_end__ = .; } > RAM
  . = ALIGN(8);
  end = .; _end = .;
  __stack_bottom = ORIGIN(RAM) + LENGTH(RAM) - 64K;
  _estack = ORIGIN(RAM) + LENGTH(RAM);
}
