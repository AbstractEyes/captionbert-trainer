# Additional frozen phase control — 2026-11-06

This exploratory development measurement separates two parts of the complete
binding checkpoint. The existing binding-off intervention returns its lookup
head to hierarchical rotation and bypasses the record-key norm. It therefore
does not isolate lexical phase variation.

The additional phase_constant condition replaces all bound word phases with
zero while preserving the original active/end/first-record masks, record norm,
value-address tables, model weights and the other heads. It tests dependence
on lexical phase variation in the already-trained model. It does not establish
which architecture would learn best after retraining.

Run intact and phase_constant on both completed KALLP screen seeds, the same
4,096 chunk061 rows, exact query labels and configured head. Include teacher and
STS-B development measurements. No final test score or this intervention enters
the default-model selection. Preserve all earlier reports; write the separate
frozen-phase-constant-20261106 folder.

The context restores the original word-feature function even after exceptions.
Its CPU test verifies zero phases, unchanged masks/record identity, unchanged
norm and restoration. The original queued source59ec953 remains untouched.
A separate third-runtime job runs this finite control after its serving check;
there is no additional accuracy training. It prepares only development chunk61,
keeps two CPU threads, uses a92GB GPU ceiling and retains the full console.
