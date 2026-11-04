<!-- SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0 -->
<!-- Copyright (C) 2026 Ahmad Ali Parr / SnapKitty Research Lab -->

# ref/

These are unmodified copies of the CPU reference from bonsai-decoder (commit `7bcdb88`): GGUF reader, model loader, tokenizer, CPU decoder, and the synthetic qwen35 GGUF writer used by the tests.

They are built into `libbd_ref.a` and used for:

- loading the model (`bd::Model`, `bd::Tokenizer`)
- the CPU reference decode that tests and `btf_golden` compare against
- `bd::make_synth_qwen35`, the tiny synthetic models in CTest

They are copies so that this project builds alone. Fix bugs upstream, not here.
