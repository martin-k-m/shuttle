# Changelog

## v0.1.0 (unreleased)

First cut of shuttle, inference and serving for twill, written in twill.

It does not run. twill's `mode systems` is still being built. See
`docs/needs.md` for what is missing and `README.md` for the status table.
Nothing below has ever executed.

Added:

- Loading a selvedge archive, with integrity verified before the weights are
  trusted, and loading a bare parameter tree with a declared signature. The
  difference between a signature that is a fact and one that is a claim is
  stated at both call sites.
- Single, batched and streaming prediction, with the forward function passed in
  by the caller so the model stays the caller's.
- Input validation against the model's declared shapes, producing twill's kind
  of message with the axis named. A batch passed to the single-input path, an
  empty batch, and a forward function that returns the wrong number of rows are
  each diagnosed specifically.
- Dynamic batching with the latency-throughput trade expressed as two required
  numbers, neither of which has a default, plus a `validate` that refuses the
  configurations where a knob does nothing and a `report` that gives the
  realised trade next to the configured one.
- Warmup, at named batch shapes, with a written-out list of what it covers and
  what it does not, and a refusal to claim a saving there is no clock to measure.
- int8, float16 and bfloat16 with min-max and percentile calibration, a size and
  accuracy table where every figure is labelled DERIVED, GATED or UNMEASURED, and
  a `compare` that measures the accuracy cost on the caller's own held-out data.
  The numerics run on the language's real dtypes now: `.to(f16)` and `.to(bf16)`
  are correctly-rounded casts rather than the earlier hand-rolled approximation,
  int8 codes are centred into a real signed i8 tensor, and `stored_bytes` takes a
  `realised` flag separating the footprint today from the one after the packed
  buffer and a narrow archive encoding land upstream.
- Batch scoring over a dataset, chunked so peak memory is one chunk's
  activations, with progress reporting and an accuracy evaluation.

Deliberately not included in v0.1, and in most cases not possible:

- A network server, a port, or anything that listens. twill has no sockets.
- Any overlap between waiting and computing. twill has no concurrency.
- A progress time estimate or a warmup saving. twill has no clock.
- A quantisation size win yet. The dtypes round for real, but the bytes drop
  only once twill NEEDS-111's packed buffer and a narrow archive encoding land.
- Activation calibration. It would mean owning the forward pass, which shuttle
  deliberately does not.
