<p align="center">
  <img alt="twill" src="https://raw.githubusercontent.com/twill-lang/twill/main/assets/twill-mark.png" width="120">
</p>

<h1 align="center">shuttle</h1>

<p align="center">
  <b>Inference and serving for <a href="https://github.com/twill-lang/twill">twill</a>.</b><br>
  Written in twill.
</p>

<p align="center">
  <img alt="shuttle" src="https://img.shields.io/badge/shuttle-v0.1.0-7FE3C4?style=flat-square&labelColor=12332C">
  <img alt="written in twill" src="https://img.shields.io/badge/written%20in-twill-D2F0E4?style=flat-square&labelColor=12332C">
  <img alt="no network server" src="https://img.shields.io/badge/network%20server-none-4FB79B?style=flat-square&labelColor=12332C">
  <img alt="status" src="https://img.shields.io/badge/tests-passing-4FB79B?style=flat-square&labelColor=12332C">
  <img alt="MIT" src="https://img.shields.io/badge/license-MIT-D2F0E4?style=flat-square&labelColor=12332C">
</p>

---

## It runs

`shuttle` is written in twill, in `.tw` files, using `mode systems`. That subset
did not exist when this library was written, so for a long time none of the code
here executed and this section said so. twill 1.6 is the release that closed it:
the 5 test suites under `tests/` pass, and CI runs them against a released
twill on every push rather than gating on the prose in this file.

```bash
twill test tests
```

You need twill 1.6.2 or newer. `docs/needs.md` is still worth reading -- it
is the list of what this library asked the language for, and it now records
which of those arrived and which are still open.

## What shuttle is

The layer between a trained model and the thing that uses it. loom trains,
[selvedge](https://github.com/twill-lang/selvedge) ships, shuttle answers.

| Piece | State |
| --- | --- |
| Load a selvedge archive, integrity verified before the weights are trusted | written, unrun |
| Single, batched and streaming prediction | written, unrun |
| Input validation against the model's declared shapes, with twill's kind of message | written, unrun |
| Dynamic batching with the latency-throughput trade as two required numbers | written, unrun |
| Warmup, with a written-out list of what it does and does not cover | written, unrun |
| int8, float16 and bfloat16 with min-max and percentile calibration | written, unrun |
| Batch scoring over a dataset, chunked, with progress | written, unrun |
| Accuracy evaluation over a labelled dataset | written, unrun |
| A quantisation size win | **numerics real; bytes gated on twill NEEDS-111.** See below |
| A progress time estimate | **not possible.** There is no clock |
| A network server, a port, a socket, a request thread | **not possible, and not planned** |
| Anything running end to end | **no** |

## The forward pass stays yours

shuttle does not own your update rule and it does not own your forward pass
either. Every entry point takes the forward function as an argument. The thing
that makes a model yours is its forward pass, and a serving library that hid it
inside itself would be a library you had to fork to serve anything it did not
anticipate.

What shuttle owns is everything around the call: the signature check, the
batching, the warmup, the quantised parameters, the chunking, and the counting.

```rust
mode systems

import "twill_modules/shuttle/src/model.tw" as md
import "twill_modules/shuttle/src/predict.tw" as pr
import "twill_modules/shuttle/src/batcher.tw" as bat
import "twill_modules/shuttle/src/warmup.tw" as wu

fn forward(p: Tree, x: Tensor) -> Tensor {
  let h = relu(x @ transpose(p.w1) + p.b1)
  h @ transpose(p.w2) + p.b2
}

# The signature comes out of the archive, so the shape check is against what the
# model was published with rather than what this file believes.
let loaded = md.from_archive("models/blobs-1.2.0.slv")
let model = loaded.model

# Both numbers are required. Neither has a default. See below.
let cfg = bat.config(32, 4)
wu.warm(model, wu.sizes_for(cfg.max_batch), forward)

let one = pr.predict(model, [1.2, 0.4, -0.8, 2.1], forward)
```

`examples/serve.tw` is a complete program: load, warm, one request, a batched
stream of requests, a scored file, and a measured comparison against int8.

## Shape errors, at the boundary

twill's best property is that a shape mistake is an error you see before the
program runs. Serving gives that up: the input arrives at runtime, from a file
or a caller, and the checker never saw it. A served model is exactly the place
twill's guarantee does not reach.

So shuttle rebuilds it by hand, at the entry point, in the same vocabulary:

```
toy: shape error: the input axis 1 is 6 but the model expects 4 ([8, 6] against [_, 4])
```

rather than what happens without it, which is a matmul failing four calls deep
with the shapes of two intermediate tensors nobody named. The axis is named, not
just the shapes: for a rank-4 image tensor, "expected `[_, 1, 28, 28]` and got
`[16, 1, 28, 32]`" makes a reader diff two lists, and "axis 3 is 32 and the
model expects 28" does not.

Validation happens once, at the entry point, and nothing downstream re-checks. A
check inside a loop is a check someone removes for speed.

Three cases get their own message because they are the mistakes people actually
make. Passing a batch to the single-input path is diagnosed as that, and names
`predict_batch`. An empty batch is refused rather than answered with an empty
output, because a service that silently returns nothing for an empty request is
a service whose caller has a bug it cannot see. And a forward function that
returns a different number of rows than it was given is caught and blamed on the
model rather than on the request.

## The batching knob is two numbers and neither has a default

Batching n requests into one forward pass is faster per request and slower for
the first request in the batch. That is not a tuning detail. It is the deciding
question about a serving system, its answer is different for a fraud check and
for an overnight job, and neither answer is knowable from inside a library.

So shuttle has no heuristic. It has two numbers, both required:

| Knob | What it buys | What it costs |
| --- | --- | --- |
| `max_batch` | throughput per row, wider matmuls, fixed cost paid once | memory, proportional to the batch |
| `max_hold` | throughput, by letting a batch fill | latency, for every request that waits |

`max_hold = 0` means never wait: every request runs alone, latency as low as it
goes, throughput as bad as it gets. `max_hold = max_batch - 1` means always fill
the batch. Everything useful is in between and where in between is your
decision.

`validate` refuses the configurations that do nothing, rather than accepting
them quietly:

```
max_hold 8 is not below max_batch 8, so the hold never expires and only
max_batch has any effect
```

And `report` gives you the realised trade next to the configured one, because a
configured `max_batch` of 64 against a realised mean of 3 means the knobs are
not doing anything and nobody would otherwise notice.

### The hold counts arrivals, not milliseconds

Every real dynamic batcher holds a batch for a duration. shuttle cannot: twill
has no clock. `max_hold` counts arrivals instead.

This is a worse knob and it is the honest one. Counting arrivals bounds the wait
in requests and not in time, so under a trickle of traffic a request can wait
indefinitely for the next one and the tail latency is unbounded.

**shuttle's hold is safe for a bounded workload and unsafe for an open-ended
stream of requests.** Batch scoring a file is bounded. Anything driven by
arrivals you do not control is not. `flush` runs whatever is queued regardless,
and calling it is the caller's job because only the caller knows its input is
exhausted. [`docs/needs.md`](docs/needs.md) entry 3 is the clock that fixes
this, and it is the first entry for a reason.

There is also no concurrency, so nothing arrives while something else is
running. This is a batcher in the sense of accumulating work and cutting it into
passes of a chosen size, which is the part that matters for throughput. It does
not overlap waiting with computing, because there is nothing to overlap with.

## What warmup covers, and what it does not

Warmup runs the model on synthetic input, at the shapes real requests will have,
before the service starts answering. A warmup with an unstated scope is a
ritual, so here is the scope.

**Covered.** Lazy work inside your forward function, which is the largest real
item and is entirely in your code: a constant, a mask, a positional encoding or
a lookup table built on first use. Dequantisation, if the model is quantised,
which for int8 is the largest single item on this list. Faulting the parameter
pages in after loading a large archive. And every distinct batch shape you name,
which is why it takes a list: a model warmed at batch 1 and asked for batch 64
pays the batch-64 costs on the batch-64 request.

**Not covered.** JIT compilation, kernel selection and autotuning, because twill
has none of them. If you have read about warmup in another framework, most of
what it bought there does not exist here to buy. Nothing on a GPU, because there
is no GPU backend. Data-dependent branches, because synthetic input warms the
wrong path for a model whose cost depends on its values rather than its shapes:
`warm_with` takes real rows for exactly that. And anything after the first
request.

**The honest summary.** For a dense float model in twill today, warmup buys the
lazy work in your own forward function and the page faults. Real, and not large.
For an int8 model it buys the dequantisation, which is large. shuttle reports
the shapes it warmed and refuses to claim a saving, because there is no clock to
measure one with.

Synthetic input is drawn from the standard normal, not zeros. A relu on a zero
input is uniformly on the flat side and every comparison against zero takes one
branch, which warms half the code.

## Quantisation, and what it actually costs

Read this before using `src/quant.tw`.

**twill now has narrow dtypes.** A tensor carries a dtype, `x.to(f16)` and
`x.to(i8)` are correctly-rounded casts, and the conversions match the formats
exactly, subnormals and overflow included. So **the numerics shuttle applies are
real today**: quantising rounds the way an int8, f16 or bf16 deployment would,
and you can measure the accuracy cost on your data before committing to it.

**The bytes do not drop yet, and that is a gate rather than a fiction.** A narrow
tensor holds narrow values but still occupies eight-byte slots until two things
land upstream: the packed byte buffer that backs it (twill NEEDS-111, four
native primitives) and a narrow on-disk encoding (a selvedge archive format
bump). When they do, the ratios below are real and no code in `src/quant.tw`
changes, because the rounding was always the hard half. Until then, `stored_bytes`
with `realised: false` reports the true current footprint and with `true` the
footprint after the gate clears; the default is never the aspiration.

Every number below is labelled with where it comes from. Nothing here is a
benchmark result: shuttle runs as of twill 1.6, but no benchmark has been run.

**Quantising does not make the file smaller today.** twill stores every float as
an f64 whatever dtype it carries, so rounding a weight to bfloat16 changes the
value and not the eight bytes it sits in. The ratios below are what the file
becomes once the packed storage lands.

| Scheme | Bytes/param | vs f64 | Max representation error | Task accuracy |
| --- | --- | --- | --- | --- |
| float64 | 8 | 1.00x | 0 | baseline |
| bfloat16 | 2 | 4.00x | 3.9e-3 relative, DERIVED | **UNMEASURED** |
| float16 | 2 | 4.00x | 4.88e-4 relative, DERIVED | **UNMEASURED** |
| int8 | 1 + scales | ~7.9x | half a step, DERIVED | **UNMEASURED** |

- **GATED, not PROJECTED.** The ratios are real once the two upstream pieces
  named above land; the tensor already rounds to the dtype, so nothing but the
  storage stands between here and the number. `stored_bytes(m, scheme, tensors,
  realised)` returns the current footprint or the gated one by its last argument.
- **DERIVED, bfloat16.** Seven stored significand bits round a value to within
  2^-8, about 3.9e-3 relative. It keeps f32's eight exponent bits, so its range
  is f32's and it does not overflow a value f32 could hold: no clamp, no loss
  scaling. This is the safer 16-bit default.
- **DERIVED, float16.** An 11-bit significand rounds a value to within 2^-11 of
  itself, which is 4.88e-4 relative, finer than bf16. Its range is only about
  6e-5 to 65504, and a weight outside it flushes to zero or to infinity. shuttle
  no longer hides that behind a clamp, because the cast is real: an f16 overflow
  is a genuine hazard and the reason to prefer bf16 unless the finer step is
  measured to pay.
- **DERIVED, int8.** Over a calibrated range the step is `(hi - lo) / 255` and
  rounding puts each weight within half a step. The 256 levels are centred to a
  real signed i8 tensor, one byte per code. The `~7.9x` rather than `8x` is the
  per-tensor scale and zero point: `n / (n/8 + 2)` bytes, which is 7.94x at
  n = 1024 and 7.999x at n = 1e6.
- **UNMEASURED.** What happens to your model's accuracy. It depends on the depth
  of the network, on whether the errors accumulate or cancel, and on how close
  your decision boundary is. shuttle will not guess, and `compare` runs both
  models over your held-out data in one call so you do not have to. It reports the
shape below, and the values are yours rather than ours:

```
f16:  over N rows: mean |diff| ..., max |diff| ..., argmax disagreements ...
int8: over N rows: mean |diff| ..., max |diff| ..., argmax disagreements ...
```

The argmax disagreement count is the one that decides whether a quantisation is
acceptable, and it is not derivable from the weight error: two models can differ
by 1e-6 everywhere and disagree on 3% of rows near the boundary.

Calibration is the whole decision for int8. `minmax` loses nothing at the
extremes and everything in the middle if there is one outlier weight: a single
weight at 100 with the rest inside [-1, 1] gives a step of 0.4, and every real
weight rounds to one of five values. `percentile(0.999)` clips the tail and gets
a step a hundred times finer. It is almost always the better trade and it is
not the default, because clipping a weight should be a decision someone made.

Activation calibration is not implemented. Most of the accuracy loss in a real
int8 deployment comes from activations rather than weights, and calibrating them
means watching the inside of the forward pass, which is deliberately the
caller's. `docs/needs.md` entry 8 records that as a decision rather than an
omission.

## Batch scoring

The offline half. No requests arrive, the input is a file, and the failure that
hurts is running out of memory at 80% through a four-hour job.

```rust
let scored = sc.score_csv(model, "batch.csv", 512,
  sc.progress(5000, "scoring", 0), forward)
```

```
scoring 5000/120000 (4%)
scoring 10000/120000 (8%)
```

Chunked, so peak memory is one chunk's activations rather than the file's:
`predict_batch` on a million rows materialises a million rows of every
intermediate in the forward pass. `score_to` holds nothing at all, handing each
chunk to a sink as it is produced, and a sink that returns an error stops the
job, which is how a caller cancels.

The chunk size does not change the answer. It changes peak memory and nothing
else, and there is a test that says so.

**The input is not streamed.** `read_csv` reads the whole file, because twill
has no streaming reader, so a file larger than memory cannot be scored. The
chunking bounds the activations and not the input. `docs/needs.md` entry 5.

**No time estimate.** There is no clock. The useful part of a progress report on
a four-hour job is the time remaining, and shuttle prints a percentage.

## Stated limits

Collected, so none of them has to be discovered.

- **No network server, no socket, no port, no request thread.** twill has none
  of the primitives and this is not planned.
- **No concurrency.** The batcher accumulates and cuts; it does not overlap.
- **No clock.** The batching hold counts arrivals, warmup cannot report a
  saving, and progress has no estimate.
- **Quantisation rounds for real but does not shrink the file yet.** The
  numerics are exact; the bytes drop once twill NEEDS-111 and a narrow archive
  encoding land. Until then it measures what shrinking will cost.
- **The input to `score_csv` is read whole.** Only the activations are chunked.
- **Lit progress lines, but no stateful bar.** The progress line is coloured
  from twill's palette, vendored now that the terminal layer is reachable from a
  package, and drops to plain text when piped. The rate-and-ETA bar from
  `src/cli/progress.tw` is still not adopted; `docs/needs.md` entry 11.
- **`flush` is the caller's job.** Forget it and the last partial batch is
  never run.

## Install

Once spool and `mode systems` both work:

```
spool add shuttle https://github.com/twill-lang/shuttle
```

spool vendors into `twill_modules/`, and twill's import is a path, so the import
lines are the long ones in the example above and they resolve relative to the
project root. That is twill's rule rather than shuttle's; see spool's README.

## Repository layout

```
src/signature.tw      shapes, and the errors twill would have given you
src/model.tw          a loaded model, from an archive or from bare parameters
src/predict.tw        single, batched and streaming prediction
src/batcher.tw        the latency-throughput trade, as two required numbers
src/warmup.tw         what it covers and what it does not
src/quant.tw          int8, float16 and bfloat16, calibration, and an honest table
src/score.tw          batch scoring, progress, and accuracy
tests/                tests, named as sentences
examples/serve.tw     load, warm, predict, batch, score, compare
docs/needs.md         what the language still has to provide
```

## Dependencies

twill, and [selvedge](https://github.com/twill-lang/selvedge) for reading model
archives. Nothing else, and no Go.

selvedge is a real dependency rather than a convenience: `from_archive` gets the
model's signature out of the file, so the shape check is against what the model
was published with. `from_params` takes a bare `save`d tree and a signature you
declare, which works and turns the signature into a claim. A declared signature
that is wrong produces a check that passes and a forward pass that fails, which
is worse than no check, because it moved the error later without removing it.

## Contributing

The most useful contribution right now is not code. It is a correction to
[`docs/needs.md`](docs/needs.md): a feature listed there that the language
already has, a workaround that is worse than described, or a missing entry found
by reading the source. After that, the quantisation table is the part most worth
arguing with, because every number in it is labelled with where it came from and
a wrong label is a bug.

## License

MIT. See [LICENSE](LICENSE).
