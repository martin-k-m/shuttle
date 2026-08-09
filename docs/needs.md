# What shuttle needs from twill

shuttle is written in twill and does not run yet. This file is the reason: the
language and runtime features the source uses that twill does not provide today,
with the file and the function that needs each one, and what shuttle does in the
meantime.

It is a work queue for the language, not a complaint. Every entry was reached by
writing real code and hitting the wall, which is the only way a list like this
is worth anything. Where shuttle has a workaround the workaround is described,
because how ugly a workaround is measures how badly the feature is wanted.

The baseline is milestone 1 of `docs/self-hosting.md` in the twill repository:
`mode systems`, `I64`, `Str` with length and byte indexing, `Arr[T]`,
`Dict[Str, V]`, `struct`, and `read_file`. Everything below is on top of that.

Entries 1, 4, 9, 10, 11, 12 and 13 are walls loom, spool or selvedge also hit.
They are restated here with shuttle's call sites rather than cross-referenced,
because a work queue that makes you read four repositories to find out what is
blocking is a work queue nobody reads.

## Blocking: shuttle cannot run at all without these

### 1. `mode systems` itself

**Used by:** every file
**Status:** designed in `docs/self-hosting.md`, not implemented.

Nothing else on this list matters until this does.

### 2. A narrow tensor storage: int8, float16 and bfloat16

**Needs:** a tensor whose elements are stored in fewer than eight bytes, and an
on-disk encoding for one
**Used by:** `src/quant.tw`, the whole file
**Status:** half landed. The dtype semantics are in the language now:
`docs/dtypes.md` in raster designs seven dtypes, `src/tensor.tw` stores each
tensor's elements rounded to its dtype, `x.to(f16)` and `x.to(int8)` are
correctly-rounded casts, and `src/quant.tw` was rewritten onto them. Two pieces
are still missing before the bytes actually drop.

What landed changed what `src/quant.tw` is. The f16 path was a hand-rolled
approximation that was wrong for subnormals; it is now one `.to(f16)` and exact.
bf16 was added, the int8 codes are centred into a real signed i8 tensor rather
than held as float64, and the accuracy cost of any of the three is measurable
today with `compare`. That is the half of the file's name that now works end to
end in principle.

What is still missing, and gates the size win:

  1. **The packed byte buffer.** raster NEEDS-111 names four native primitives
     (`buf_new`, `buf_len`, `buf_get8`, `buf_set8`); the twill side of it is
     written in raster's `src/buf.tw`. Until the runtime provides them, a narrow
     tensor rounds correctly and still fills eight-byte slots.
  2. **A narrow on-disk encoding.** The save format still writes eight bytes per
     element. A `Buf`-backed tensor written through a new value tag closes this,
     and by selvedge's compatibility rule that is a new archive format version,
     which is the right way round: the format is versioned so this can happen.

`stored_bytes` now takes a `realised` flag for exactly this gap: false is the
true footprint today, true is the footprint once the two pieces above land.

### 3. A monotonic clock

**Needs:** `mono_ms() -> I64`
**Used by:** `src/batcher.tw` (`max_hold`, which counts arrivals instead),
`src/score.tw` (`Progress`, which has no time estimate), `src/warmup.tw`
(which cannot report what it saved)
**Status:** not in the language. loom entry 16 and bobbin's first entry are the
same requirement.

This is the most damaging absence in this repository and it damages the batcher
most.

Every real dynamic batcher holds a batch for a duration: "up to 5ms". shuttle
counts arrivals, because there is no clock. The knob is therefore bounded in
requests rather than in time, which means under thin traffic a queued request
waits indefinitely for the next arrival and tail latency is unbounded. `flush`
exists for that and calling it is the caller's job. The consequence, stated in
the source and in the README: shuttle's hold is safe for a bounded workload and
unsafe for an open-ended stream of requests.

Second, warmup's whole justification is that it moves cost off the first
request, and shuttle cannot say how much cost. `src/warmup.tw` reports the
shapes it covered and refuses to claim a number, which is honest and is not what
anyone wants to read.

Third, a progress report on a four-hour scoring job is useful because of the
time remaining, and `src/score.tw` prints a percentage.

One primitive fixes all three. It is the same primitive three other repositories
in this ecosystem want.

### 4. Function values as parameters

**Needs:** a function type in an annotation, and a function value passed to a
systems-mode function
**Used by:** `src/predict.tw` (`predict`, `predict_batch`, `predict_stream`
take `forward`; `predict_stream` also takes `sink`), `src/batcher.tw` (`run`,
`flush`), `src/warmup.tw` (`warm`, `warm_with`), `src/score.tw` (`score`,
`score_to`, `evaluate`), `src/quant.tw` (`compare`)
**Status:** functions are values in numeric twill; whether a systems-mode
function may take one, and how the type is spelled, is not stated anywhere.

Every entry point in this library takes the forward function. shuttle writes
`forward: fn(Tree, Tensor) -> Tensor` and assumes that syntax.

This is not a convenience, in exactly the way loom's version of this entry is
not. The caller owning the forward pass is the design: the thing that makes a
model yours is its forward pass, and a serving library that hid it inside itself
would be a library you had to fork to serve anything it did not anticipate.
Without function parameters there is no way to hand one in and this repository
has no API.

The closure passed to `predict_stream` inside `src/score.tw` `score_to` needs
the stronger version, a function value that captures its environment.

### 5. Streaming file reads

**Needs:** a reader that yields part of a file, or `read_csv` with a row range
**Used by:** `src/score.tw` (`score_csv`)
**Status:** `read_file` returns the whole file; `read_csv` returns the whole
tensor. `std/io` says as much at the top of itself.

`score_csv` chunks the forward pass, which bounds the activations. It does not
bound the input, because the whole CSV is in memory before the first chunk runs.
A file larger than memory cannot be scored at all.

That is a real limit on the batch-scoring half of this library and it is the
half where the files are large. It is stated at the function rather than implied
by the word "streaming", which would otherwise be read as more than it is.

### 6. `Tree`, and a type for a parameter tree

**Needs:** a spelling for "a tensor, or a list or record nesting tensors"
**Used by:** `src/model.tw` (`Model.params`), and every function that takes a
forward function, since `Tree` is its first parameter
**Status:** the concept exists at runtime and has no name in the type language.
loom entry 2 and selvedge entry 9 are the same wall.

Systems mode makes annotations mandatory, so `Model` is currently undeclarable.

### 7. Multiple return values, or `Res[T, E]`

**Used by:** `src/model.tw` (`Loaded`), `src/predict.tw` (`Prediction`,
`Labels`), `src/batcher.tw` (`Batch`), `src/warmup.tw` (`Warmup`),
`src/quant.tw` (`Quantized`, `Comparison`), `src/score.tw` (`Score`,
`Evaluation`)
**Status:** `Res[T, E]` needs generics; multiple returns are not designed
anywhere.

Nine structs in this repository exist to return a value alongside an error
string. Not one of them is a type anyone wanted; each is a tuple with a name and
a constructor for its failure case.

The convention itself, an error string that is empty on success, is spool's and
loom's and it has their problem: the compiler does not make anyone read it. In a
serving library that is worse than it is in a trainer, because the value beside
the ignored error is a tensor of zeros and a caller who skips the check gets
predictions rather than a crash.

## Blocking: features the source assumes exist

### 8. Activation calibration, which needs the forward pass to be observable

**Needs:** a way to observe intermediate activations without owning the forward
function
**Used by:** `src/quant.tw`, which calibrates weights only
**Status:** not designed. It is not obviously a language feature at all.

Most of the accuracy loss in a real int8 deployment comes from quantising
activations, not weights, and activation ranges can only be calibrated by
running representative input and watching what comes out of each layer.

shuttle cannot, because the forward pass is deliberately the caller's and
shuttle never sees inside it. That is the right trade and this entry records
what it costs: `src/quant.tw` measures the weight half of the question honestly
and says nothing about the larger half.

The honest resolution may be that this is not shuttle's to solve, and that a
caller who wants activation quantisation instruments their own forward function.
The entry stays so that the gap is a decision rather than an omission.

### 9. Counting over a boolean tensor

**Needs:** `count(t)` over a comparison result, or `sum` accepting one
**Used by:** `src/quant.tw` (`f64_of_bool`, `compare`), `src/score.tw`
(`evaluate`)
**Status:** comparisons give a boolean tensor; `sum` wants numbers.

shuttle writes `where(t, 1.0, 0.0)` and sums that. Three call sites, one helper,
and it allocates a full-size float tensor to count a handful of disagreements.
Small, and it is the kind of thing that is in every accuracy computation anyone
will ever write, so it belongs in the language rather than in every library.

### 10. `std/bytes`

**Needs:** the byte-level helpers as a `std/` module
**Used by:** `src/bytes_compat.tw`, which is the third copy in the ecosystem
**Status:** twill has `src/bytes.tw` and it is not reachable from a package.

twill resolves a non-`std/` import as a path relative to the importing file, so
only `std/` modules are reachable from an installed package. selvedge carries a
copy, shuttle carries a copy, and twill's own is the original. The fix is not to
widen the import rule; it is `std/bytes`.

### 11. twill's terminal layer, reachable from a package

**Needs:** `src/term/` and `src/cli/progress.tw` available as `std/` modules
**Used by:** `src/score.tw` (`Progress`), which would call them and does not
**Status:** they exist, in the twill repository, as files.

Same wall as entry 10 and as loom's entry 8. `src/score.tw` prints plain
uncoloured lines with no bar and no width handling. Duplicating twill's progress
bar here was the obvious alternative and was rejected for loom's reason: two
progress bars in one ecosystem drift, and the drift is visible to users. Three
would be worse.

## Not blocking, but the source is worse without them

### 12. A test runner

**Would improve:** `tests/`
**Status:** none. `tests/harness.tw` is a hand-rolled counter and `report` calls
`exit(1)`.

`tests/harness.tw` is now the fourth identical copy of the same file across four
repositories. A `twill test` that collected `*_test.tw`, ran each in a fresh
interpreter and reported once would delete all four.

### 13. Concurrency, and a process interface

**Needs:** threads, or an event loop, or sockets
**Used by:** nothing, which is the point
**Status:** section 1.2 of the self-hosting design explicitly stops at "no
sockets".

shuttle has no network server and the README says so as its first limit rather
than leaving a reader to find out. This entry exists so the absence is recorded
as a known boundary rather than an oversight.

The narrower version worth wanting first is not sockets. It is concurrency: a
dynamic batcher's real value is overlapping the wait for a batch to fill with
the computation of the previous one, and without threads there is nothing to
overlap with. `src/batcher.tw` is a batcher in the sense of accumulating work
and cutting it into passes of a chosen size, which is the part that matters for
throughput, and it says so at the top.

### 14. `abs` and `round` over tensors, confirmed

**Would improve:** `src/quant.tw` (`simulate_f16`, `simulate_int8`)
**Status:** `abs` is listed as an elementwise builtin; `round` is not in the
table in the README.

`src/quant.tw` calls `round` on a tensor in both quantisation paths. If it does
not exist, or exists only for scalars, both are unwritable as they stand and the
fallback is `floor(x + 0.5)`, which is wrong at the halfway point in a way that
biases every weight upward by a fraction of a step. That bias is exactly what
the file's "round, do not truncate" comment argues against, so it would be a
silent regression rather than a compile error.

Confirming it is a documentation fix if the builtin is there and a small
addition if it is not.

### 15. Temporary files, and cleaning up after a test

**Would improve:** `tests/`
**Status:** there is no `remove_file`.

shuttle's tests avoid the filesystem entirely, which is why this entry is last
and short. It is here because the moment a test wants to exercise
`md.from_archive` against a real file it hits selvedge's entry 15 as well, and
neither repository can do anything about it.

### 16. An import that names a package rather than locating a file

**Needs:** a way to import a dependency by name
**Used by:** `src/model.tw`, which imports selvedge as `../../selvedge/src/...`
**Status:** twill resolves a non-`std/` import as a path relative to the
importing file, and there is no other form.

shuttle depends on selvedge, and the only way to say so in source is a relative
path that walks out of shuttle's own tree and into a sibling. That works because
spool vendors every package as a sibling under `twill_modules/`, which means
spool's directory layout is now hard-coded into shuttle's source: a package
manager that vendored one level deeper would break this file.

It also means the version constraint in `spool.toml` and the path in the source
are two independent statements of the same dependency, and nothing checks that
they agree.

This is the first cross-package dependency in the ecosystem, so it is the first
time the import rule has been asked to express one. loom, spool and selvedge all
depend on twill and on nothing else, which is why none of their needs files has
this entry.
