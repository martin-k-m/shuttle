# What shuttle needs from twill

shuttle is written in twill and it runs: `twill test tests` passes six suites
and `twill run examples/serve.tw` loads a published model and answers with it.
This file is no longer the reason it does not run. It is the record of what this
library asked the language for, with the file and the function that needed each
one, what shuttle did in the meantime, and, for the ones that have since
arrived, whether shuttle has taken them up. An entry the language delivered and
shuttle has not wired up says exactly that, because "twill cannot" and "shuttle
has not" are different sentences and only one of them is a language work item.

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

## Was blocking: shuttle could not run at all without these

### 1. `mode systems` itself

**Used by:** every file
**Status:** DELIVERED in twill 1.6. Closed.

Nothing else on this list mattered until this did.

### 2. A narrow tensor storage: int8, float16 and bfloat16

**Needs:** a tensor whose elements are stored in fewer than eight bytes, and an
on-disk encoding for one
**Used by:** `src/quant.tw`, the whole file
**Status:** half landed. The dtype semantics are in the language now:
`docs/dtypes.md` in twill designs seven dtypes, `src/tensor.tw` stores each
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

  1. **The packed byte buffer.** twill NEEDS-111 names four native primitives
     (`buf_new`, `buf_len`, `buf_get8`, `buf_set8`); the twill side of it is
     written in twill's `src/buf.tw`. Until the runtime provides them, a narrow
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
**Status:** DELIVERED in twill 1.7, and shuttle has not taken it up. `mono_ns()`
returns a monotonic nanosecond count and `clock_now_ms()` a wall-clock
millisecond one; both were checked against the 1.7.1 binary. No file under
`src/` calls either.

So all three consequences below still hold and none of them is twill's fault any
more. Two of the three are now small changes: `src/warmup.tw` can time its
passes and `src/score.tw` can extrapolate a remaining time, and neither needs
anything from the language.

The batcher is the one that is still hard, and the reason is entry 13 rather
than this one. A hold that expires after 5 ms needs something to notice the
expiry, and with no concurrency nothing in this library runs between one
`submit` and the next, so a deadline can only ever be checked when the next
request arrives. That is what counting arrivals already does. A duration-bounded
hold needs either a caller-driven tick, which is an API decision shuttle has not
made, or concurrency, which is entry 13.

Every real dynamic batcher holds a batch for a duration: "up to 5ms". shuttle
counts arrivals. The knob is therefore bounded in
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

That primitive exists now. Two of the three are shuttle's to write and the
third needs a design decision first.

### 4. Function values as parameters

**Needs:** a function type in an annotation, and a function value passed to a
systems-mode function
**Used by:** `src/predict.tw` (`predict`, `predict_batch`, `predict_stream`
take `forward`; `predict_stream` also takes `sink`), `src/batcher.tw` (`run`,
`flush`), `src/warmup.tw` (`warm`, `warm_with`), `src/score.tw` (`score`,
`score_to`, `evaluate`), `src/quant.tw` (`compare`)
**Status:** DELIVERED, including the closure. A systems-mode function takes a
function value, the type is spelled `fn(A, B) -> C`, and the closure
`score_to` passes to `predict_stream` captures its environment and runs.
`tests/score_test.tw` and `tests/predict_test.tw` cover both. Closed.

Every entry point in this library takes the forward function. shuttle writes
`forward: fn(Tree, Tensor) -> Tensor` and that is the syntax.

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
**Status:** partly DELIVERED, and not usable for this. twill 1.7.1 has
`read_file_at(path, offset, length)` and `file_size(path)`, so a byte range of a
file is readable now. `read_csv` still returns the whole tensor and there is no
row range, so `score_csv` would have to find its own line boundaries inside a
byte window and parse the rows itself, which is a CSV reader in this repository.
What is wanted is still `read_csv` with a row range, or a reader that yields
rows.

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
**Status:** DELIVERED. `Tree` is the name, `Model.params` is declared with it,
and every entry point that takes a forward function runs. Closed. loom entry 2
and selvedge entry 9 closed the same way.

### 7. Multiple return values, or `Res[T, E]`

**Used by:** `src/model.tw` (`Loaded`), `src/predict.tw` (`Prediction`,
`Labels`), `src/batcher.tw` (`Batch`), `src/warmup.tw` (`Warmup`),
`src/quant.tw` (`Quantized`, `Comparison`), `src/score.tw` (`Score`,
`Evaluation`)
**Status:** DELIVERED in 1.6, and still not taken up. This is the entry that
cost something.

1.6 shipped `Res[T, E]`, `Opt[T]` and postfix `?`, so the language side of this
entry is closed. The nine structs are still here, because converting them
changes every public return type in the repository at once and that is a release
of its own rather than a tidy-up. The order to do it in is `src/model.tw` first,
since `Loaded` is the one a caller meets before anything else works.

selvedge did the conversion and shuttle did not, and the seam between them broke
without anything noticing: `src/model.tw` `read_unverified` went on reading an
`err` field off `arc.read`, which had become a `Res`, so every archive load
failed at runtime while `twill check` and the whole suite stayed green. It was
found by running `examples/serve.tw`. The fix is a `match`, and
`tests/model_test.tw` now loads an archive so the seam is covered.

Nine structs in this repository exist to return a value alongside an error
string. Not one of them is a type anyone wanted; each is a tuple with a name and
a constructor for its failure case.

The convention itself, an error string that is empty on success, is spool's and
loom's and it has their problem: the compiler does not make anyone read it. In a
serving library that is worse than it is in a trainer, because the value beside
the ignored error is a tensor of zeros and a caller who skips the check gets
predictions rather than a crash.

## Was blocking: features the source assumed exist

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
**Status:** DELIVERED, and taken up. `equal(a, b)` yields a 0/1 tensor that
`sum` adds directly, which is what `q.compare` counts argmax disagreements with;
`tests/quant_test.tw` and the example both exercise it. Closed.

shuttle writes `where(t, 1.0, 0.0)` and sums that. Three call sites, one helper,
and it allocates a full-size float tensor to count a handful of disagreements.
Small, and it is the kind of thing that is in every accuracy computation anyone
will ever write, so it belongs in the language rather than in every library.

### 10. `std/bytes`

**Needs:** the byte-level helpers as a `std/` module
**Used by:** nothing, now
**Status:** RESOLVED in 1.6.

twill resolved it the way this entry asked, by making the helpers reachable
rather than by widening the import rule: `bytes_new`, `bytes_push` and
`bytes_to_str` are builtins, and `std/text` carries `push_str`, `find`,
`slice`, `starts_with` and `join` with the same semantics the copy had.
`src/bytes_compat.tw` is deleted and its callers import `std/text`. The one
function with no `std/` equivalent, `push_hex_byte`, had no call sites left and
went with it; a digest that needs hex again should ask for `std/hex` rather
than start a fourth copy. selvedge's byte-identical copy can go the same way.

### 11. twill's terminal layer, reachable from a package

**Needs:** `src/term/` reachable from a package
**Used by:** `src/score.tw` (`Progress`), which now calls it
**Status:** DELIVERED for colour and capability detection, and taken up, but not
in the way recorded below. The terminal layer is under `std/`:
`std/term/caps`, `std/term/ansi` and `std/term/theme`, which `src/score.tw`
imports directly. Nothing is vendored and the import rule did not change. There
is no `std/cli`, so the rate-and-ETA bar this entry wanted does not exist to
adopt; building one here needs the clock from entry 3, which shuttle now has and
does not call.

The older account, kept because it was wrong and the correction is the point:

Resolved the same way as loom's entry 8: twill's terminal modules import each
other by a path relative to the importer, so `src/score.tw` vendors the palette
and capability detection under `twill_modules/` and lights its progress line,
dropping to plain text the moment the output is piped. `src/cli/progress.tw`'s
stateful bar is still not adopted, and the reason is unchanged: a second (or
third) progress bar in the ecosystem drifts. shuttle keeps its own stateless
line, now lit from the shared palette so it never drifts in colour.

## Not blocking, but the source is worse without them

### 12. A test runner

**Would improve:** `tests/`
**Status:** DELIVERED. `twill test tests` collects `*_test.tw`, runs each in a
fresh interpreter and reports once. CI calls it and so does the README.
`tests/harness.tw` stays, because the runner names the file that failed and the
harness names the assertion inside it; deleting the three copies across three
repositories wants a `std/test`.

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
**Status:** CONFIRMED on twill 1.7.1. Both exist over tensors, and `round` is
half away from zero: `round([0.5, 1.5, -0.5, 2.4])` gives `[1, 2, -1, 2]`.
Closed.

That is the behaviour `src/quant.tw` wanted. The `floor(x + 0.5)` fallback this
entry feared, which biases every weight upward by a fraction of a step at the
halfway point, is not needed and should not be written.

What is still not written down is the tie rule itself, in twill's own
documentation. It was established here by running it, and a behaviour a caller
has to discover by experiment is a behaviour that can change without anyone
calling it a break.

### 15. Temporary files, and cleaning up after a test

**Would improve:** `tests/`
**Status:** RESOLVED in 1.6.

1.6 has `temp_dir(prefix)`, `remove_file`, `remove_all`, `mkdir_all` and the
rest. Nothing in `tests/` changed for it, because shuttle's tests still touch
the filesystem nowhere and so leave nothing behind. The entry is kept, marked
resolved, because the constraint it recorded was real: a test that wants to
exercise `md.from_archive` against a real file can now be written and cleaned
up after.

### 16. An import that names a package rather than locating a file

**Needs:** a way to import a dependency by name
**Used by:** `src/model.tw`, which imports selvedge as `../../selvedge/src/...`
**Status:** unchanged in 1.7.1. twill resolves a non-`std/` import as a path
relative to the importing file, and there is no other form. This is the one
entry in this file that twill 1.7 did not move at all, and it is the reason the
CI workflow clones selvedge into `../selvedge` before it can run the tests.

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
