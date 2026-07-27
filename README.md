# VSTG
VSTG Skips Table Gets

A pure Python, zero dependency bloom filter, built so a check that would
otherwise hit your database on every request almost never has to.

The name is also short for *vestige*, a trace left behind by something
that passed through, which is what the underlying bit array actually is.

## The problem this solves

**Before:**

```python
def check_username(request):
  username = request.GET["username"]
  taken = User.objects.filter(username=username).exists()
  return JsonResponse({"available": not taken})
```

**After:**

```python
import vstg

def check_username(request):
  username = request.GET["username"]

  if not vstg.might_contain("usernames", username.encode("utf-8")):
    return JsonResponse({"available": True})

  taken = User.objects.filter(username=username).exists()
  return JsonResponse({"available": not taken})
```

Every username that was never registered returns without the database
being touched. The rare "maybe" still runs the exact same query as
before, so correctness never changes, only how often the expensive
path is needed.

## Install

Not yet published to PyPI. Until it is:

```bash
pip install git+https://github.com/vszlx4/vstg.git
```

Or, cloned and editable, for contributing:

```bash
git clone https://github.com/vszlx4/vstg.git
cd vstg
pip install -e '.[dev]'
```

Once published, this becomes `pip install vstg`. No dependencies are
ever installed alongside it either way.

## Usage

```python
from pathlib import Path
import vstg

vstg.init(Path("bloom_state"))
vstg.register("usernames", capacity=10_000_000, error_rate=0.001)

vstg.might_contain("usernames", username.encode("utf-8"))  # check
vstg.add("usernames", username.encode("utf-8"))            # record
```

`init` and `register` run once, at startup. `might_contain` and `add`
run everywhere else, for the life of the process.

## Customization

```python
vstg.register(
  "usernames",
  capacity=10_000_000,
  error_rate=0.001,
  policy=vstg.PersistPolicy(
    checkpoint_interval_seconds=300,
    checkpoint_insert_threshold=50_000,
    checkpoint_on_shutdown=True,
  ),
)

vstg.register(
  "email_verification_tokens",
  capacity=500_000,
  error_rate=0.01,
  shard_count=8,
)
```

Every named filter is independent, its own capacity, error rate, and
policy. `shard_count`, left unset, gives a single checkpoint file. Set
it and a checkpoint only rewrites the shards that actually changed.
Capacity, error rate, and policy are fixed the moment `register` is
called, for the rest of the process.

## The math

A filter is a bit array of length $m$, all bits starting at $0$.
Inserting a member sets $k$ of those bits, each position derived by
hashing the member. Checking a member reads the same $k$ positions: any
bit still $0$ proves the member was never inserted, since insertion
only ever sets bits. All $k$ bits set means probably inserted, since an
unrelated combination of other members could have set the same
positions by coincidence.

**Sizing**, given an expected item count $n$ (`capacity`) and a target
false positive probability $p$ (`error_rate`):

$$
m = -\frac{n \ln p}{(\ln 2)^2}
\qquad\qquad
k = \frac{m}{n} \ln 2
$$

`optimal_size` and `optimal_hash_count` in `core.py` compute exactly
these two values.

**Bit positions**, rather than running $k$ separate hash functions,
two hashes are computed once and every position derived from them
(Kirsch-Mitzenmacher):

$$
i_j = (h_1 + j \cdot h_2) \bmod m, \qquad j = 0, \dots, k-1
$$

**Resulting false positive rate**, after $n$ insertions into an
$m$-bit array with $k$ hash rounds:

$$
p \approx \left(1 - e^{-kn/m}\right)^k
$$

The sizing formulas above are exactly this expression solved for the
$m$ and $k$ that minimize it at the requested $n$ and $p$. Exceeding
the configured `capacity` degrades the real rate past what was
requested, since $n$ in this formula grows while $m$ and $k$ stay
fixed.

## Full API

- `BloomFilter` — the bit array, `add`, `might_contain`, in memory only.
- `ManagedBloomFilter` — a `BloomFilter` plus automatic checkpointing.
- `ShardedBloomFilter` — checkpointing partitioned across shard files.
- `BloomFilterRegistry` — the named collection `vstg.register` uses internally.
- `PersistPolicy` — checkpoint trigger configuration for all three above.

## Contributing

Two space indentation, `mypy --strict`, no runtime dependencies, a
docstring on every module, class, and function.

```bash
git clone https://github.com/vszlx4/vstg.git
cd vstg
pip install -e '.[dev]'
python -m unittest discover -s tests -v
```

## License

MIT, see [LICENSE](LICENSE).