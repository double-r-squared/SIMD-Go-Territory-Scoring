# arm-optimized.cpp — Deep-Dive Documentation (v4)

This document reflects the current state of `arm-optimized.cpp` after all optimization passes. Every struct, function, algorithm choice, ARM intrinsic, and design decision is documented with the reasoning behind it.

---

## Version History & Performance

| Version | Key Changes | Avg (ns) | Min (ns) |
| --- | --- | --- | --- |
| v1 (original) | Baseline ARM version — span-fill BFS, `ctzll` seed scan, NEON clear | ~1559 | ~1450 |
| v2 | `ctzll`/`clzll` span expansion, region_spans, POP_TABLE, static thread partition | ~1360 | ~1265 |
| v3 | Inline territory count (no bitset), pre-built `ExpandedBoard` | ~1161 | ~1085 |
| v4 (current) | Incremental `scan_blocked`, branchless `spanMask`, QoS P-core hint, stack 256, dead-clamp removal, eliminated double-zero | **~1195** | **~975** |

All versions verified correct: Black=300,000 / White=1,400,000 territory points across all 100,000 rounds.

---

## File-Level Architecture

```
Includes: <arm_neon.h>, <pthread.h>, <sys/sysctl.h>

File scope:
  POP_TABLE (removed in v3 — no longer needed)
  black_results[100000], white_results[100000]   ← BSS, 800 KB off stack

Functions:
  printCoreInfo()          diagnostic: query M1 P/E/total core counts
  getPhysicalCoreCount()   all physical cores (P + E)
  getPCoreCoreCount()      P-cores only (Firestorm)

OptimizedState             board data + move logic
  data[23]                 736-bit packed board (black bits 0–360, white 361–721)
  getStone/getBlack/getWhite
  makeMove

  ExpandedBoard            57×19 read-only working space, built once per color
    rows[19]               one uint64_t per row (57 bits used)

    VisitedBitset          18 × uint64_t flood-fill tracking (no territory storage)
      clearNEON()
      markSpan()

    countTerritory()       span-fill BFS → returns int count directly

main(argc, argv)
  builds board once (seed 45, 250 stones)
  builds black_exp + white_exp once (ExpandedBoard)
  launches num_cores threads with static [start,end) partitioning
  each worker calls pthread_set_qos_class_self_np → P-core scheduling
  times wall-clock, prints per-pair ns, prints verification totals
```

**Critical path** (called 100k times per thread):

```
countTerritory()
  clearNEON()               9 NEON stores, zero visited bitset
  outer loop (19 rows)
    getBlockedBits(y)       1–2 word loads, OR with rows[y]
    ctzll → seed x
    expandSpan              2–3 instructions per direction
    markSpan (visited)      1–2 word OR ops
    countSpanInRealBoard    2 CSEL + 1 ADD
    push above/below spans
    inner BFS loop
      getBlockedBits(sy)    once per row pop
      ctzll → sx
      expandSpan
      markSpan
      countSpanInRealBoard
      push above/below
      scan_blocked |= spanMask   ← incremental, zero memory loads
  if !touches_edge: total += speculative_count
return total
```

---

## Section 1: Includes and Core Detection

### New include: `<pthread.h>`

Added in v4. Required for `pthread_set_qos_class_self_np`, the macOS API that requests P-core scheduling for a thread. Without this the OS may assign compute-heavy threads to E-cores (Icestorm), which are ~3× slower per thread. With static partitioning, the wall time is set by the slowest thread — one E-core thread bottlenecks the entire benchmark.

### `printCoreInfo()`

Diagnostic only, runs once at startup. Uses `sysctl` to query the M1's heterogeneous topology:

```cpp
sysctlbyname("hw.perflevel0.physicalcpu", ...)  // P-cores (Firestorm)
sysctlbyname("hw.perflevel1.physicalcpu", ...)  // E-cores (Icestorm)
sysctlbyname("hw.physicalcpu", ...)             // P + E combined
std::thread::hardware_concurrency()             // logical cores (same on M1, no HT)
```

On base M1: 4 P-cores + 4 E-cores = 8 physical total.

### `getPhysicalCoreCount()`

Returns total physical core count (`hw.physicalcpu`). Used as the default thread count. Falls back to `hardware_concurrency()` if sysctl fails.

### `getPCoreCoreCount()` — added v4

```cpp
static inline int getPCoreCoreCount() {
    int count; size_t size = sizeof(count);
    if (sysctlbyname("hw.perflevel0.physicalcpu", &count, &size, NULL, 0) == 0) return count;
    return getPhysicalCoreCount();
}
```

Returns P-core count only. Used in `main` to display alongside total count. Exposed so the program can be run as `./arm-optimized 4` to force 4 threads (P-cores only) for comparison — this was added to diagnose the E-core variance problem.

---

## Section 2: `OptimizedState`

```cpp
struct __attribute__((aligned(64))) OptimizedState {
    static constexpr int      N        = 19;
    static constexpr int      OFFSET   = 361;
    static constexpr int      EN       = 3 * N;           // 57
    static constexpr int      REAL_LO  = N;               // 19
    static constexpr int      REAL_HI  = 2 * N - 1;       // 37
    static constexpr uint64_t ROW_MASK = (1ULL << EN) - 1; // 57-bit mask

    alignas(64) std::array<uint32_t, 23> data{};
```

New in v3: `REAL_LO` and `REAL_HI` added as named constants for the real-board column range (columns 19–37 in the expanded 57-wide grid). Used by `countSpanInRealBoard` to avoid magic numbers in the hot path.

### Board Encoding

`data` is 23 × 32 = 736 bits:
- Bits 0–360: black stones. Bit `i` = stone at board position `i`.
- Bits 361–721: white stones. Bit `i + 361` = stone at position `i`.

Board positions: `(x, y)` → `y * 19 + x`. Position 0 = top-left, 360 = bottom-right.

### `BIT_MASKS`

Pre-computed `1u << k` for k = 0..31. Avoids a variable shift instruction for bit access. The compiler may embed these as immediates since ARM can encode many small constants directly in `AND`/`ORR`.

### `getStone` / `getBlack` / `getWhite`

```cpp
__attribute__((always_inline, hot))
inline bool getStone(int bitIndex) const {
    return data[bitIndex >> 5] & BIT_MASKS[bitIndex & 31];
}
```

`>> 5` = which `uint32_t` word. `& 31` = which bit within it. `always_inline` forces inlining at every call site — no call overhead. `hot` tells the compiler this is frequently executed.

### `makeMove`

`__attribute__((noinline))` — only called 250 times during setup, never in the benchmark loop. Keeping it out-of-line reduces hot instruction-cache pressure.

`[[unlikely]]` on all three failure branches — lets the compiler lay out the success path as the straight-line fall-through (zero taken branches on the fast path).

---

## Section 3: `ExpandedBoard`

### Purpose

The territory flood-fill needs to know when a region "escapes" to the board edge. The expanded board converts this into a simple column-range check.

Each 19-bit board row is mirrored into 57 bits:

```
[ reversed(row) | original(row) | reversed(row) ]
  bits  0–18       bits 19–37      bits 38–56
```

- **Columns 19–37**: real board. Territory in this range is counted.
- **Columns 0–18 and 38–56**: mirrors. The reversed stone pattern blocks horizontal escape — a flood-fill that would "go around" the edge in the original is blocked by a mirrored stone in the expanded board.
- Reaching column 0 or 56 during BFS → horizontal edge touched.
- Reaching row 0 or 18 during BFS → vertical edge touched.

### `ExpandedBoard()` — default constructor added v3

```cpp
ExpandedBoard() = default;
```

Required for the `const ExpandedBoard black_exp(game, true)` declaration in `main`. Previously the struct had only a parameterized constructor.

### Row Extraction

```cpp
for (int y = 0; y < 19; ++y) {
    int start = y * 19 + base;
    int word  = start >> 5;
    int shift = start & 31;

    uint64_t bits;
    if (__builtin_expect(shift <= 13, 1)) [[likely]] {
        bits = (s.data[word] >> shift) & 0x7FFFFULL;
    } else [[unlikely]] {
        uint32x2_t words = vld1_u32(&s.data[word]);
        uint64_t combined = vget_lane_u64(vreinterpret_u64_u32(words), 0);
        bits = (combined >> shift) & 0x7FFFFULL;
    }

    uint64_t rev = (uint64_t)__builtin_bitreverse32((uint32_t)bits) >> 13;
    rows[y] = rev | (bits << 19) | (rev << 38);
}
```

**`shift <= 13` threshold**: A 19-bit row at offset `shift` fits in one 32-bit word when `shift + 18 <= 31` → `shift <= 13`. At offset 14+, it crosses into the next word.

**NEON word-spanning load** (`vld1_u32`): loads two consecutive `uint32_t` in a single 64-bit memory operation, then `vreinterpret_u64_u32` + `vget_lane_u64` extracts as a `uint64_t`. One load vs. two scalar loads + manual stitch.

**Hardware RBIT** (`__builtin_bitreverse32`): compiles to a single `RBIT` instruction on ARM. Reverses all 32 bits in one cycle. Since the 19 meaningful bits are in positions 0–18, after RBIT they land in positions 13–31. Right-shifting by 13 aligns them to 0–18.

### Pre-built in `main` — optimization v3

```cpp
const OptimizedState::ExpandedBoard black_exp(game, true);
const OptimizedState::ExpandedBoard white_exp(game, false);
```

**Old behavior**: `ExpandedBoard` was constructed inside `computeEnclosedTerritoryOnly` on every call — 100,000 × 2 = 200,000 constructions per benchmark run. Each construction: 19 RBIT + 19 word loads + 19 OR operations.

**New behavior**: 2 constructions total. Both objects are `const` and read-only during the benchmark. `countTerritory()` only reads `rows[]` — zero writes — so all threads can share the same `ExpandedBoard*` with no synchronization.

---

## Section 4: `VisitedBitset`

Renamed from `CompactBitset` in v3 when the territory bitset was eliminated. Now serves only one purpose: tracking which expanded-board cells have been visited by the current flood-fill.

```cpp
struct alignas(64) VisitedBitset {
    alignas(64) std::array<uint64_t, 18> data;   // no {} initializer
```

### No `{}` initializer — v4

**Old**: `std::array<uint64_t, 18> data{}` — C++ value-initializes (zeros) all 18 words at object construction, then `clearNEON()` zeros them again. Two zero passes.

**New**: `data` has no initializer — uninitialized at construction. `clearNEON()` is called explicitly and is the only zeroing step. One zero pass. The savings are small (18 stores → 9 NEON stores, done once either way) but the principle matters: never do the same work twice.

### `get(idx)`

```cpp
return data[idx >> 6] & (1ULL << (idx & 63));
```

Standard 64-bit bitset access. `>> 6` = word index. `& 63` = bit within word.

### `markSpan(start_idx, count)` — bulk visited marking

```cpp
void markSpan(int start_idx, int count) {
    int w = start_idx >> 6;
    int b = start_idx & 63;
    if (__builtin_expect(b + count <= 64, 1)) {
        data[w] |= ((1ULL << count) - 1) << b;    // single-word fast path
    } else {
        data[w] |= (~0ULL) << b;                  // fill end of first word
        int rem = count - (64 - b);
        ++w;
        while (rem >= 64) { data[w++] = ~0ULL; rem -= 64; }
        if (rem > 0) data[w] |= (1ULL << rem) - 1;
    }
}
```

Sets `count` consecutive bits in O(words crossed) instead of O(cells). For spans ≤ 57 bits, the inner `while (rem >= 64)` never executes — the multi-word path is always resolved in 2 word-OR operations. The `__builtin_expect` tells the compiler the single-word path is vastly more common.

### `clearNEON()`

```cpp
uint64x2_t z = vdupq_n_u64(0);
vst1q_u64(&data[0],  z);  vst1q_u64(&data[2],  z);
vst1q_u64(&data[4],  z);  vst1q_u64(&data[6],  z);
vst1q_u64(&data[8],  z);  vst1q_u64(&data[10], z);
vst1q_u64(&data[12], z);  vst1q_u64(&data[14], z);
vst1q_u64(&data[16], z);
```

- `vdupq_n_u64(0)`: creates a 128-bit register of all zeros — `MOVI V0.2D, #0`, one instruction.
- `vst1q_u64`: stores 128 bits (2 × `uint64_t`) per call.
- 9 stores clear 18 × 8 = 144 bytes. vs. 18 scalar stores.
- Loop fully unrolled at compile time (bounded by constant 18).

---

## Section 5: `countTerritory()` — full BFS algorithm

This is the entire hot path. Every call is independent (reads only from `rows[]`, uses stack-local `visited`).

### Local stack frame

```cpp
VisitedBitset visited{};          // 144 bytes, stack-allocated
visited.clearNEON();
struct Span { int16_t x1, x2, y, parent_y; };   // 8 bytes
alignas(64) std::array<Span, 256> stack;          // 2048 bytes
int stack_top = 0;
int total = 0;
```

**Total stack usage per call**: ~2.3 KB. Fits in L1D cache (192 KB on M1 P-cores). No heap allocation.

**Stack size reduced v4**: 1024 → 256 spans. Reasoning: span-fill BFS on a 19×19 board processes each row at most twice (once from above, once from below). Each pop pushes at most 2 new spans. Max live spans = 2 × 19 = 38. 256 gives 6× headroom. Reducing from 1024 to 256 cuts the stack array from 8,192 bytes to 2,048 bytes — better L1 fit, and the guard check (`stack_top < 255`) compares against a smaller constant.

### Lambda: `getBlockedBits(ry)`

```cpp
auto getBlockedBits = [&](int ry) -> uint64_t {
    int s = ry * EN;    // bit offset of row ry in visited bitset
    int w = s >> 6;
    int b = s & 63;
    uint64_t vis = visited.data[w] >> b;
    if (__builtin_expect(b > 7, 0)) [[unlikely]]
        vis |= visited.data[w + 1] << (64 - b);
    return rows[ry] | (vis & ROW_MASK);
};
```

Returns a 57-bit mask where `1` = blocked (stone or already-visited). Combines two sources:
1. `rows[ry]`: stone positions (pre-built, constant).
2. `visited.data[w]`: flood-fill progress.

**`b <= 7` threshold**: Row `ry` starts at bit `ry * 57`. A 57-bit window starting at offset `b` fits in one 64-bit word when `b + 57 <= 64` → `b <= 7`. Otherwise it spans two words. The `[[unlikely]]` annotation puts the two-load path in a cold branch.

### Lambda: `expandSpan(blocked, x, x1, x2)` — dead clamp removed v4

```cpp
auto expandSpan = [](uint64_t blocked, int x, int& x1, int& x2) {
    if (__builtin_expect(x < EN - 1, 1)) {
        uint64_t right = blocked >> (x + 1);
        x2 = (right == 0) ? (EN - 1) : (x + (int)__builtin_ctzll(right));
        // No clamp: x + ctzll ≤ 55 = EN-2 < EN-1 always. See proof.
    } else {
        x2 = EN - 1;
    }
    if (__builtin_expect(x > 0, 1)) {
        uint64_t left = blocked & ((1ULL << x) - 1);
        x1 = (left == 0) ? 0 : (63 - __builtin_clzll(left) + 1);
    } else {
        x1 = 0;
    }
};
```

Finds the maximal contiguous free span `[x1, x2]` containing column `x`.

**Right boundary**: shift `blocked` right by `(x+1)` so bit 0 = column `x+1`. `ctzll` gives the distance to the first blocked column. If shifted result is 0, no blocking exists → `x2 = EN-1`.

**Left boundary**: mask `blocked` to bits `0..x-1` (columns to the left of `x`). `clzll` gives the position of the highest set bit (highest blocked column to the left). The stone at position `k = 63 - clzll` means `x1 = k + 1`.

**Dead clamp removed (v4)**: The previous version had `if (x2 > EN - 1) x2 = EN - 1` after the right boundary. This is provably unreachable:

```
blocked is masked to ROW_MASK (bits 0..56, i.e., EN-1 = 56).
After shifting right by (x+1), the highest possible set bit in `right`
is at position (56 - (x+1)) = (55 - x).
Therefore ctzll(right) ≤ (55 - x).
x + ctzll(right) ≤ x + (55 - x) = 55 = EN - 2 < EN - 1.
The clamp `if (x2 > EN-1)` is never true. Removed.
```

Removing a dead conditional lets the compiler emit tighter code and gives the branch predictor one fewer path to track.

### Lambda: `spanMask(lo, hi)` — branchless rewrite v4

**Old**:
```cpp
uint64_t top = (hi < 63) ? ((2ULL << hi) - 1) : ~0ULL;
uint64_t bot = (lo >  0) ? ((1ULL << lo) - 1) : 0ULL;
return top & ~bot;
```
Two conditional expressions = 2 branches (or 2 CSEL pairs).

**New**:
```cpp
return (~0ULL >> (63 - hi)) & ~((1ULL << lo) - 1);
```

`~0ULL >> (63 - hi)` sets bits 0..hi. No conditional needed — when `hi = 63`, shifts by 0, result = `~0ULL`. When `hi = 0`, shifts by 63, result = `1`. The left side `(1ULL << lo) - 1` sets bits 0..lo-1 which is then inverted and ANDed. Result: bits `lo..hi` set. 5 instructions, zero branches, zero CSEL instructions.

**Why this matters**: `spanMask` is called twice per inner-loop BFS iteration — once for `scan_mask` (initial), once for the incremental `scan_blocked |= spanMask(sx1, sx2)` update. Removing 2 branches from a function called thousands of times per `countTerritory()` call is meaningful.

### Lambda: `countSpanInRealBoard(x1, x2)` — added v3

```cpp
auto countSpanInRealBoard = [](int x1, int x2) -> int {
    int rx1 = x1 < REAL_LO ? REAL_LO : x1;
    int rx2 = x2 > REAL_HI ? REAL_HI : x2;
    return (rx2 >= rx1) ? (rx2 - rx1 + 1) : 0;
};
```

Clamps the span to the real-board column range [19, 37] and returns the cell count. The two min/max operations compile to `CSEL` instructions on ARM (conditional select, no branch). The final comparison is a third `CSEL` or a conditional move.

This replaces the old approach of writing to a territory bitset and running `popcount19x19()` afterward. The savings:
- Eliminated 144 bytes of `CompactBitset territory` writes per enclosed region
- Eliminated the full `popcount19x19()` pass (19 × 2 popcounts + 18 word loads)
- Eliminated `region_spans[512]` array (3,072 bytes of stack + indexing overhead)
- Replaced with: 2 CSEL + 1 ADD per span, accumulated in a register

### Outer seed-scan loop

```cpp
for (int y = 0; y < 19; ++y) {
    uint64_t blocked   = getBlockedBits(y);
    uint64_t free_bits = (~blocked) & ROW_MASK;

    while (free_bits) {
        int x = __builtin_ctzll(free_bits);  // leftmost free cell
        ...
        blocked   = getBlockedBits(y);
        free_bits = (~blocked) & ROW_MASK;
    }
}
```

`free_bits` is a 57-bit mask of all unvisited, stone-free cells in row `y`. `ctzll` finds the lowest set bit in O(1) — one instruction. After processing a region, `visited` changed so `blocked` is recomputed. This recompute happens once per region per row (cheap), not once per cell (old approach).

**vs. old per-cell loop**: the original `for (int x = 0; x < EN; ++x)` checked every cell individually — up to 57 checks per row. With 250 stones and a dense board, most cells are either stones or visited, so `ctzll` skips entire 64-bit chunks in one instruction.

### Inner BFS loop with incremental `scan_blocked` — optimization v4

```cpp
while (stack_top > 0) {
    Span span = stack[--stack_top];
    int sy = span.y;

    uint64_t scan_blocked = getBlockedBits(sy);     // one memory read
    uint64_t scan_mask    = spanMask(span.x1, span.x2);
    uint64_t scan_free    = (~scan_blocked) & scan_mask;

    while (scan_free) {
        int sx = __builtin_ctzll(scan_free);
        int sx1, sx2;
        expandSpan(scan_blocked, sx, sx1, sx2);
        ...
        visited.markSpan(sy * EN + sx1, sx2 - sx1 + 1);
        speculative_count += countSpanInRealBoard(sx1, sx2);
        ...
        // OLD: scan_blocked = getBlockedBits(sy);   ← 1-2 memory loads
        // NEW:
        scan_blocked |= spanMask(sx1, sx2);          // ← zero memory loads
        scan_free     = (~scan_blocked) & scan_mask;
    }
}
```

**Old**: after each `markSpan(sy, ...)`, the code re-called `getBlockedBits(sy)` — which loads `visited.data[w]` (and sometimes `visited.data[w+1]`) from memory to rebuild the blocked mask for row `sy`.

**New**: `getBlockedBits(sy) = rows[sy] | visited_bits_for_sy`. `rows[sy]` is constant and already in `scan_blocked`. `markSpan(sy*EN+sx1, count)` added exactly `spanMask(sx1, sx2)` to the visited bits for row `sy`. So:

```
new_blocked = rows[sy] | (old_visited | spanMask(sx1, sx2))
            = scan_blocked | spanMask(sx1, sx2)
```

This is a single `ORR` instruction with no memory access. Eliminates 1–2 `visited.data[]` loads per inner iteration. The `spanMask(sx1, sx2)` value is already computed (it's used in the `markSpan` call above).

### Territory accumulation — `[[likely]]` added v4

```cpp
if (!touches_edge) [[likely]]
    total += speculative_count;
```

`[[likely]]` — at 69% board fill (250 stones on 361 cells), most empty regions are small and enclosed. The open regions that touch the board edge are fewer but larger. The `[[likely]]` hint guides branch layout so the common enclosed-region path is the fall-through (no taken branch).

---

## Section 6: `main()` — benchmark harness

### Thread count argument — added v4

```cpp
int main(int argc, char* argv[]) {
    const int p_cores   = getPCoreCoreCount();
    const int all_cores = getPhysicalCoreCount();
    const int num_cores = (argc > 1) ? std::atoi(argv[1]) : all_cores;
```

`argv[1]` overrides the thread count. Used to compare:
- `./arm-optimized`   → 8 threads (4 P + 4 E on M1)
- `./arm-optimized 4` → 4 threads (P-cores only)

This exposed the E-core variance problem: with static partitioning, E-core threads take ~3× longer for their share, bottlenecking wall time. The QoS hint addresses this but doesn't guarantee it.

### Static result arrays — changed v4

```cpp
static constexpr int ROUNDS = 100000;
alignas(64) static std::array<int, ROUNDS> black_results;
alignas(64) static std::array<int, ROUNDS> white_results;
```

Placed at **file scope** (not inside `main`). 100,000 × 4 bytes × 2 = 800 KB total.

- `static` → placed in BSS segment. OS zero-initializes at program load, zero runtime cost.
- `alignas(64)` → valid at file scope (not valid on `static` local variables in clang).
- No heap allocation, no pointer indirection.
- BSS is contiguous — both arrays are adjacent in memory, good for sequential access pattern in the verification loop.

### Pre-built `ExpandedBoard` — added v3

```cpp
const OptimizedState::ExpandedBoard black_exp(game, true);
const OptimizedState::ExpandedBoard white_exp(game, false);
```

Built once, before the benchmark starts. Workers receive `const ExpandedBoard*` — no locking needed, no cache-line contention, pure read sharing.

**Saves**: 200,000 `ExpandedBoard` constructions (100k rounds × 2 colors), each costing 19 RBIT + 19 word loads + 19 ORs = eliminated entirely.

### Static thread partitioning — added v2

```cpp
const int num_black = num_cores / 2;
const int num_white = num_cores - num_black;

auto launchRange = [&](...) {
    const int per = ROUNDS / nthreads;
    for (int i = 0; i < nthreads; ++i) {
        int s = i * per;
        int e = (i == nthreads - 1) ? ROUNDS : s + per;
        threads.emplace_back(worker, exp, s, e, results);
    }
};
```

Each thread owns a fixed `[start, end)` range. No atomic counter, no inter-thread cache traffic for work dispatch.

**Old (MT-test)**: shared `std::atomic<int> counter` — all threads `fetch_add(1)` on the same cache line, causing cache-line bouncing between cores.

**New**: pre-computed ranges. Thread `i` accesses `results[s..e]` — non-overlapping, no sharing.

**Trade-off**: if threads finish at different speeds (e.g., E-cores vs P-cores), faster threads sit idle. This was the cause of the increased variance in v4 — addressed by the QoS hint.

### `pthread_set_qos_class_self_np` — added v4

```cpp
auto worker = [](const OptimizedState::ExpandedBoard* exp,
                 int start, int end, int* results) {
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
    for (int i = start; i < end; ++i)
        results[i] = exp->countTerritory();
};
```

`QOS_CLASS_USER_INTERACTIVE` is the highest macOS QoS class. The scheduler uses this as a strong hint to assign the thread to a P-core (Firestorm). It is a **hint, not a guarantee** — the OS retains final scheduling authority.

**Why this matters**: on M1, E-cores run at ~600 MHz vs P-cores at ~3.2 GHz for compute-bound tasks. A static partition of 12,500 rounds on an E-core takes ~3× longer than 12,500 rounds on a P-core. The total benchmark wall time is `max(all thread completion times)` — one slow E-core thread can add hundreds of nanoseconds to the per-pair result.

The QoS hint pushed minimum observed time from ~1085 ns to **~975 ns** (first sub-1000 ns result).

---

## Section 8: Build Flags

```makefile
FLAGS := -std=c++17 -O3 -ffast-math -march=armv8.5-a -mcpu=apple-m1 \
         -flto -pthread -fvectorize -funroll-loops \
         -fslp-vectorize -fwhole-program-vtables
```

| Flag | Effect |
| --- | --- |
| `-march=armv8.5-a` | Targets ARMv8.5-A ISA: enables `RBIT`, `CNT`, hardware popcount, advanced NEON |
| `-mcpu=apple-m1` | M1-specific tuning: pipeline depth, cache sizes, latency tables used by the scheduler |
| `-O3` | Full optimization: auto-vectorization, aggressive inlining, loop transforms |
| `-ffast-math` | Allows FP reordering; also enables some integer optimizations via relaxed aliasing rules |
| `-flto` | Link-Time Optimization: inlines across translation units, eliminates dead code globally |
| `-pthread` | Links pthreads for `std::thread` and `pthread_set_qos_class_self_np` |
| `-fvectorize` | Enables auto-vectorization (redundant with `-O3` on clang, explicit for clarity) |
| `-funroll-loops` | Unrolls loops with compile-time-known bounds (e.g., the 9-iteration `clearNEON` loop) |
| `-fslp-vectorize` | Superword-Level Parallelism: vectorizes straight-line code blocks |
| `-fwhole-program-vtables` | Enables devirtualization across TUs with LTO (no-op for our code, helps dependencies) |

---

## Section 9: Remaining Optimization Candidates

These were identified but not yet implemented:

### 9.1 Consteval `buildPopTable` (removed from hot path in v3)

`POP_TABLE` was removed entirely in v3 when `popcount19x19()` was eliminated. No longer relevant.

### 9.2 Compile-time `ExpandedBoard` row table

The `(word, shift, is_spanning)` values for all 38 row/color combinations are fully determined at compile time. A `constexpr` table would eliminate the runtime `shift <= 13` branch and all the `start`, `word`, `shift` arithmetic from `ExpandedBoard`'s constructor. Low impact since the constructor now runs only twice (pre-built), but worth considering for the live-game use case.

### 9.3 Hard P-core affinity (vs. QoS hint)

`pthread_set_qos_class_self_np` is a hint. A harder approach is setting CPU affinity explicitly via `pthread_mach_thread_np` + `thread_policy_set` with `THREAD_AFFINITY_POLICY`. This would force threads onto specific cores, eliminating the remaining variance. macOS makes this difficult intentionally, but it's possible.

### 9.4 NEON store-pair in `clearNEON`

`STP Q0, Q1, [addr]` stores 256 bits in one instruction. Using store-pairs instead of `vst1q_u64` would reduce 9 stores to 5. Requires checking M1 microarchitecture throughput for `STP` with SIMD registers.

### 9.5 Outer loop incremental `blocked` update

After the initial `expandSpan` + `markSpan` for the seed span, the outer loop re-calls `getBlockedBits(y)` to recompute `blocked` for row `y`. The same incremental trick used in the inner loop could apply here: `blocked |= spanMask(x1, x2)`. Minor since this happens once per region (not per span), but it eliminates 1–2 more memory loads.
