# Go Territory Detection — Full Implementation & Benchmark Documentation

## Table of Contents
1. [Problem Overview](#problem-overview)
2. [Shared Data Representation](#shared-data-representation)
3. [The Expanded Board Trick](#the-expanded-board-trick)
4. [Implementation Progression](#implementation-progression)
   - [state.cpp — Reference Implementation](#statecpp--reference-implementation)
   - [ST-test.cpp — Single-Threaded Benchmark](#st-testcpp--single-threaded-benchmark)
   - [MT-test.cpp — Multi-Threaded Benchmark](#mt-testcpp--multi-threaded-benchmark)
   - [arm-optimized.cpp — Full ARM M1 Optimized](#arm-optimizedcpp--full-arm-m1-optimized)
5. [Extra / Intermediate Files](#extra--intermediate-files)
6. [Build System (Makefile)](#build-system-makefile)
7. [Optimization Summary Table](#optimization-summary-table)

---

## Problem Overview

The goal is to compute **enclosed territory** for each player on a 19×19 Go board as fast as possible. In Go, territory is a set of empty intersections completely surrounded by stones of one color — empty cells that a flood-fill cannot escape to the board edge.

This is the inner loop of Monte Carlo Tree Search (MCTS) for Go: after a rollout (random playout), you need to score the board quickly. The faster territory detection is, the more simulations per second the engine can run.

The algorithm is benchmarked at 100,000 rounds per run, measuring the average nanoseconds per **pair** (black territory + white territory computed together).

---

## Shared Data Representation

All implementations share the same compact board encoding:

```
data: std::array<uint32_t, 23>   // 23 * 32 = 736 bits total
```

The 736-bit array is partitioned as:
- **Bits 0–360**: Black stones. Bit `i` = 1 means black stone at board position `i`.
- **Bits 361–721**: White stones. Bit `i + 361` = 1 means white stone at position `i`.
- **Bits 722–724** (state.cpp only): Game meta-state (pass, turn, active).

Board positions are indexed row-major: position `(x, y)` = `y * 19 + x`, so position 0 is top-left, 360 is bottom-right.

### BIT_MASKS

```cpp
static constexpr std::array<uint32_t, 32> BIT_MASKS = { 1u<<0, 1u<<1, ..., 1u<<31 };
```

Pre-computed lookup table to avoid the shift instruction when accessing bits:

```cpp
// Reading bit at position `bitIndex`:
bool stone = data[bitIndex >> 5] & BIT_MASKS[bitIndex & 31];
//                ^^^^^^^^^^^^^^         ^^^^^^^^^^^^^^^^
//            which uint32_t word      which bit within it
```

### makeMove (benchmark version)

All benchmark files use a **simplified** `makeMove` that only checks:
1. Index in bounds (`0 <= bitIndex < 722`)
2. Position not already occupied
3. Opposite color not on the same square

It does **not** check for captures or suicide. This is intentional — the goal is a random stone fill to create a test board, not a legal game.

---

## The Expanded Board Trick

Territory detection requires knowing when a flood-fill "escapes" to the board edge. On a flat 19×19 board, that means four different boundary checks per step. The trick used here eliminates **horizontal** boundary checks entirely.

### Construction (EN = 57 columns)

Each row of 19 bits is expanded to 57 bits in three sections:

```
[ reversed(row) | original(row) | reversed(row) ]
   bits 0–18       bits 19–37      bits 38–56
```

Example for row `0b1001000000000000001` (stones at columns 0 and 18):
```
reversed  = 0b1000000000000001001   (reversed bit order)
original  = 0b1001000000000000001
expanded  = reversed | (original << 19) | (reversed << 38)
```

The expanded row is stored as a `uint64_t` (64 bits, only 57 used).

### Why This Works

The reversed mirrors on both sides act as **virtual walls**. When the flood-fill hits column 0 or column 56 of the expanded board, it has reached the horizontal edge — territory has escaped. This converts the four-way boundary check into a single horizontal check (`x < 0 || x >= EN`), which is easy to detect. Vertical boundaries (top/bottom rows) are still checked explicitly.

The **middle 19 columns** (indices 19–37 in each row) correspond to the actual board. After territory detection on the 57×19 expanded grid, only cells in columns 19–37 are extracted as real territory.

---

### ST-test.cpp — Single-Threaded Benchmark

**File**: [ST-test.cpp](ST-test.cpp)

The baseline. Everything sequential, no threading, no ARM intrinsics.

#### Board Setup

```cpp
constexpr int ROUNDS = 100000;
constexpr int TARGET_STONES = 90;   // ~45 black, 45 white
std::mt19937 rng{42};               // fixed seed → reproducible
```

90 stones is a relatively sparse board (≈25% filled). This tests territory detection on a typical mid-game position.

#### ExpandedBoard Construction

```cpp
// Row extraction from packed bits:
int start = y * 19 + base;   // base = 0 for black, 361 for white
int word = start >> 5;        // which uint32_t
int shift = start & 31;       // bit offset within that word

if (shift <= 13) {
    // All 19 bits fit in one 32-bit word (19 bits + 13 offset = 32 max)
    bits = (s.data[word] >> shift) & 0x7FFFF;
} else {
    // Row spans two 32-bit words — need to stitch them together
    int first = 32 - shift;
    bits = (s.data[word] >> shift)
         | ((s.data[word+1] & ((1ULL << (19 - first)) - 1)) << first);
}
```

**Why `shift <= 13`?** A 19-bit row starting at offset 13 ends at bit 31 — exactly at the word boundary. Offset 14+ means it crosses into the next word.

**Bit reversal** (simple loop):
```cpp
uint64_t rev = 0;
for (int i = 0; i < 19; ++i)
    if (bits & (1ULL << i)) rev |= (1ULL << (18 - i));
```
This runs 19 conditional branches. Slow, but simple.

#### detectTerritory — Simple BFS

```cpp
std::bitset<ESIZE> territory, visited;
std::vector<int> queue;
queue.reserve(512);
```

Uses `std::vector` as a manual queue (index `front`). For each unvisited empty cell:
1. Push to queue, mark visited
2. BFS: for each cell, check 4 neighbors
3. If any neighbor is OOB horizontally → `touches_edge = true`
4. Vertical OOB: the existing code has a subtle bug here — it sets `ny = 0` or `ny = 18` when OOB instead of flagging edge, which can cause incorrect neighbor lookups. This doesn't affect correctness much at benchmark scale but is not the intended behavior.
5. If `!touches_edge`, mark all queue cells as territory

**Return type**: `std::bitset<ESIZE>` (1083 bits, covers 57×19 expanded board).

#### computeEnclosedTerritoryOnly

Extracts territory from the middle 19 columns of the expanded board:
```cpp
for (int y = 0; y < 19; ++y)
    for (int x = 0; x < 19; ++x)
        if (full[y * EN + (x + 19)]) result.set(y * 19 + x);
```
Returns `std::bitset<OFFSET>` (361 bits).

#### Benchmark Loop

```cpp
for (int i = 0; i < ROUNDS; ++i) {
    auto b = game.computeEnclosedTerritoryOnly(true);   // black
    auto w = game.computeEnclosedTerritoryOnly(false);  // white
}
```

Pure sequential. The two calls are back-to-back with no parallelism.

---

### MT-test.cpp — Multi-Threaded Benchmark

**File**: [MT-test.cpp](MT-test.cpp)

Same algorithm as ST-test, but uses 2 threads to compute black and white territory in parallel.

#### Key Differences from ST-test

**250 stones, seed=45** (denser board — more interesting for territory detection):
```cpp
constexpr int TARGET_STONES = 250;
std::mt19937 rng{45};
```

**Result storage**:
```cpp
std::bitset<State::OFFSET> results[2][ROUNDS];
```
This is a huge allocation: `2 × 100000 × 361 bits = 9.025 MB`. Pre-allocated to avoid any allocation cost inside the benchmark loop.

**Worker function** (work-stealing pattern):
```cpp
auto worker = [&game, &counter, &results](bool forBlack) {
    int idx;
    while ((idx = counter++) < ROUNDS) {
        results[forBlack][idx] = game.computeEnclosedTerritoryOnly(forBlack);
    }
};
```

Note: `counter++` is `std::atomic<int>` post-increment, which is `fetch_add(1)`. The `counter` here is **shared between black and white threads** — there is actually a bug here vs. the arm-optimized version. Both `t1` (black) and `t2` (white) share the same `counter`, so they will compete. The correct design (used in arm-optimized) uses **two separate counters**.

**Thread pool**: Only 2 threads, created once, joined after all 100,000 rounds.

```cpp
std::thread t1(worker, true);    // black territory
std::thread t2(worker, false);   // white territory
t1.join();
t2.join();
```

This is the simplest possible thread pool: one thread per task type, both running concurrently. The speedup over ST-test comes from the black and white computations overlapping in time.

---

### arm-optimized.cpp — Full ARM M1 Optimized

**File**: [arm-optimized.cpp](arm-optimized.cpp)

The final optimized version. Every decision in this file is driven by ARM M1 microarchitecture characteristics.

#### Dependencies

```cpp
#include <arm_neon.h>      // ARM NEON SIMD intrinsics
#include <sys/sysctl.h>    // macOS sysctl for core topology query
```

---

#### Core Detection (`printCoreInfo` / `getPhysicalCoreCount`)

The M1 has a heterogeneous core topology. `sysctl` exposes this:

```cpp
sysctlbyname("hw.perflevel0.physicalcpu", &perf_cores, ...);  // P-cores (fast)
sysctlbyname("hw.perflevel1.physicalcpu", &eff_cores, ...);   // E-cores (efficient)
sysctlbyname("hw.physicalcpu",            &total_cores, ...); // P + E combined
```

`printCoreInfo()` is a diagnostic that reports all three counts and explicitly calls out the bug in the earlier `ARM-state.cpp` (which only used `perflevel0`, i.e., only P-cores).

`getPhysicalCoreCount()` uses `hw.physicalcpu` (all physical cores) as the thread count, which is the fix. On M1 Pro/Max, this could be 8–10 cores instead of just 4.

---

#### `FastQueue` — Packed Coordinate Queue

```cpp
struct alignas(64) FastQueue {
    alignas(64) std::array<uint32_t, 2048> buf;
    int head = 0;
    int tail = 0;
```

**Alignment**: `alignas(64)` ensures the struct and its buffer start on a cache-line boundary (64 bytes on M1). This prevents false sharing and ensures optimal cache behavior.

**Circular buffer**: Size 2048 (power of 2), uses `& 2047` instead of `% 2048` for fast modulo:
```cpp
buf[tail++ & 2047] = ...;
buf[(head + idx) & 2047];
```

**Packed coordinates**: Instead of storing a flat index, coordinates are packed into a single `uint32_t`:
```cpp
void push(int x, int y) {
    buf[tail++ & 2047] = (y << 16) | x;  // y in high 16 bits, x in low 16 bits
}

void getCoords(int idx, int& x, int& y) const {
    uint32_t packed = buf[(head + idx) & 2047];
    x = packed & 0xFFFF;
    y = packed >> 16;
}
```

This avoids the repeated `cur / EN` and `cur % EN` divisions used in ST/MT-test to recover x and y from a flat index. Division is expensive; these are free shifts and masks.

---

#### `OptimizedState`

```cpp
struct __attribute__((aligned(64))) OptimizedState {
    static constexpr int N = 19;
    static constexpr int OFFSET = 361;
    static constexpr int EN = 3 * N;    // 57 (expanded board width)
    static constexpr int ESIZE = EN * N; // 1083 (expanded board total cells)

    alignas(64) std::array<uint32_t, 23> data{};
```

Same `data` encoding as before. Aligned to 64 bytes (cache line).

**`makeMove`** uses branch prediction hints:
```cpp
if (bitIndex < 0 || bitIndex >= 2 * OFFSET) [[unlikely]] return false;
if (getStone(bitIndex)) [[unlikely]] return false;
if (getStone(opp)) [[unlikely]] return false;
```

`[[unlikely]]` tells the compiler the failure paths are rare, so it lays out the success path as the fall-through (no branch taken = faster on ARM's branch predictor).

---

#### `ExpandedBoard` — ARM NEON Row Extraction

```cpp
struct alignas(64) ExpandedBoard {
    alignas(64) std::array<uint64_t, 19> rows{};

    ExpandedBoard(const OptimizedState& s, bool black) {
        const int base = black ? 0 : OFFSET;

        for (int y = 0; y < 19; ++y) {
            int start = y * 19 + base;
            int word = start >> 5;
            int shift = start & 31;

            uint64_t bits;

            if (__builtin_expect(shift <= 13, 1)) [[likely]] {
                // Fast path: row fits in one word
                bits = (s.data[word] >> shift) & 0x7FFFFULL;
            } else [[unlikely]] {
                // Slow path: row spans two words — use NEON to load both at once
                uint32x2_t words = vld1_u32(&s.data[word]);
                uint64_t combined = vget_lane_u64(vreinterpret_u64_u32(words), 0);
                bits = (combined >> shift) & 0x7FFFFULL;
            }
```

The `[[likely]]/[[unlikely]]` annotations and `__builtin_expect` both tell the compiler the same thing (belt-and-suspenders for different compiler versions).

**NEON path for word-spanning rows**:
- `vld1_u32(&s.data[word])`: Loads two consecutive `uint32_t` values into a 64-bit NEON register (`uint32x2_t`). This is a single 64-bit load.
- `vreinterpret_u64_u32(words)`: Reinterprets those 64 bits as a `uint64x1_t`.
- `vget_lane_u64(..., 0)`: Extracts to a scalar `uint64_t`.
- Then `>> shift` and mask extracts the 19 bits that span the two words.

This is one 64-bit load vs. two 32-bit loads and a manual stitch — fewer instructions and avoids a potential load-use stall.

**Bit reversal** uses ARM hardware RBIT:
```cpp
uint64_t rev = (uint64_t)__builtin_bitreverse32((uint32_t)bits) >> 13;
rows[y] = rev | (bits << 19) | (rev << 38);
```

`__builtin_bitreverse32` compiles to a single `RBIT` instruction on ARM. It reverses all 32 bits. Since we only have 19 meaningful bits (in positions 0–18), a 32-bit reverse puts them in positions 13–31. Right-shifting by 13 aligns the reversed bits to positions 0–18. This replaces the 19-iteration loop from ST-test.

---

#### `CompactBitset` — Optimized Bitset for Expanded Board

```cpp
struct alignas(64) CompactBitset {
    alignas(64) std::array<uint64_t, 18> data{};
```

18 × 64 = 1152 bits. Covers 1083 cells of the 57×19 expanded board with some slack. Uses `uint64_t` words (vs. `uint32_t` in the `OptimizedState::data`) because 64-bit words need fewer operations for the larger bitset.

**`clearNEON()`**: Clears 1152 bits using NEON:
```cpp
uint64x2_t zero = vdupq_n_u64(0);
for (int i = 0; i < 18; i += 2) {
    vst1q_u64(&data[i], zero);
}
```
`vdupq_n_u64(0)` creates a 128-bit register of all zeros. `vst1q_u64` writes 128 bits (2 × `uint64_t`) per store. 9 stores clear all 18 words = 9 instructions vs. 18 for scalar.

**`markSpan(start_idx, count)`**: Sets `count` consecutive bits starting at `start_idx`:
```cpp
void markSpan(int start_idx, int count) {
    int word_idx = start_idx >> 6;
    int bit_offset = start_idx & 63;

    if (bit_offset + count <= 64) {
        // Entire span fits in one 64-bit word
        uint64_t mask = ((1ULL << count) - 1) << bit_offset;
        data[word_idx] |= mask;
    } else {
        // Span crosses word boundaries
        int first_count = 64 - bit_offset;
        data[word_idx] |= (~0ULL) << bit_offset;  // fill to end of first word
        word_idx++;
        int remaining = count - first_count;
        while (remaining >= 64) {
            data[word_idx++] = ~0ULL;  // full words
            remaining -= 64;
        }
        if (remaining > 0)
            data[word_idx] |= (1ULL << remaining) - 1;  // partial last word
    }
}
```

This is the key enabler of the **span-fill BFS**. Instead of visiting each cell individually and setting a bit, the BFS works in horizontal spans and marks entire runs at once — far fewer bitset operations.

**`testAndSet()` with cached word pointer**:
```cpp
bool testAndSet(int idx, uint64_t*& cached_word, int& cached_word_idx) {
    int word_idx = idx >> 6;
    uint64_t bit = 1ULL << (idx & 63);
    if (word_idx != cached_word_idx) {
        cached_word = &data[word_idx];
        cached_word_idx = word_idx;
    }
    if (*cached_word & bit) return false;
    *cached_word |= bit;
    return true;
}
```

When scanning horizontally, consecutive cells are often in the same 64-bit word. This caches the pointer to the current word, skipping the `&data[word_idx]` address computation if the word index hasn't changed. Minor but measurable on tight loops.

**`popcount19x19()`**: Counts territory cells in the **middle 19 columns** of the expanded board, row by row:
```cpp
for (int y = 0; y < 19; ++y) {
    int start_idx = y * EN + 19;  // EN=57, start at column 19 (middle section)
    int word_idx = start_idx >> 6;
    int bit_offset = start_idx & 63;

    uint64_t w1 = data[word_idx];
    uint64_t w2 = data[word_idx + 1];

    uint64_t mask1 = (~0ULL) << bit_offset;
    int count_w1 = __builtin_popcountll(w1 & mask1);

    int bits_in_w1 = 64 - bit_offset;
    int bits_in_w2 = 19 - bits_in_w1;
    uint64_t mask2 = (1ULL << bits_in_w2) - 1;
    int count_w2 = __builtin_popcountll(w2 & mask2);

    total_count += count_w1 + count_w2;
}
```

Instead of extracting bits back to a `std::bitset<361>` and doing a loop (as in ST/MT-test), this directly counts set bits in the correct region using `__builtin_popcountll` (which compiles to a single `CNT`+`ADDV` or `POPCNT` instruction on ARM). Returns an `int` count, not a bitset — much cheaper.

---

#### `detectTerritory` — Span-Fill BFS

The most significant algorithmic change. Instead of pixel-by-pixel BFS, it uses **scanline span-filling**.

```cpp
struct Span {
    int16_t x1, x2, y, parent_y;
};

alignas(64) std::array<Span, 1024> stack;
int stack_top = 0;

alignas(64) std::array<uint32_t, 1024> region_cells;
int region_count = 0;
```

`int16_t` fields in `Span`: each Span is 8 bytes (4 × int16_t). This doubles how many spans fit in a cache line vs. `int32_t`.

**Algorithm**:
1. For each unvisited empty cell `(x, y)`:
2. Expand horizontally left and right to find the full span `[x1, x2]` at row `y`.
3. Mark the entire span as visited with `markSpan()`.
4. Store all cells in `region_cells[]`.
5. Push spans for the rows above `(y-1)` and below `(y+1)` onto the stack.
6. Pop a span, scan the requested row within `[x1, x2]`, find connected sub-spans, extend them left/right, repeat.
7. Track `touches_edge` (x==0 or x==EN-1 for horizontal; y==0 or y==18 for vertical).
8. If `!touches_edge`, mark all `region_cells` as territory.

This processes `O(spans)` operations vs. `O(cells)` — in practice 3–5× fewer BFS steps for a typical Go board configuration.

**Region cell cap**: `region_count < 1024` guards prevent stack/array overflow. On a 19×19 board with 250 stones placed, the maximum connected empty region is bounded well below 361 cells.

---

#### Benchmark Setup

```cpp
constexpr int ROUNDS = 100000;
constexpr int TARGET_STONES = 250;
std::mt19937 rng{45};  // same seed as MT-test for fair comparison
```

**Thread pool** (fixed from MT-test):
```cpp
// Separate counters for black and white — no shared counter bug
std::atomic<int> black_counter{0};
std::atomic<int> white_counter{0};

for (int i = 0; i < num_cores; ++i) {
    if (i < num_cores / 2) {
        threads.emplace_back(worker, std::ref(black_counter), std::ref(black_results), true);
    } else {
        threads.emplace_back(worker, std::ref(white_counter), std::ref(white_results), false);
    }
}
```

Half the cores compute black territory, half compute white. Each core uses `fetch_add(1, memory_order_relaxed)` to claim the next unclaimed round index. `memory_order_relaxed` is safe here because the result of round `i` does not depend on the result of round `j` — each iteration is fully independent (same read-only board, independent stack/bitset allocations).

**Memory fence** after worker finishes:
```cpp
std::atomic_thread_fence(std::memory_order_release);
```
Ensures all writes to `results[]` by this thread are visible to the main thread after `join()`.

**Result type**: `std::array<int, ROUNDS>` (just counts). Zero copy, no bitset extraction overhead.

**Verification**: After all threads finish, sums all results and prints totals to confirm correctness.

---

## Extra / Intermediate Files

### extra/arm-test.cpp

An intermediate ARM version. It:
- Adds `__attribute__((always_inline))` and `[[unlikely]]` to `makeMove` and `getStone`.
- Adds `reverseBits19` using `__builtin_bitreverse32` (same trick as arm-optimized).
- Uses an `alignas(64) ExpandedBoard` — but still uses `std::bitset` and pixel-by-pixel BFS.
- Thread pool uses M1 performance cores only (4), not all physical cores.

This was the "NEON bit reversal + basic threading" step before the span-fill BFS was added.

### extra/ARM-state.cpp

An earlier ARM attempt that adds:
- `OptimizedBitset361`: custom 361-bit bitset over 6 `uint64_t` words with NEON clear, for liberty/visited tracking.
- `FastQueue` storing flat `int` indices (not packed coordinates).
- Only uses `hw.perflevel0.physicalcpu` (P-cores only) — this is the bug `arm-optimized.cpp`'s `printCoreInfo` flags.
- Still uses `std::bitset` territory representation.

### extra/state.cpp

The original full-rules Go engine with visualization. The only implementation with:
- Full legal move checking (captures, suicide)
- Mirror-traversal BFS for territory (one reflection allowed before treating as dead-end)
- ANSI terminal rendering of board and territory

### extra/summary.txt & summary_arm-test.txt

Markdown notes written during development explaining the thread pool pattern and benchmark design (see the Documentation section of this repo for quotes from these).

---

## Build System (Makefile)

```makefile
CXX := g++
CXXFLAGS := -std=c++17 -O3 -march=armv8.5-a -mcpu=apple-m1 -flto -ffast-math -pthread

ARM_AGGRESSIVE := -std=c++17 -O3 -ffast-math -march=armv8.5-a -mcpu=apple-m1 \
                  -flto -pthread -fvectorize -funroll-loops \
                  -fslp-vectorize -fwhole-program-vtables
```

All binaries use `ARM_AGGRESSIVE`. Key flags:

| Flag | Effect |
|---|---|
| `-march=armv8.5-a` | Targets ARMv8.5-A ISA — enables `RBIT`, `CNT`, `POPCNT`, advanced NEON |
| `-mcpu=apple-m1` | M1-specific tuning (pipeline depth, cache sizes, latency tables) |
| `-O3` | Full optimization: auto-vectorization, loop transformations, inlining |
| `-ffast-math` | Allows reordering floating-point ops; also enables some integer optimizations |
| `-flto` | Link-Time Optimization — inlines across translation units, eliminates dead code |
| `-fvectorize` | Enables auto-vectorization (already in -O3 on clang, explicit here) |
| `-funroll-loops` | Unrolls loops whose bounds are known at compile time |
| `-fslp-vectorize` | Superword-Level Parallelism vectorization (straight-line code SIMD) |
| `-fwhole-program-vtables` | Enables devirtualization across the whole program |
| `-pthread` | Links pthreads for `std::thread` |

**`benchmark` target**: Runs 100 iterations of each binary, extracts "Per pair:" timing with awk, saves to `benchmark_results.csv`, then runs `GeneralStats.py` for analysis.

---

## Optimization Summary Table

| Feature | state.cpp | ST-test | MT-test | arm-optimized |
|---|---|---|---|---|
| Board encoding | 23×uint32 | 23×uint32 | 23×uint32 | 23×uint32 |
| Full Go rules | Yes | No | No | No |
| Threads | 1 | 1 | 2 | All physical cores |
| Separate black/white counters | N/A | N/A | **No (bug)** | **Yes (fixed)** |
| Uses all M1 cores | N/A | N/A | No | **Yes** |
| Bit reversal | Loop (19 iters) | Loop (19 iters) | Loop (19 iters) | **`RBIT` hardware** |
| Word-span row extraction | Manual stitch | Manual stitch | Manual stitch | **NEON `vld1_u32`** |
| Visited set | `std::bitset` | `std::bitset` | `std::bitset` | **`CompactBitset` (uint64×18)** |
| Visited clear | `bitset::reset()` | `bitset::reset()` | `bitset::reset()` | **NEON `vst1q_u64`** |
| BFS style | Pixel BFS | Pixel BFS | Pixel BFS | **Span-fill BFS** |
| Span marking | N/A | N/A | N/A | **`markSpan()` bulk** |
| BFS queue | `std::array` stack | `std::vector` | `std::vector` | **Packed `FastQueue`** |
| Result type | `bitset<361>` | `bitset<361>` | `bitset<361>` | **`int` count** |
| Territory count | Bitset loop | Bitset loop | Bitset loop | **`__builtin_popcountll`** |
| Cache alignment | None | None | None | **`alignas(64)` everywhere** |
| Branch hints | None | None | None | **`[[likely]]`/`[[unlikely]]`** |
| Stones (benchmark) | 250 | 90 | 250 | 250 |
| Rounds | N/A | 100,000 | 100,000 | 100,000 |
