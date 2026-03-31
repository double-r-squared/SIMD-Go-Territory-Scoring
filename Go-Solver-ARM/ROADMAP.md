# Project Polish Roadmap
## Goal: Professional Resume Piece

---

## Phase 1 — Clean Up the Repository

- [x] Add a proper **README.md** — project summary, build instructions, benchmark instructions, results table, and a one-paragraph "why this matters" blurb
- [ ] Consolidate `extra/` — move relevant files in or delete noise (old pngs, drafts)
- [x] Add a `.gitignore` — binaries, `*.o`, `*.csv`, `__pycache__`, `venv/`
- [ ] Rename files to be self-explanatory (`ST-test.cpp` → `baseline_singlethread.cpp`, etc.) or add a clear naming explanation in README *(covered in README project structure table)*

---

## Phase 2 — Lock in the Numbers

- [x] Run a **fresh full benchmark** (`make benchmark`) with the current v4 code
- [x] Run `python3 GeneralStats.py` to regenerate `final_benchmark_figure.png` with current numbers
- [x] Add **`DetailedStats.py`** — prints terminal stats table + ready-to-paste markdown

**Locked numbers (100-run benchmark, Apple M1, fixed seed):**

| Version | Threads | Mean (ns) | Median (ns) | Min (ns) | Std Dev | CV% | Speedup vs ST |
| --- | --- | --- | --- | --- | --- | --- | --- |---|
| Single-threaded | 1 | 15,152 | 15,144 | 14,945 | 106 | 0.7% | 1.0× |
| Multi-threaded | 2 | 2,582 | 2,569 | 2,543 | 56 | 2.2% | **5.9×** |
| ARM Optimized | 4 P-cores | 956 | 954 | 900 | 31 | 3.3% | **15.8×** |

---

## Phase 3 — Code Quality Pass

- [x] Add a **file-level comment block** at the top of `arm-optimized.cpp` — author, algorithm summary, benchmark numbers
- [x] Inline comments audit — all ARM intrinsics and bit tricks are explained
- [x] `arm-optimized.cpp` compiles **warning-free** (`-Wall -Wextra`) ✓

---

## Phase 4 — The Resume Artifact (README)

- [x] README.md written — tells the full story: what, why hard, how, numbers, techniques, build instructions

---

## Phase 5 — Optional Stretch (high signal for resume)

- [ ] **Hard P-core affinity** (`thread_policy_set` + `THREAD_AFFINITY_POLICY`) — would eliminate E-core variance and potentially push mean below 900 ns
- [ ] **Compile-time board table** — `constexpr`-generated expanded board for live-game integration
- [ ] **Live game integration stub** — a thin `score_territory(State&)` public API showing production-readiness

---

## Target Headline

> "Implemented ARM M1-optimized Go territory scoring using NEON SIMD intrinsics,
> packed bitboards, and cache-aligned span-fill BFS. Achieved **sub-microsecond**
> per-evaluation latency (956 ns mean, 900 ns min), a **15.8× speedup** over the
> single-threaded baseline — fast enough for real-time game-tree search."

---

## Progress Tracker

| Phase | Status | Notes |
|---|---|---|
| Phase 1 — Repo cleanup | Mostly done | `extra/` cleanup optional |
| Phase 2 — Lock in numbers | **Complete** | 956 ns mean, 15.8× speedup |
| Phase 3 — Code quality | **Complete** | Zero warnings, full comments |
| Phase 4 — README | **Complete** | See README.md |
| Phase 5 — Stretch goals | Optional | See items above |
