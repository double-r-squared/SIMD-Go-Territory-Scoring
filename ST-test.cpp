#include <bitset>
#include <array>
#include <vector>
#include <cstdint>
#include <algorithm>
#include <random>
#include <chrono>
#include <iostream>
#include <iomanip>

struct State {
    static constexpr int N = 19;
    static constexpr int OFFSET = 361;
    static constexpr int EN = 3 * N;
    static constexpr int ESIZE = EN * N;

    std::array<uint32_t, 23> data{};  // 736 bits exactly

    static constexpr std::array<uint32_t, 32> BIT_MASKS{
        1u<<0, 1u<<1, 1u<<2, 1u<<3, 1u<<4, 1u<<5, 1u<<6, 1u<<7,
        1u<<8, 1u<<9, 1u<<10, 1u<<11, 1u<<12, 1u<<13, 1u<<14, 1u<<15,
        1u<<16, 1u<<17, 1u<<18, 1u<<19, 1u<<20, 1u<<21, 1u<<22, 1u<<23,
        1u<<24, 1u<<25, 1u<<26, 1u<<27, 1u<<28, 1u<<29, 1u<<30, 1u<<31
    };

    static std::array<std::vector<int>, OFFSET> initNeighbors() {
        std::array<std::vector<int>, OFFSET> neighbors;
        for (int i = 0; i < OFFSET; ++i) {
            int x = i % 19, y = i / 19;
            if (x > 0)  neighbors[i].push_back(i - 1);
            if (x < 18) neighbors[i].push_back(i + 1);
            if (y > 0)  neighbors[i].push_back(i - 19);
            if (y < 18) neighbors[i].push_back(i + 19);
        }
        return neighbors;
    }

    static const std::array<std::vector<int>, OFFSET> all_neighbors;

    State() = default;

    inline bool getStone(int bitIndex) const {
        return data[bitIndex >> 5] & BIT_MASKS[bitIndex & 31];
    }

    inline bool getBlack(int idx) const { return getStone(idx); }
    inline bool getWhite(int idx) const { return getStone(idx + OFFSET); }

    bool makeMove(int bitIndex) {
        if (bitIndex < 0 || bitIndex >= 2*OFFSET) return false;
        if (getStone(bitIndex)) return false;
        int opp = (bitIndex < OFFSET) ? bitIndex + OFFSET : bitIndex - OFFSET;
        if (getStone(opp)) return false;

        data[bitIndex >> 5] |= BIT_MASKS[bitIndex & 31];
        return true;  // Simplified: no capture/suicide check (still safe for random fill)
    }

    struct ExpandedBoard {
        std::array<uint64_t, 19> rows{};

        ExpandedBoard(const State& s, bool black) {
            const int base = black ? 0 : OFFSET;
            for (int y = 0; y < 19; ++y) {
                int start = y * 19 + base;
                uint64_t bits = 0;
                int word = start >> 5;
                int shift = start & 31;
                if (shift <= 13) {
                    bits = (s.data[word] >> shift) & 0x7FFFFULL;
                } else {
                    int first = 32 - shift;
                    bits = (s.data[word] >> shift) | ((s.data[word + 1] & ((1ULL << (19 - first)) - 1)) << first);
                }
                uint64_t rev = 0;
                for (int i = 0; i < 19; ++i) if (bits & (1ULL << i)) rev |= (1ULL << (18 - i));
                rows[y] = rev | (bits << 19) | (rev << 38);
            }
        }

        std::bitset<ESIZE> detectTerritory() const {
            std::bitset<ESIZE> territory, visited;
            std::vector<int> queue;
            queue.reserve(512);

            const int dx[4] = {0, 0, -1, 1};
            const int dy[4] = {-1, 1, 0, 0};

            for (int y = 0; y < 19; ++y) {
                for (int x = 0; x < EN; ++x) {
                    int idx = y * EN + x;
                    if ((rows[y] & (1ULL << x)) || visited[idx]) continue;

                    queue.clear();
                    queue.push_back(idx);
                    visited[idx] = 1;
                    bool touches_edge = false;
                    int front = 0;

                    while (front < queue.size()) {
                        int cur = queue[front++];
                        int cy = cur / EN, cx = cur % EN;

                        for (int d = 0; d < 4; ++d) {
                            int nx = cx + dx[d], ny = cy + dy[d];
                            if (nx < 0 || nx >= EN) { touches_edge = true; continue; }
                            if (ny < 0) { if (cy == 0) { touches_edge = true; continue; } ny = 0; }
                            if (ny >= 19) { if (cy == 18) { touches_edge = true; continue; } ny = 18; }

                            int ni = ny * EN + nx;
                            if ((rows[ny] & (1ULL << nx)) == 0 && !visited[ni]) {
                                visited[ni] = 1;
                                queue.push_back(ni);
                            }
                        }
                    }
                    if (!touches_edge) {
                        for (int i : queue) territory[i] = 1;
                    }
                }
            }
            return territory;
        }
    };

    std::bitset<OFFSET> computeEnclosedTerritoryOnly(bool forBlack) const {
        ExpandedBoard exp(*this, forBlack);
        auto full = exp.detectTerritory();
        std::bitset<OFFSET> result;
        for (int y = 0; y < 19; ++y)
            for (int x = 0; x < 19; ++x)
                if (full[y * EN + (x + 19)]) result.set(y * 19 + x);
        return result;
    }
};

const std::array<std::vector<int>, State::OFFSET> State::all_neighbors = State::initNeighbors();

int main() {
    constexpr int ROUNDS = 100000;
    constexpr int TARGET_STONES = 90;

    State game;
    std::vector<int> order(361);
    for (int i = 0; i < 361; ++i) order[i] = i;
    std::mt19937 rng{42}; // Fixed seed for reproducibility
    std::shuffle(order.begin(), order.end(), rng);

    int placed = 0;
    for (int idx : order) {
        if (placed >= TARGET_STONES) break;
        bool black = (placed % 2 == 0);
        if (game.makeMove(black ? idx : idx + State::OFFSET)) placed++;
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ROUNDS; ++i) {
        [[maybe_unused]] auto b = game.computeEnclosedTerritoryOnly(true);
        [[maybe_unused]] auto w = game.computeEnclosedTerritoryOnly(false);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    std::cout << "Single-threaded | Stones: " << placed
              << " | Rounds: " << ROUNDS
              << " | Total: " << ns << " ns"
              << " | Per pair: " << ns / (double)ROUNDS << " ns\n";
}