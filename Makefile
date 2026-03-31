CXX := clang++

FLAGS := -std=c++17 -O3 -ffast-math -march=armv8.5-a -mcpu=apple-m1 \
         -flto -pthread -fvectorize -funroll-loops \
         -fslp-vectorize -fwhole-program-vtables

BINS := ST-test MT-test arm-optimized

all: $(BINS)

ST-test: ST-test.cpp
	$(CXX) $(FLAGS) -o $@ $<

MT-test: MT-test.cpp
	$(CXX) $(FLAGS) -o $@ $<

arm-optimized: arm-optimized.cpp
	$(CXX) $(FLAGS) -o $@ $<

benchmark: all
	@echo "run_number,test_name,per_pair_ns" > benchmark_results.csv
	@echo "Running 100 iterations of each test..."
	@for i in $$(seq 1 100); do \
		[ $$(( $$i % 10 )) -eq 0 ] && echo "Progress: $$i/100"; \
		./ST-test       2>/dev/null | grep "Per pair:" | awk -v r=$$i '{print r ",ST-test,"       $$(NF-1)}' >> benchmark_results.csv; \
		./MT-test       2>/dev/null | grep "Per pair:" | awk -v r=$$i '{print r ",MT-test,"       $$(NF-1)}' >> benchmark_results.csv; \
		./arm-optimized 2>/dev/null | grep "Per pair:" | awk -v r=$$i '{print r ",arm-optimized," $$(NF-1)}' >> benchmark_results.csv; \
	done
	@echo "Done. $$(( $$(wc -l < benchmark_results.csv) - 1 )) records in benchmark_results.csv"

clean:
	rm -f $(BINS) *.o benchmark_results.csv

.PHONY: all benchmark clean
