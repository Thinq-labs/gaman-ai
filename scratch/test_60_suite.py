
# ==============================================================================
# EXECUTION & BENCHMARK HARNESS
# ==============================================================================
print("\n" + "="*85)
print("RUNNING SUITE A: CHOICE (25 CASES)")
print("="*85)

a_correct = 0
a_latencies = []
for idx, (state, opts, exp) in enumerate(suite_a, 1):
    res = engine.choice(state=state, options=opts)
    sel = res["selection"]
    conf = res["confidence"]
    lat = res["latency_ms"]
    a_latencies.append(lat)
    is_corr = (sel == exp)
    if is_corr:
        a_correct += 1
    v = "PASS ✔" if is_corr else "FAIL ✖"
    snippet = state["text"][:38] + "..." if len(state["text"]) > 38 else state["text"]
    print(f"[{idx:02d}] {snippet:<42} Exp: {exp:<20} Got: {sel:<20} {conf:>4.1%} ({lat:>4.1f}ms) {v}")

print(f"\nSuite A Score: {a_correct}/{len(suite_a)} ({(a_correct/len(suite_a))*100:.1f}%) | Median Latency: {np.median(a_latencies):.2f}ms")

print("\n" + "="*85)
print("RUNNING SUITE B: NOUL (20 CASES)")
print("="*85)

b_correct = 0
b_latencies = []
for idx, (state, pred, exp) in enumerate(suite_b, 1):
    res = engine.noul(state=state, predicate=pred)
    out = res["passed"]
    prob = res["probability"]
    lat = res["latency_ms"]
    b_latencies.append(lat)
    is_corr = (out == exp)
    if is_corr:
        b_correct += 1
    v = "PASS ✔" if is_corr else "FAIL ✖"
    st = str(state)[:38] + "..." if len(str(state)) > 38 else str(state)
    print(f"[{idx:02d}] {st:<42} Exp: {str(exp):<5} Out: {str(out):<5} Prob: {prob:>4.2f} ({lat:>4.1f}ms) {v}")

print(f"\nSuite B Score: {b_correct}/{len(suite_b)} ({(b_correct/len(suite_b))*100:.1f}%) | Median Latency: {np.median(b_latencies):.2f}ms")

print("\n" + "="*85)
print("RUNNING SUITE C: SCORE (15 CASES ACROSS 5 RUBRICS)")
print("="*85)

c_monotonic = 0
c_latencies = []
for rubric_name, criterion, cases in suite_c:
    scores = []
    print(f"\nRubric: {rubric_name} ('{criterion}')")
    for state, tier in cases:
        res = engine.score(state=state, criterion=criterion)
        val = res["value"]
        lat = res["latency_ms"]
        c_latencies.append(lat)
        scores.append(val)
        bar = "█" * int(val * 20) + "░" * (20 - int(val * 20))
        st = list(state.values())[0]
        st_clip = st[:38] + "..." if len(st) > 38 else st
        print(f"  [{tier:<4}] Score: {val:>6.4f} [{bar}] {st_clip}")
    is_mono = (scores[0] > scores[1] > scores[2])
    if is_mono:
        c_monotonic += 1
    v = "PASS ✔" if is_mono else "FAIL ✖"
    print(f"  --> Monotonicity (High > Med > Low): {v}")

print(f"\nSuite C Score: {c_monotonic}/{len(suite_c)} ({(c_monotonic/len(suite_c))*100:.1f}% Monotonic) | Median Latency: {np.median(c_latencies):.2f}ms")

total_tests = len(suite_a) + len(suite_b) + sum(len(c) for _, _, c in suite_c)
print("\n" + "="*85)
print(f"GRAND TOTAL: {total_tests} TEST CASES EVALUATED")
print("="*85)
