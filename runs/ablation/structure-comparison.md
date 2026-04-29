# Structure Comparison

Date: 2026-04-29
Model: `Qwen/Qwen3-0.6B` via MLX LoRA (`iters=200`, `batch=1`, `num_layers=8`)

## Validation Loss

| Experiment | Final val loss | Structure fidelity | jondaet% | banmal-like% | Completion-only term hits (TC/밀빵/케어/ㅋㅋ/ㅠㅠ) |
|---|---:|---:|---:|---:|---|
| A1-flat-v2 | 4.221 | 0% | 15.0 | 50.0 | 0 / 0 / 0 / 3 / 55 |
| A2-structured-v3 | 3.455 | 90% | 10.0 | 45.0 | 0 / 0 / 0 / 19 / 1 |
| A3-context-stream | 4.332 | 0% | 35.0 | 70.0 | 0 / 0 / 0 / 1 / 45 |
| A4-flat-v2-title | 4.604 | 0% | 5.0 | 55.0 | 0 / 0 / 0 / 226 / 83 |

## Sample Quality

- `A2` is the clear winner. It is the only format variant that reliably preserves wrappers, closing `18/20` samples correctly, and it beats every non-structured run on validation loss.
- `A1` learns some casual tone but not the format. Post prompts often collapse into repeated `제목:` scaffolds or short broken continuations, and comments overuse `ㅠㅠ`.
- `A3` confirms that thread-style context without explicit wrappers does not solve the serialization problem. It drifts into reactive chatter and overproduces emotional markers.
- `A4` is the weakest condition. Prepending an explicit `제목:` marker to flat data amplifies title echo and produces the worst marker spam in the set.

## Takeaways

- For local 0.6B ranking, explicit structure tokens matter more than raw context.
- Title-prepending is not a substitute for structured serialization.
- The structured v3 format should be the default corpus format for further local ablations and the first 8B CPT attempt.
