# Refinement Report

- Before diagnostic: `runs/refinement-20260427-163226/cycle-1/diagnostic.json`
- After diagnostic: `runs/refinement-20260427-163757/cycle-1/diagnostic.json`
- Regenerated CPT: `46,856 -> 63,288` rows (`+16,432`, `+35.1%`)
- Recovered hybrid comments: `9,947`

## Summary

| Metric | Before | After |
| --- | --- | --- |
| Total gaps | 9 moderate | 7 total (`1 critical`, `6 moderate`) |
| Resolved terms | - | `도파민`, `갯수`, `선릉` |
| New regression | - | `ㅅㅂ` (`1.9631 -> 2.0235`, `ok -> moderate`) |
| Worst remaining gap | `TC` `0.1048` (`moderate`) | `TC` `0.0655` (`critical`) |

## Term-Level Before/After

| Term | Before ratio | After ratio | Train count delta | Result |
| --- | --- | --- | --- | --- |
| `도파민` | `0.4992` (`moderate`) | `0.5856` (`ok`) | `3478 -> 3989` | resolved |
| `갯수` | `0.4588` (`moderate`) | `0.5044` (`ok`) | `2000 -> 2150` | resolved |
| `선릉` | `0.3980` (`moderate`) | `0.5380` (`ok`) | `143 -> 189` | resolved |
| `쩜오` | `0.3979` (`moderate`) | `0.4059` (`moderate`) | `2351 -> 2345` | slight improvement, still gap |
| `하이퍼` | `0.3486` (`moderate`) | `0.3893` (`moderate`) | `87 -> 95` | improved, still gap |
| `밀빵` | `0.2289` (`moderate`) | `0.2345` (`moderate`) | `1087 -> 1089` | nearly flat, still gap |
| `케어` | `0.2564` (`moderate`) | `0.2604` (`moderate`) | `2608 -> 2589` | nearly flat, still gap |
| `ㅡㅡ` | `0.4915` (`moderate`) | `0.4543` (`moderate`) | `3885 -> 3511` | worsened |
| `TC` | `0.1048` (`moderate`) | `0.0655` (`critical`) | `463 -> 283` | worsened materially |
| `ㅅㅂ` | `1.9631` (`ok`) | `2.0235` (`moderate`) | `2811 -> 2833` | new over-representation |

## Interpretation

- Hybrid split recovery itself worked at volume: `9,947` comments were recovered into CPT.
- Net outcome improved only partially because the regenerated CPT is a full rebuild, not an additive patch. Three target gaps moved to `ok`, but the global CPT token mix changed enough to create one new regression and to worsen `TC`.
- `TC` appears especially sensitive to the split strategy. Many recovered comments keep the user preamble and drop the promo suffix, and the dropped suffix is where `TC` often lived. That reduced `TC` train hits from `463` to `283` even though total CPT rows increased.
- `ㅡㅡ` also lost representation after rebuild, suggesting some recovered user parts are shorter or cleaner than the original CPT examples that previously carried this token.

## Recommended Follow-Up

- Inspect recovered hybrid comments that originally contained `TC` and `ㅡㅡ` to confirm whether the separator split is stripping the only occurrence.
- If the goal is term-gap repair rather than full CPT cleanup, compare an additive merge strategy against full CPT replacement before adopting this rebuild.
