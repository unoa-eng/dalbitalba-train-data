# Post-Clean Integrity Check

- Date: `2026-04-28`
- Worker: `worker-3`
- Repo: `/Users/unoa/projects/dalbitalba-train-data`
- Dataset: `cpt_context_stream.jsonl`
- Validation set: `val_set.v2.jsonl`
- Audit seed: `20260428`

## Verdict

Overall verdict: `FAIL`

The cleaned stream passes the comment-tag, duplicate, overlap, manual-sample, and `context_comment` structure gates. The remaining failing criterion is promo heuristics: comment-side contamination appears cleared, but promo-like post rows still remain in the live file.

## Criterion Results

| # | Criterion | Result | Evidence |
| --- | --- | --- | --- |
| 1 | Count rows by kind | `PASS` | `42,516` total = `20,554 comment`, `11,899 context_comment`, `10,063 post`. |
| 2 | Verify `0` comment rows still have leading `[N]` tags | `PASS` | Matching `comment` rows: `0`. |
| 3 | Verify `0` rows match promo heuristics | `FAIL` | Broad heuristic hits: `46` rows, all in `post`. `comment` + reply-side `context_comment` hits: `0`. Parent-only promo context retained by design: `99` `context_comment` rows. |
| 4 | Sample `30` rows for manual inspection | `PASS` | Flagged sampled rows: `0 / 30`. Replacement chars: `0`. NUL rows: `0`. |
| 5 | Verify val overlap is `0` | `PASS` | Exact text overlap `0`, exact `source_id` overlap `0`, canonical thread overlap `0`. |
| 6 | Check duplicate rate | `PASS` | Exact-text duplicate groups `0`; exact-row duplicate groups `0`. |
| 7 | Check `context_comment` structure | `PASS` | Malformed rows `0`; exact reconstruction matches `10 / 10`. |

## 1. Row Count by Kind

| kind | rows |
| --- | ---: |
| `comment` | `20,554` |
| `context_comment` | `11,899` |
| `post` | `10,063` |
| `total` | `42,516` |

Compared with the pre-clean re-audit baseline (`43,635` total), the live file is down to `42,516` rows: `comment -1,004`, `context_comment -115`, `post +0`.

## 2. Leading `[N]` Tags in `comment` Rows

Result: `PASS`

No `comment` row in the cleaned file still matches the leading reply-tag patterns (`[N]`, `비회원 [N]`, or short author-prefix + `[N]`).

## 3. Promo Heuristics

Result: `FAIL`

- Broad lexical promo hits: `46`
- By kind: `post 46`, `comment 0`, `context_comment 0`
- Parent-only promo context retained by design: `99` `context_comment` rows

Interpretation:

- Worker-1 scope appears clean: there are no remaining promo hits on plain `comment` rows and no remaining promo hits on the reply/comment portion of `context_comment` rows.
- The failing residue is entirely in `post` rows, which worker-1 was not asked to rewrite.
- The broad lexical heuristic still over-fires on benign `post` text containing words like `카카오뱅크`, `카카오티`, `화려라인`, or cosmetic `헤어라인` / `얼굴라인`, so not all `46` hits are true ads.
- Even after accounting for that precision issue, several unmistakable ad/recruiter posts still remain in the live stream.

Representative true-positive style examples:

- `cpt_context_stream.jsonl:2729` / `source_id=1197826` — 일요일도 근무.. / 도파민 으로 오세요 365일 풀상주
- `cpt_context_stream.jsonl:3188` / `source_id=1189001` — 신림 하퍼 ♡명월♡ 언제든 문의 환영 / 24시간 풀상주케어
- `cpt_context_stream.jsonl:5943` / `source_id=1193113` — 면접만 봐도 면접비 들여요 착한실장.
- `cpt_context_stream.jsonl:14295` / `source_id=1195616` — 스타트톡 수금 칼같이 / 1,2,3부 풀상주 / 3~40대만 20대x
- `cpt_context_stream.jsonl:34579` / `source_id=1194718` — 도파민 1등 케어 G팀 근황 ... 연락주시면 24시간으로 답장 ... 모시겠습니다

Representative false-positive style examples from the same broad heuristic:

- `cpt_context_stream.jsonl:8679` / `source_id=1196186` — 카카오티 우버 ㅈㄴ 안잡히네요
- `cpt_context_stream.jsonl:11451` / `source_id=1200061` — 아님 화려라인인가요??
- `cpt_context_stream.jsonl:21245` / `source_id=1190108` — 얼굴라인은 맘에드는데
- `cpt_context_stream.jsonl:25243` / `source_id=1188856` — 헤어라인 그리는거요 문신(?)

## 4. Manual Inspection: 30 Random Rows

Result: `PASS`

- `30 / 30` sampled rows looked structurally valid on manual inspection.
- No sampled row showed mojibake, Unicode replacement chars (`�`), or NUL bytes.
- No sampled row showed leading tag leakage or reply-side promo leakage.
- Sampled `context_comment` rows preserved the expected `제목 / 원글 / 댓글|부모댓글 / 답글` scaffold.

Sample log:

- `line 40710` / `comment` / `source_id=1188087` — clean — 애초에 여기가 영업진들 결제한돈으로 / 운영되는 커뮤니티잉
- `line 32202` / `context_comment` / `source_id=1196986` — clean — 제목: 하이 부장들 제안은 보통 / 원글: ᄑᄑ ᄃᄐ에서 많이받져? / 부모댓글: 퀸에서도 많이합니당 / 답글: 뭘 퀸에서 많이해욬ᄏᄏᄏᄏ 진짜 / 짭쩜같이 광고하는데나 그렇겠지 아무나 다받고보니까
- `line 12839` / `comment` / `source_id=1197043` — clean — 얼굴이 이쁘면 가능할거같은뎅
- `line 7693` / `comment` / `source_id=1188796` — clean — 어렸을때부터 연예계 준비하는애들은 중고딩때도 뼈치는애들도 있긴하던데 / 근데 얘는 한것같진않은데 예전얼굴도 원체 작기도했고 해서 큰이득이 있을만한 얼굴형은 아니라 경락이나 시술같은거 오히려 했...
- `line 9199` / `post` / `source_id=1199164` — clean — 요즘 경기에 / 꾸준히 10개씩 찍는 가게면 여기 있는게 / 나은거죠? ㅠㅠ 몇푼 더벌려고 옮겼다가 들쑥날쑥 할까봐요
- `line 13184` / `comment` / `source_id=1185930` — clean — 2번얼굴 턱이 각진거지 중안부가 긴거 아닌데 ㅋㅋ 눈깔 무엇
- `line 25333` / `post` / `source_id=1194436` — clean — 이일하는거이해하는사람잇을까요? / 회피겟죠? 짭점다니는데 터치어느정도잇는건알거구ㅜ 돈빨리모으길바란다는데 그렇다고 돈도 다쓰는거아님 나이차이도5살이상 나요 둘다30대긴하는데 일하는거이해한대요 사...
- `line 41199` / `context_comment` / `source_id=1199987` — clean — 제목: 저 오늘 가씨한테 심쿵한거있음 / 원글: 쩜 마쓰방인데 손님이 어떤언니한테 짓궃게굴길래 / 화장실가다가 마주쳤을때 우우ᅮᅮᅮ하면서 / 팔양쪽으로 뻗고 안아주려는 위로해주려는제스처했는데 ...
- `line 12609` / `comment` / `source_id=1188956` — clean — 3 ㅆㅂ ㅈ같애여...
- `line 890` / `comment` / `source_id=1198784` — clean — 흠 / 언니라면 어디 갈거예영?
- `line 23951` / `comment` / `source_id=1198529` — clean — 3000요
- `line 5373` / `comment` / `source_id=1189733` — clean — 저는 강남이 생각보다 안예뻤음 .. 거긴 개나소나 일하는 느낌이였고 대전이 사이즈 더 좋았어요 일도 잘함
- `line 40532` / `context_comment` / `source_id=1190093` — clean — 제목: 보수의 심장 대구에서 왔다했더니 팅겨뿟스... / 원글: 니가 먼저 ᅮ 윤석열 어케생각하냐매 ᅮ / 그래서 오빠 나 보수의심장 대구에서 왓다하니까 나가래 ᄏᄏᄏᄏᄏᄏᄏᄏᄏᄏᄏᄏᄏᄏᄏᄏᄏ...
- `line 16180` / `comment` / `source_id=1194752` — clean — 도파민애들 다 대답없는거봐ㅋㅋ
- `line 31806` / `context_comment` / `source_id=1188972` — clean — 제목: ᄀᄀᄃ 진짜 손질 수위 극악이에요? / 원글: 하 돈급한데 ᅲᅲ ᄋᄐ 안하면 방 못보나요 진짜로 ᅲ / 쩜 중띠 가정하에 ᅮ 글고 여기 아나운서 못놀게생긴스탈 안먹히나요 / 댓글: ᄂᄂ...
- `line 36625` / `comment` / `source_id=1186900` — clean — 저ㅏ도 비슷한일 당해서 선수실명 같을까봐 / 물어봣어여 ㅋㅋㅋ이걸 다행이라해 머라해야대..
- `line 162` / `post` / `source_id=1187941` — clean — 구글이정도면 앤미 원초원메임? / 예쁜데여? 원초원메가능?
- `line 34568` / `post` / `source_id=1199828` — clean — 심심하네요 -태주- / 댓글로 끝말잇기 하시죠 ㄱ ㄱ /  / 시작 기후
- `line 36134` / `context_comment` / `source_id=1197934` — clean — 제목: ᄃᄐ ᄐᄌ ᄋᄒ ᄅᄌ / 원글: 여친 잇나요?? ᄌᄂ 내 취향들 ᅲ 솔까 사귀는 건 에반 거 아는데 유사연애라도 하려구요 / 댓글: 다 있음 섹파도 여럿 꿈 깨세요
- `line 20700` / `comment` / `source_id=1194427` — clean — 저는 걍 오늘 너무 힘들었다 술마니먹었다 이정도여
- `line 24872` / `post` / `source_id=1194263` — clean — 성형 많이한 얼굴보면 / 정신에 병있어 보임 류 ㅅㅇ 와이프도 수학 강산데도 성괴틱하니까 사람이 이상해보이고 그러더라고요
- `line 11680` / `comment` / `source_id=1198488` — clean — 도깨비 35살 엔티제 이손님 말하는건데 못생겼어요? 전 잘생겼던데
- `line 34035` / `comment` / `source_id=1187856` — clean — ㅇㅎ오빠 ㄱㅊ은데 여기서 젤 케어잘해서 바쁜듯..ㅜ
- `line 13338` / `comment` / `source_id=1188881` — clean — ㅜㅜ본가라서여.. 가족 근처살면 좋을듯해서
- `line 31400` / `context_comment` / `source_id=1188074` — clean — 제목: 맨정신에 퇴근하면 담초뺌‘ᅳᅳ? / 원글: 업진샛기들 도라이노 / 댓글: 만취아닌이상 술 많이 마셔도 티안나고 취했다가도 물 많이 마시면 금방 깨는 나같은 언니들 많을텤데..ᄏᄏᄏᄏ 진...
- `line 13961` / `comment` / `source_id=1191720` — clean — ㅍㄹ,ㅁㅇ,ㅁㅈ 이 언니들도 되게 착하던데 얼굴도 이쁜데 일잘하고 싹싹하고 손님들이 좋아하더라구요
- `line 19567` / `comment` / `source_id=1200408` — clean — 저용!!!서로 씻으라고 난리쳐주기 할래여?
- `line 15595` / `comment` / `source_id=1191358` — clean — 과일전문 유준입니다 뭐가 드시고 싶으신지요
- `line 3952` / `comment` / `source_id=1192305` — clean — 언니 너무 고생하셨고 지금도 너무 예뻐요 조금 릴렉스해요
- `line 11717` / `comment` / `source_id=1187652` — clean — 별 거 아닌 사생활로 ㄱㄹ까지라고 말하냐ㅋㅋ 안 나오겄지 뭐

## 5. Train / Val Overlap

Result: `PASS`

- Exact text overlap: `0`
- Exact `source_id` overlap: `0`
- Canonical thread overlap (`source_id.split(':', 1)[0]`): `0`

## 6. Duplicate Rate

Result: `PASS`

- Exact-text duplicate groups: `0`
- Exact-text duplicate rows: `0` / `42,516` (`0.0000%`)
- Exact-row duplicate groups: `0`
- Exact-row duplicate rows: `0` / `42,516` (`0.0000%`)

The exact-text dedup requested in the cleanup task appears to have landed: both exact-text and full-row duplicate group counts are zero.

## 7. `context_comment` Structure

Result: `PASS`

Method:

- sampled `10` random `context_comment` rows
- loaded the raw thread by `source_id` from `/Users/unoa/Downloads/crawled-data-v2`
- re-parsed raw comment keys from source comments
- rebuilt the expected serialized context using the live `build_context_cpt.py` rules (`제목`, `원글`, optional `부모댓글`, then `댓글` or `답글`)
- compared the reconstructed string to the stored `context_comment` text byte-for-byte

Result: exact reconstruction match `10 / 10`; malformed serialized rows `0`.

Matched samples:

- `cpt_context_stream.jsonl:19568` / `source_id=1189955` / `comment_key=3` / `context_mode=root` — exact match
- `cpt_context_stream.jsonl:29178` / `source_id=1196509` / `comment_key=2-2` / `context_mode=reply` — exact match
- `cpt_context_stream.jsonl:19736` / `source_id=1193811` / `comment_key=2` / `context_mode=root` — exact match
- `cpt_context_stream.jsonl:11205` / `source_id=1196433` / `comment_key=2-4` / `context_mode=reply` — exact match
- `cpt_context_stream.jsonl:8885` / `source_id=1198455` / `comment_key=1-1` / `context_mode=reply` — exact match
- `cpt_context_stream.jsonl:31021` / `source_id=1191404` / `comment_key=5` / `context_mode=root` — exact match
- `cpt_context_stream.jsonl:31414` / `source_id=1200082` / `comment_key=6` / `context_mode=root` — exact match
- `cpt_context_stream.jsonl:23147` / `source_id=1186790` / `comment_key=3` / `context_mode=root` — exact match
- `cpt_context_stream.jsonl:11703` / `source_id=1194412` / `comment_key=1` / `context_mode=root` — exact match
- `cpt_context_stream.jsonl:13807` / `source_id=1195184` / `comment_key=1-1` / `context_mode=reply` — exact match

## Bottom Line

- The cleaned `comment` and reply-side `context_comment` surfaces look materially fixed: tag leakage is zero, duplicate groups are zero, overlap with val is zero, and sampled thread reconstruction remains exact.
- The only remaining failing gate is promo contamination at the `post` level. That residue is much narrower than the pre-clean audit, but it still means the full-file `0 promo rows` criterion is not yet satisfied.
