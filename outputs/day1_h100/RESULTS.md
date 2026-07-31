# Day-1 H100 routing results — 2026-07-30

## Rules freeze

- Direct training/LID/LM data: `google/WaxalNLP` train and validation only.
- Phase-2 audio: inference and speech/transcript-derived routing only.
- SALT revision: `7448016c50bdec469b8454c9631c76fc1d1dd40e` (MIT).
- af51 revision: `3648a5b3bec96a9d72c5f96f7d5aa94add2a4a1f` (MIT).
- No external corpus, pseudo-label training, test labels, paid API, or AutoML.

## Validated findings

- Learned char-ngram LID on af51 validation transcripts: 3032/3040 = 99.7368%.
- Frozen SALT-encoder acoustic LID: 3038/3040 = 99.9342%.
- Fused LID: 3038/3040 = 99.9342%.
- Phase-2 fused routing: Ach 500 / Nyn 500 / Myx 499 / Xog 1.
- The old hand-written routing's 84 Xog rows were reassigned to Myx 80 / Nyn 4.
- Full SALT raw macro error:
  - Auto language: 0.2596863
  - True language forced: 0.2502131
- True-language forcing beat auto for all four validation languages.

## Candidate submissions

All files have 1500 rows, 1500 unique IDs, correct test order, and zero empty targets.

1. `submissions/phase2_routingfix_conservative.csv`
   - First public probe.
   - Keeps the exact current 0.699825 output on old Ach/Nyn/Myx rows.
   - Replaces only old `xog` and `unk` rows.
   - 106 transcript changes versus the current best.
2. `submissions/phase2_salt_fusedlid_lp08_oldmyxadapter.csv`
   - Corrected routing + LP0.8 base, while retaining the field-tested adapter on the
     original 267 Myx rows.
   - 147 transcript changes versus the current best.
3. `submissions/phase2_salt_fusedlid_lp08.csv`
   - Pure corrected-routing SALT LP0.8 candidate without the adapter.
   - 137 changes versus the selected LP0.8 composite and 401 versus the current best.

Do not replace either selected private-leaderboard submission until a new candidate
has a public score. Submit candidate 1 first and record its score before candidate 2.
