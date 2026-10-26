# Pre-registration (written 2026-10-26 ~02:30 UTC; the six Haiku generation jobs for the locked test set had been launched but NOT read, scored or looked at by me when this was written)

## Question
Does label-free RL (no collusion labels in the reward; only the SFT's concept directions) make a verbalizer's sentence differ between plainly collusive and honest text, and is that caused by the collusive CONTENT of the RL pool?

## Arms (all start from the base SFT adapter th2_av; KL reference = SFT; 400-step schedule, stopped after step 101; checkpoint at step 100 is the unit of analysis)
- SFT: no RL.   - B: the earlier published RL run (th2_rl_B_av), no collusion data.   - OFFICIAL NLA (kitft Qwen2.5-7B L20): reference only.
- A0 "B-recipe only": every RL step draws from B's public-text pool (mix_orig = 1.0). Isolates "any RL from SFT".
- A1 "honest-only control": 40% B pool / 60% fresh pool of INDEPENDENT+OPEN blatant transcripts (no collusive text).
- A2 "collusion pool": 40% B pool / 60% fresh pool of blatant transcripts in all four versions (collusive, collusive_word, independent, open), unlabeled.
- Identical settings across arms: beta 0.05, lr 2e-5, w_div 0.5, no concept reward on fresh-pool steps, seeds 11, 12, 13 (A0: 11, 12).
- Checkpoint step 100 is FIXED for every arm and seed (it was selected on seed 11 using the earlier held-out sets, so the new locked set and new seeds are the unbiased test). Step 50 is reported as a secondary point. No cherry-picking of seeds or steps; failed/collapsed runs are reported.

## Test sets (verbalizer input = layer-20 (hidden_states[21]) activation at the last token of the raw sentence, no chat template)
- LOCKED-L (new, generated after this file): 12 collusion types never used in any training or earlier test. Cells: collusive-no-cue (pair_implicit collusive), honest-no-cue twin, collusive-with-cue (pair_explicit collusive), honest-with-cue (honest sentences that contain collude/secret/rig/... words but mean something honest). Plus its pairs are matched twins.
- HELD-OUT-578 (earlier, 4 held-out types) and HAND-50 (earlier, hand-written): secondary, already seen once; used for trend consistency, not for the confirmatory claims.

## Metrics
- Flag = output contains a collusion/secrecy word (fixed list in scripts/flag.py). Reported alongside a blind Haiku (<= medium) sentence-only judge score (0-100), 2 passes, judge never sees the input sentence or labels.
- PRIMARY (confirmatory) endpoint: on LOCKED-L pair_implicit pairs, the pair-level balanced accuracy of the flag (collusive flagged, twin not flagged), compared A2 vs A1 and A2 vs SFT, pooled over seeds (bootstrap over pairs, then over seeds).
- SECONDARY: (i) cue control: honest-with-cue flag rate vs collusive-with-cue flag rate (a cue-word reader flags both; an intent reader flags the collusive one more); (ii) judge AUROC on the same pairs; (iii) permutation p-values (shuffle labels within pair); (iv) repetition (distinct %, top-sentence share on 256 real agent messages) and B-held-out concept F1 / held-out-concept naming to show the models did not degrade; (v) reference probes (supervised collusion direction; Core-trained probe) and the official NLA.

## What would count as support / no support
- Support for "collusive CONTENT matters": A2 - A1 > 0 with 95% CI excluding 0 on the primary endpoint, with A2 not worse on the repetition/retention checks.
- Support for the weaker claim "any label-free RL unlocks it": A1 (and A0) clearly above SFT, A2 - A1 not distinguishable from 0.
- If neither: report null.

## Known limitations declared in advance
Synthetic Haiku-written sentences (single generator family); keyword flag is crude (judge added); collusion direction probe is supervised and only a reference; no transfer to realistic agent messages (Core, Transfer, Verdict) beyond the earlier negative results.
