# Approved concept directions (layer 20, Qwen2.5-7B-Instruct)

Rule (fixed in advance): >= 2 independent datasets; leave-one-dataset-out AUROC mean >= 0.85 and worst >= 0.75; and mean ablation raises the loss of the concept's texts more than matched other texts and random directions (CI above 0). Direction = mean difference (or logistic probe where only that earned the readout; the use test uses the same vector). Effect grade: strong >= 0.10 nats/token, moderate 0.03-0.10, weak < 0.03 (statistically positive, practically small).

**53 approved**: {'strong': 24, 'moderate': 15, 'weak': 14}. All concepts: {'APPROVED': 53, 'READOUT-ONLY': 10, 'NOT APPROVED': 34, 'SINGLE-DATASET': 19}.

| concept | group | direction | datasets | AUROC (worst) | keyword | ablation (nats/token) | effect | evidence |
|---|---|---|---|---|---|---|---|---|
| arabic | form | mean | 2 | 1.00 (1.00) | 0.99 | +4.193 | strong | public datasets |
| french | form | mean | 2 | 1.00 (1.00) | 0.99 | +3.135 | strong | public datasets |
| japanese | form | mean | 2 | 1.00 (1.00) | 0.90 | +2.187 | strong | public datasets |
| german | form | mean | 2 | 1.00 (1.00) | 1.00 | +1.222 | strong | public datasets |
| spanish | form | mean | 2 | 1.00 (1.00) | 0.99 | +0.898 | strong | public datasets |
| non_english | form | mean | 2 | 0.98 (0.97) | 0.99 | +0.761 | strong | public datasets |
| russian | form | mean | 2 | 1.00 (0.99) | 0.97 | +0.245 | strong | public datasets |
| future_reference | form | mean | 2 | 1.00 (1.00) | 1.00 | +0.192 | strong | written data |
| structured_data | form | mean | 2 | 1.00 (1.00) | 1.00 | +0.108 | strong | written data |
| past_tense | form | mean | 2 | 0.95 (0.90) | 0.95 | +0.088 | moderate | written data |
| chinese | form | mean | 2 | 0.99 (0.99) | 0.85 | +0.058 | moderate | public datasets |
| place_entity | form | mean | 2 | 0.92 (0.88) | 0.73 | +0.027 | weak | public datasets |
| eval_awareness | safety | mean | 2 | 0.97 (0.94) | 0.96 | +0.123 | strong | written data |
| prompt_injection | safety | logreg | 2 | 0.90 (0.82) | 0.73 | +0.034 | moderate | public datasets |
| jailbreak | safety | logreg | 2 | 0.89 (0.81) | 0.86 | +0.029 | weak | public datasets |
| secrecy | safety | mean | 2 | 0.99 (0.98) | 0.99 | +0.018 | weak | written data |
| deception | safety | logreg | 2 | 0.96 (0.94) | 0.65 | +0.009 | weak | written data |
| sexual_content | safety | logreg | 2 | 0.96 (0.96) | 0.85 | +0.007 | weak | public datasets |
| discrimination | safety | logreg | 2 | 0.93 (0.92) | 0.85 | +0.006 | weak | public datasets |
| illegal_activity | safety | logreg | 2 | 0.86 (0.86) | 0.78 | +0.005 | weak | public datasets |
| req_brainstorm | task | mean | 2 | 0.90 (0.84) | 0.90 | +0.499 | strong | public datasets |
| question_form | task | mean | 2 | 0.87 (0.77) | 0.82 | +0.306 | strong | public datasets |
| req_creative | task | mean | 2 | 0.86 (0.81) | 0.83 | +0.277 | strong | public datasets |
| req_howto | task | mean | 2 | 0.96 (0.92) | 0.97 | +0.142 | strong | written data |
| req_translate | task | mean | 2 | 0.96 (0.95) | 0.97 | +0.069 | moderate | written data |
| formal | tone | mean | 2 | 0.93 (0.89) | 0.80 | +0.377 | strong | public datasets |
| pride | tone | mean | 2 | 0.86 (0.84) | 0.72 | +0.100 | moderate | public datasets |
| mental_health | tone | mean | 2 | 0.94 (0.93) | 0.92 | +0.039 | moderate | cross-community (Reddit) |
| hedging | tone | logreg | 2 | 1.00 (1.00) | 0.99 | +0.013 | weak | written data |
| weather | topic | mean | 2 | 0.90 (0.85) | 0.93 | +0.583 | strong | public datasets |
| philosophy | topic | mean | 2 | 0.97 (0.96) | 0.77 | +0.574 | strong | public datasets |
| vehicles | topic | mean | 5 | 0.89 (0.77) | 0.87 | +0.159 | strong | public datasets |
| cryptocurrency | topic | mean | 2 | 0.95 (0.94) | 0.97 | +0.155 | strong | cross-community (Reddit) |
| environment | topic | mean | 2 | 0.87 (0.81) | 0.74 | +0.155 | strong | public datasets |
| computing_hw | topic | mean | 4 | 0.91 (0.86) | 0.85 | +0.139 | strong | public datasets |
| technology | topic | mean | 8 | 0.92 (0.75) | 0.87 | +0.130 | strong | public datasets |
| programming | topic | mean | 3 | 0.96 (0.94) | 0.96 | +0.108 | strong | public datasets |
| food | topic | mean | 5 | 0.95 (0.93) | 0.91 | +0.108 | strong | public datasets |
| religion | topic | mean | 5 | 0.92 (0.83) | 0.85 | +0.095 | moderate | public datasets |
| mathematics | topic | mean | 4 | 0.90 (0.78) | 0.88 | +0.080 | moderate | public datasets |
| politics | topic | mean | 7 | 0.91 (0.84) | 0.84 | +0.074 | moderate | public datasets |
| work_career | topic | mean | 2 | 0.93 (0.92) | 0.93 | +0.073 | moderate | cross-community (Reddit) |
| sports | topic | mean | 9 | 0.95 (0.88) | 0.94 | +0.070 | moderate | public datasets |
| books | topic | mean | 3 | 0.92 (0.89) | 0.91 | +0.066 | moderate | public datasets |
| pets | topic | mean | 2 | 0.98 (0.97) | 0.99 | +0.063 | moderate | cross-community (Reddit) |
| drugs | topic | mean | 4 | 0.89 (0.81) | 0.88 | +0.045 | moderate | public datasets |
| health | topic | mean | 6 | 0.89 (0.84) | 0.83 | +0.043 | moderate | public datasets |
| fashion_beauty | topic | mean | 3 | 0.95 (0.92) | 0.89 | +0.029 | weak | public datasets |
| music | topic | mean | 5 | 0.94 (0.85) | 0.95 | +0.028 | weak | public datasets |
| travel | topic | logreg | 5 | 0.96 (0.90) | 0.88 | +0.021 | weak | public datasets |
| law | topic | mean | 4 | 0.91 (0.88) | 0.86 | +0.020 | weak | public datasets |
| lgbtq | topic | logreg | 3 | 0.96 (0.93) | 0.90 | +0.005 | weak | public datasets |
| relationships | topic | logreg | 3 | 0.96 (0.95) | 0.92 | +0.002 | weak | public datasets |

## Not yet approved

**READOUT-ONLY** (10): refusal (0.98), science (0.91), space (0.91), sentiment_pos (0.91), entertainment (0.91), sentiment_neg (0.90), family (0.88), req_classify (0.88), person_entity (0.87), disgust (0.86)

**SINGLE-DATASET** (19): req_code (1.00), subjective (0.99), code_present (0.99), informal (0.99), req_math (0.99), sycophancy (0.99), organisation (0.99), factual_encyclopedic (0.98), scientific_text (0.98), narrative (0.98), polite (0.98), req_rewrite (0.97), fiction (0.97), numbers (0.96), math_notation (0.96), curiosity (0.90), req_support (0.88), req_command (0.88), self_harm (0.88)

**NOT APPROVED** (34): education (0.87), finance (0.85), true_claim (0.85), req_open_qa (0.83), gratitude (0.83), gaming (0.81), joy (0.80), req_extract (0.80), rude (0.79), privacy_pii (0.79), humor (0.79), anger (0.79), fear (0.78), req_advice (0.78), firearms (0.77), req_chitchat (0.76), home_living (0.76), surprise (0.76), history (0.76), love (0.75), sadness (0.74), art_culture (0.73), guilt_shame (0.71), toxicity (0.71), misinformation (0.69), harmful_request (0.66), hate_speech (0.64), violence (0.61), objective (0.59), shopping (0.54), sarcasm (0.49), crime_conflict (0.47), req_summarize (0.45), first_person

Caveats: written-data concepts use two sets from the same generator (different styles); sycophancy's second set was template-made and excluded. 'cross-community' = two disjoint groups of subreddits (same platform). The use test is teacher-forced likelihood, not free-generation behaviour; behaviour concepts (refusal, jailbreak, prompt_injection, deception, secrecy, ...) still owe a behavioural test. Best-of-two direction methods is a mild selection effect.
