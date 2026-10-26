# Official NLA comparison
Runs kitft/nla-qwen2.5-7b-L20-av/-ar (official Qwen2.5-7B NLA) on the same activations as our own verbalizers.
- run_official.py : AV generation (greedy, scale 150) + AR cosine, run on a separate A40 pod (/root/official_nla)
- analyze.py      : builds outputs/comparison_official_vs_ours.txt
- outputs/        : av_outputs.json (full text), ar_cos.json, comparison file
Inputs (activations) are not committed (see .gitignore).
