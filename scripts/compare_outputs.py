"""
Reads exp1_outputs.json, exp2_outputs.json, exp3_outputs.json from analysis/
and writes analysis/output_comparison.md with all 30 prompts side-by-side
across the three experiments.
"""

import json
import os

ANALYSIS_DIR = os.path.join(os.path.dirname(__file__), "../analysis")

EXP_FILES = [
    ("BF16 Baseline", "exp1_outputs.json"),
    ("FP8", "exp2_outputs.json"),
    ("FP8 + CUDA Graphs", "exp3_outputs.json"),
]


def load(filename):
    path = os.path.join(ANALYSIS_DIR, filename)
    with open(path) as f:
        return json.load(f)


def main():
    experiments = []
    for label, filename in EXP_FILES:
        data = load(filename)
        experiments.append((label, data))

    lines = ["# Response Comparison Across Experiments", ""]
    lines.append("All 30 prompts run with the same sampling params (`max_tokens=150`, `temperature=0.3`).")
    lines.append("Responses may differ due to quantization rounding and CUDA graph replay ordering.\n")

    # Summary table
    lines.append("## Throughput Summary\n")
    lines.append("| Experiment | Throughput (tok/s) | Wall Time (s) | Avg Latency (s) |")
    lines.append("|---|---|---|---|")
    for label, data in experiments:
        m = data["metrics"]
        lines.append(
            f"| {label} | {m['throughput_tok_s']} | {m['wall_time_s']} | {m['avg_latency_s']} |"
        )
    lines.append("")

    # Per-prompt comparison
    lines.append("---\n")
    lines.append("## Per-Prompt Responses\n")

    n = len(experiments[0][1]["responses"])
    for i in range(n):
        prompt = experiments[0][1]["responses"][i]["prompt"]
        lines.append(f"### Prompt {i + 1}")
        lines.append(f"> {prompt}\n")

        for label, data in experiments:
            r = data["responses"][i]
            lines.append(f"**{label}** ({r['tokens_generated']} tokens)")
            lines.append(f"{r['response']}\n")

        lines.append("---\n")

    out_path = os.path.join(ANALYSIS_DIR, "output_comparison.md")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Comparison written to {os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()
