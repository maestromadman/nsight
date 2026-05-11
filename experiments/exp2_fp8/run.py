"""
Experiment 2 — FP8 Quantization
Weights are quantized to FP8 (8-bit float), cutting model memory roughly
in half vs BF16. This frees KV cache headroom and reduces memory-bandwidth
pressure on matrix-vector products during decode.
enforce_eager=True is kept so CUDA graph effects don't mix with quantization.
All timing and throughput measurements come from nsys, not this script.
"""

import json
import os
from vllm import LLM, SamplingParams

SYSTEM_MESSAGE = (
    "You are a helpful customer support assistant for a financial services company. "
    "Answer clearly and concisely."
)

CUSTOMER_PROMPTS = [
    "I was charged twice for the same transaction on my debit card yesterday. How do I get a refund?",
    "My account has been locked after too many failed login attempts. How do I regain access?",
    "I noticed an international transaction I don't recognize. Can you tell me what country it came from?",
    "I tried to make a payment and it was declined even though I have enough funds. What's going on?",
    "How long does it take for a dispute to be resolved once I file one?",
    "I lost my credit card. How do I freeze it immediately without calling in?",
    "My direct deposit hasn't arrived and my employer says they sent it two days ago. Where is it?",
    "Can you explain why my credit score dropped 40 points last month when I didn't miss any payments?",
    "I want to set up automatic bill payments for my utilities. How do I do that?",
    "I received a text saying my account was compromised. Is this from your bank or is it a scam?",
    "I was charged a foreign transaction fee on a purchase made inside the US. Is that correct?",
    "How do I add my spouse as a joint account holder on my checking account?",
    "I submitted a loan application three weeks ago and haven't heard anything. What's the status?",
    "What documents do I need to bring to open a business checking account?",
    "My mortgage payment processed twice this month. Who do I contact to get one reversed?",
    "I'm traveling abroad next week. How do I notify the bank so my card isn't blocked?",
    "I deposited a check six days ago and the funds still haven't cleared. When will they be available?",
    "Can I get a temporary credit limit increase for a large purchase I'm planning this weekend?",
    "I received a paper statement fee charge but I signed up for paperless billing. Can you remove it?",
    "How do I dispute a merchant charge where the service was never delivered?",
    "My savings account interest rate changed without any notice. Why did that happen?",
    "I want to transfer money to an account at a different bank. What are the transfer limits?",
    "I see a pending charge from three weeks ago that never posted. Should I be concerned?",
    "My debit card chip isn't working at terminals. Do I need a replacement card?",
    "What is the overdraft protection policy and how do I opt into it?",
    "I closed my account last month but still received a monthly fee charge. How do I get that back?",
    "Can I get a breakdown of all the fees I've paid on my account over the last year?",
    "I need proof of account ownership for a lease application. What documents can you provide?",
    "My auto loan payment didn't apply to the principal like I requested. Can you fix that?",
    "I want to set up a high-yield savings account but I'm not sure which one fits my needs best.",
]


def build_conversations():
    conversations = []
    for question in CUSTOMER_PROMPTS:
        conversations.append(
            f"<|system|>\n{SYSTEM_MESSAGE}\n<|user|>\n{question}\n<|assistant|>\n"
        )
    return conversations


def main():
    print("=" * 60)
    print("Experiment 2: FP8 Quantization (enforce_eager=True)")
    print("=" * 60)

    llm = LLM(
        model="meta-llama/Llama-3.1-8B-Instruct",
        dtype="bfloat16",
        quantization="fp8",
        enforce_eager=True,
        enable_prefix_caching=False,
        gpu_memory_utilization=0.90,
        max_model_len=4096,
    )

    sampling_params = SamplingParams(max_tokens=150, temperature=0.3)
    prompts = build_conversations()

    # warm-up: forces kernel compilation before the profiled run
    _ = llm.generate(prompts[:2], sampling_params)

    outputs = llm.generate(prompts, sampling_params)

    token_counts = [len(o.outputs[0].token_ids) for o in outputs]
    total_tokens = sum(token_counts)

    print(f"\n{'Requests processed':<35} {len(outputs):>10}")
    print(f"{'Total output tokens':<35} {total_tokens:>10}")
    print(f"{'Min tokens generated':<35} {min(token_counts):>10}")
    print(f"{'Max tokens generated':<35} {max(token_counts):>10}")
    print(f"{'Avg tokens per response':<35} {total_tokens / len(outputs):>10.1f}")
    print("\nTiming and throughput: see nsys stats in analysis/exp2_stats.txt")

    out_path = os.path.join(os.path.dirname(__file__), "../../analysis/exp2_outputs.json")
    result = {
        "experiment": "exp2_fp8",
        "config": {
            "dtype": "bfloat16",
            "quantization": "fp8",
            "enforce_eager": True,
            "gpu_memory_utilization": 0.90,
            "max_model_len": 4096,
            "max_tokens": 150,
            "temperature": 0.3,
        },
        "token_counts": {
            "total": total_tokens,
            "min": min(token_counts),
            "max": max(token_counts),
            "avg": round(total_tokens / len(outputs), 1),
        },
        "responses": [
            {
                "index": i,
                "prompt": CUSTOMER_PROMPTS[i],
                "response": outputs[i].outputs[0].text.strip(),
                "tokens_generated": token_counts[i],
            }
            for i in range(len(outputs))
        ],
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Responses saved to {os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()
