from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-3.2-1B", gpu_memory_utilization=0.8, quantization="fp8")
params = SamplingParams(temperature=0.7, max_tokens=128)

prompts = [
    "What are the best places to visit in Japan?",
    "How do you make a classic margherita pizza from scratch?",
    "Explain the causes of World War I.",
    "What are some good habits to build in your 20s?",
    "How does the stock market work?",
    "What is the fastest animal on earth and how does it move?",
    "Give me a brief history of the Roman Empire.",
    "What are the health benefits of regular exercise?",
    "How do airplanes generate lift?",
    "What makes a good leader?",
    "Explain how the human immune system works.",
    "What are the most spoken languages in the world?",
    "How do black holes form?",
    "What is the Mediterranean diet?",
    "How does compound interest work?",
    "What are some tips for learning a new language?",
    "Explain the water cycle.",
    "What is the history of the Olympics?",
    "How do vaccines work?",
    "What are the seven wonders of the ancient world?",
]

outputs = llm.generate(prompts, params)

for o in outputs:
    print(o.outputs[0].text[:80])
