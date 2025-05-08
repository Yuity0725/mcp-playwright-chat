
# DeepSeek‑R1 Overview  
*Last updated: 2025-05-07*

## 1. What is DeepSeek‑R1?  
DeepSeek‑R1 is a **reasoning‑centric large language model (LLM)** released in *January 2025* by the Beijing‑based AI lab **DeepSeek**. It is the first public model series to:

* Incentivize chain‑of‑thought reasoning **purely via Reinforcement Learning (RL)**, without an initial Supervised Fine‑Tuning (SFT) step. citeturn2view0  
* Deliver performance **on par with OpenAI‑o1** across math, coding, and complex reasoning tasks while being fully open‑source under the MIT license. citeturn0search0turn0news29  

## 2. Model Line‑up  

| Variant | Total Params | Activated Params (MoE) | Context Len | Purpose |
|---------|--------------|------------------------|-------------|---------|
| **DeepSeek‑R1‑Zero** | 671 B | 37 B | 128 K | RL‑only “cold‑start” model (no SFT) |
| **DeepSeek‑R1** | 671 B | 37 B | 128 K | Two SFT + two RL stages (flagship) |
| **R1‑Distill (1.5 B – 70 B)** | 1.5‑70 B | dense | 32 K | Smaller dense checkpoints distilled from R1 | citeturn2view0

> **MoE** = Mixture‑of‑Experts; only a subset of parameters are active per token.

## 3. Training Pipeline (4‑Stage)  

1. **SFT‑Seed Ⅰ** – Small curated instruction set to bootstrap capabilities.  
2. **RL‑Discover Ⅰ** – Encourage the model to explore long chain‑of‑thoughts, yielding R1‑Zero.  
3. **SFT‑Seed Ⅱ** – Incorporate cold‑start data to improve fluency & coverage.  
4. **RL‑Align Ⅱ** – Align reasoning patterns with human preference & safety, producing R1. citeturn2view0  

![](https://raw.githubusercontent.com/deepseek-ai/public-assets/main/r1_pipeline_diagram.png) <!-- placeholder diagram path -->

## 4. Performance Highlights (May 2025 Eval)  

| Category | Benchmark | OpenAI‑o1 | DeepSeek‑R1 | GPT‑4o 0513 | Claude‑3.5 Sonnet |
|----------|-----------|-----------|-------------|--------------|-------------------|
| **Math** | AIME‑2024 (Pass@1) | 79.2 | **79.8** | 63.6 | 16.0 |
| **Code** | Codeforces Rating | 2061 | **2029** | 1820 | 717 |
| **Reason.** | MMLU (Pass@1) | 91.8 | **90.8** | 87.2 | 88.3 | citeturn5view0

> 🔎 *DeepSeek‑R1 consistently scores within ~1‑2 % of GPT‑4o on common reasoning tasks, and often surpasses OpenAI‑o1‑mini.*

## 5. Distillation Breakthrough  

DeepSeek demonstrates that **RL‑discovered reasoning traces can be transferred to smaller dense models**. The 32 B Qwen‑based distillation *outperforms OpenAI‑o1‑mini* on several benchmarks while being **23× smaller**. citeturn2view0  

## 6. Licensing & Ecosystem  

* **MIT License** – Free for commercial and research use, including derivative works. citeturn0search0  
* Official resources:  
  * 📝 [arXiv paper (2501.12948)](https://arxiv.org/abs/2501.12948)  
  * 🤗 [Hugging Face Model Card](https://huggingface.co/deepseek-ai/DeepSeek-R1)  
  * 🌐 [Chat Demo](https://chat.deepseek.com) / [API Docs](https://api-docs.deepseek.com)  

## 7. Strengths & Opportunities  

* **Open & Reproducible** – First high‑end model (671 B MoE) fully open‑sourced with weights.  
* **Cost‑Efficient R&D** – RL pipeline slashes human‑label cost and compute vs. SFT‑heavy recipes. citeturn0news30  
* **Benchmark Versatility** – Top‑tier results in math, code, multilingual reasoning, and Chinese tasks.  

## 8. Limitations & Considerations  

* **Safety / Policy Alignment** – RL alignment tuned primarily for English & Chinese; other languages may lag.  
* **Censorship Concerns** – Being China‑based, hosted APIs may apply additional content filters. citeturn0news29  
* **Resource Needs** – Full 671 B MoE requires > 2× A100‑80GB (or equivalent) for inference; distill models recommended for local use.  

## 9. Getting Started Locally  

```bash
pip install transformers accelerate
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1",
                                             device_map="auto", trust_remote_code=True)
tok   = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1")
```

> 🧩 *See the model card for FP8 & Flash‑Attention 2 tips.*

## 10. Further Reading  

1. DeepSeek‑R1 GitHub repo – training scripts & config.  
2. Community eval notes on *r/LocalLLaMA* (“how good vs. o1”).   
3. Business Insider & Financial Times coverage on the cost‑efficiency of R1.  

---  
© 2025-05-07 DeepSeek‑AI / Community summaries. Feel free to reuse under MIT.
