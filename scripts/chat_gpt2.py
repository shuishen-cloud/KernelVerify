#!/usr/bin/env python3
"""通用 LLM 交互式对话 — 支持 GPT-2 / Qwen / Llama / Phi 等

用法:
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate novel_llm

    # 默认 gpt2 (文本补全模式)
    python scripts/chat.py

    # Instruct 模型 (真正的对话)
    python scripts/chat.py --model Qwen/Qwen2-1.5B-Instruct
    python scripts/chat.py --model meta-llama/Llama-3.2-1B-Instruct
    python scripts/chat.py --model microsoft/phi-2

    # 项目内 tiny GPT-2 (1层, 纯测试)
    python scripts/chat.py --tiny
"""
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Config, GPT2LMHeadModel


# ---- 极小 GPT-2 (项目导出实验用) ----
def build_tiny_gpt2():
    config = GPT2Config(vocab_size=50257, n_embd=64, n_layer=1, n_head=4, n_positions=64)
    return GPT2LMHeadModel(config).eval()


# ---- 模型加载 ----
def load_model(model_name: str, tiny: bool, device: str):
    if tiny:
        print("模型: 自建 tiny GPT-2 (1层, 64 hidden)")
        model = build_tiny_gpt2().to(device)
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        return model, tokenizer, "completion"

    print(f"模型: {model_name} (首次运行会自动下载)")

    # Instruct 模型有的需要 trust_remote_code
    trust_remote = any(x in model_name.lower() for x in ["qwen", "baichuan", "chatglm"])
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=trust_remote, torch_dtype=torch.float16,
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote)

    # 判断是否有 chat_template → 决定用哪种对话模式
    mode = "chat" if tokenizer.chat_template else "completion"
    return model, tokenizer, mode


# ---- 生成 ----
def generate_chat(model, tokenizer, messages: list, device: str,
                  max_tokens: int, temperature: float, top_p: float):
    """Instruct 模型: 用 chat_template 生成."""
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_tokens,
            temperature=temperature, top_p=top_p, do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def generate_completion(model, tokenizer, text: str, device: str,
                        max_tokens: int, temperature: float, top_p: float):
    """Base 模型: 纯文本续写."""
    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_tokens,
            temperature=temperature, top_p=top_p, do_sample=True,
            pad_token_id=tokenizer.eos_token_id, repetition_penalty=1.1,
        )
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


# ---- 主循环 ----
def main():
    parser = argparse.ArgumentParser(description="LLM 对话")
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    if device == "cuda":
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"显存: {vram:.1f} GB")

    model, tokenizer, mode = load_model(args.model, args.tiny, device)
    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"参数量: {param_count:.1f}M")
    print(f"模式: {'💬 对话' if mode == 'chat' else '📝 文本补全'}")
    print()

    if mode == "completion" and not args.tiny:
        print("⚠️  GPT-2 是文本补全模型，不是聊天模型。它只会续写你的输入。")
        print('   想对话请用: python scripts/chat.py --model Qwen/Qwen2-1.5B-Instruct')
        print()
    if args.tiny:
        print("⚠️  1 层 64 hidden 极小模型，输出乱码，仅供测试。")

    print("输入 /quit 退出  /clear 清空对话")
    print("=" * 50)

    if mode == "chat":
        # Instruct 模型: 用 messages 列表管理对话
        messages = []
        while True:
            try:
                user_input = input("\n🧑 > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n再见~"); break
            if not user_input:
                continue
            if user_input == "/quit":
                print("再见~"); break
            if user_input == "/clear":
                messages = []; print("对话已清空"); continue

            messages.append({"role": "user", "content": user_input})
            response = generate_chat(
                model, tokenizer, messages, device,
                args.max_tokens, args.temperature, 0.95,
            )
            messages.append({"role": "assistant", "content": response})
            print(f"\n🤖 > {response}")

    else:
        # Completion 模型: 拼接所有历史
        history = ""
        while True:
            try:
                user_input = input("\n🧑 > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n再见~"); break
            if not user_input:
                continue
            if user_input == "/quit":
                print("再见~"); break
            if user_input == "/clear":
                history = ""; print("对话已清空"); continue

            full = history + user_input
            response = generate_completion(
                model, tokenizer, full, device,
                args.max_tokens, args.temperature, 0.95,
            )
            print(f"\n🤖 > {response}")
            history = full + response + "\n"


if __name__ == "__main__":
    main()
