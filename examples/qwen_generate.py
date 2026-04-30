"""UniVis Qwen2.5-0.5B demo: easiest way using model.generate().

Usage (on L20 server or any GPU machine):
    python examples/qwen_generate.py

This is the simplest integration: just pass a logits_processor to model.generate().
UniVis handles all the tracking automatically. No manual loop needed.
"""

import os
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import univis


MODEL_NAME = os.environ.get('UNIVIS_MODEL', 'Qwen/Qwen2.5-0.5B-Instruct')
PROMPT = 'Explain the concept of computational redundancy in neural networks.'


def main() -> None:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')
    print(f'Model:  {MODEL_NAME}')

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
        device_map='auto',
    )
    model.eval()

    tracker = univis.attach(model, project='qwen_generate_test', transport='file')
    logits_processor = tracker.logits_processor(tokenizer)

    input_ids = tokenizer(PROMPT, return_tensors='pt').input_ids.to(device)
    print(f'Prompt: "{PROMPT[:60]}..." ({input_ids.shape[1]} tokens)')

    with torch.no_grad():
        output = model.generate(
            input_ids,
            logits_processor=[logits_processor],
            max_new_tokens=50,
        )

    report_path = tracker.finish()

    generated = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f'\nGenerated: {generated[:200]}...')
    print(f'Report:    {report_path}')


if __name__ == '__main__':
    main()
