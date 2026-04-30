"""UniVis basic demo: track GPT-2 inference and generate analysis report.

Usage:
    python examples/gpt2_basic.py

Uses tiny-gpt2 by default (no GPU needed).
Set HF_ENDPOINT=https://hf-mirror.com if in China.
"""

import os
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys
from pathlib import Path
# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import univis


MODEL_NAME = os.environ.get('UNIVIS_MODEL', 'sshleifer/tiny-gpt2')
PROMPT = 'The meaning of life is'
MAX_TOKENS = 30


def main() -> None:
    print(f'Loading {MODEL_NAME}...')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.eval()

    input_ids = tokenizer(PROMPT, return_tensors='pt').input_ids
    print(f'Prompt: "{PROMPT}" ({input_ids.shape[1]} tokens)')
    print(f'Generating {MAX_TOKENS} tokens...\n')

    # Attach UniVis tracker
    tracker = univis.attach(model, project='gpt2_demo', transport='file')

    # Manual inference loop
    with torch.no_grad():
        for i in range(MAX_TOKENS):
            outputs = model(input_ids)
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            token_text = tokenizer.decode(next_token[0])
            input_ids = torch.cat([input_ids, next_token], dim=-1)

            tracker.on_step(
                token_index=i,
                generated_token=token_text,
                logits=outputs.logits[:, -1, :],
            )

    # Finish and generate report
    report_path = tracker.finish()
    generated = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    print(f'Generated: {generated}')
    print(f'Report:    {report_path}')
    print(f'Data:      univis_data_*.jsonl')


if __name__ == '__main__':
    main()
