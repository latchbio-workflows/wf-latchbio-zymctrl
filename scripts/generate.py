import argparse
import math
import os
import sys

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, GPT2LMHeadModel

sys.stdout.reconfigure(line_buffering=True)


def remove_characters(sequence, char_list):
    "This function removes special tokens used during training."
    columns = sequence.split("<sep>")
    seq = columns[1]
    for char in char_list:
        seq = seq.replace(char, "")
    return seq


def calculatePerplexity(input_ids, model, tokenizer):
    "This function computes perplexities for the generated sequences"
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    loss, logits = outputs[:2]
    return math.exp(loss)


def main(label, model, special_tokens, device, tokenizer, num_return_sequences):
    # Generating sequences
    input_ids = tokenizer.encode(label, return_tensors="pt").to(device)
    outputs = model.generate(
        input_ids,
        top_k=9,  # tbd
        repetition_penalty=1.2,
        max_length=1024,
        eos_token_id=1,
        pad_token_id=0,
        do_sample=True,
        num_return_sequences=num_return_sequences,
    )  # Depending non your GPU, you'll be able to generate fewer or more sequences. This runs in an A40.

    # Check sequence sanity, ensure sequences are not-truncated.
    # The model will truncate sequences longer than the specified max_length (1024 above). We want to avoid those sequences.
    new_outputs = [output for output in outputs if output[-1] == 0]
    if not new_outputs:
        print("not enough sequences with short lengths!!")

    # Compute perplexity for every generated sequence in the batch
    ppls = [
        (tokenizer.decode(output), calculatePerplexity(output, model, tokenizer))
        for output in new_outputs
    ]

    # Sort the batch by perplexity, the lower the better
    ppls.sort(key=lambda i: i[1])  # duplicated sequences?

    # Final dictionary with the results
    sequences = {}
    sequences[label] = [(remove_characters(x[0], special_tokens), x[1]) for x in ppls]

    return sequences


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ec_number", required=True, help="EC number to generate sequences for"
    )
    parser.add_argument(
        "--output_dir", required=True, help="Output directory for sequences"
    )
    parser.add_argument("--model_path", default="AI4PD/ZymCTRL", help="Path to model")
    parser.add_argument("--num_batches", default="20", help="Number of batches")
    parser.add_argument(
        "--num_return_sequences", default="20", help="Number of sequences per batch"
    )
    args = parser.parse_args()

    device = torch.device(
        "cuda"
    )  # Replace with 'cpu' if you don't have a GPU - but it will be slow
    print("Reading pretrained model and tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = GPT2LMHeadModel.from_pretrained(args.model_path).to(device)
    special_tokens = ["<start>", "<end>", "<|endoftext|>", "<pad>", " ", "<sep>"]

    os.makedirs(args.output_dir, exist_ok=True)

    # Run 100 batches for the EC number
    for i in range(0, int(args.num_batches)):
        sequences = main(
            args.ec_number,
            model,
            special_tokens,
            device,
            tokenizer,
            args.num_return_sequences,
        )
        for key, value in sequences.items():
            for index, val in enumerate(value):
                # Sequences will be saved with the name of the label followed by the batch index,
                # and the order of the sequence in that batch.
                fn = open(f"{args.output_dir}/{key}_{i}_{index}.fasta", "w")
                fn.write(f">{key}_{i}_{index}\t{val[1]}\n{val[0]}")
                fn.close()
