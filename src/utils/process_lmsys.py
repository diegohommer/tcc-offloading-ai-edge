"""Processes the LMSYS Chatbot Arena dataset to extract token count distributions.

This script downloads a sample of the LMSYS-Chat-1M dataset, tokenizes the
first turn of each conversation (user prompt and assistant response) using
a standard tokenizer, and exports the token lengths to a Parquet file.
This extracted data is used to simulate realistic AI workloads in the PON
network edge-offloading simulator.
"""

import os
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer


def extract_token_counts(conversations, tokenizer):
    """Extracts token counts for the first user-assistant interaction.

    Analyzes a conversation array and counts the tokens for the initial
    user prompt and the corresponding assistant generation.

    Args:
        conversations (list of dict): A list of dictionaries representing the
            conversation turns. Expected keys inside dicts are 'role' and 'content'.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer used to
            encode the text and calculate token lengths.

    Returns:
        tuple: A tuple containing:
            - prompt_tokens (int): Number of tokens in the user's prompt.
            - generation_tokens (int): Number of tokens in the assistant's response.
            Returns (None, None) if the conversation format is invalid or missing.
    """
    if (
        len(conversations) >= 2
        and conversations[0]["role"] == "user"
        and conversations[1]["role"] == "assistant"
    ):
        prompt_text = conversations[0]["content"]
        response_text = conversations[1]["content"]

        # Calculate token lengths without truncation to get the true size
        prompt_tokens = len(tokenizer.encode(prompt_text, truncation=False))
        generation_tokens = len(tokenizer.encode(response_text, truncation=False))

        return prompt_tokens, generation_tokens

    return None, None


def main():
    """Executes the dataset processing pipeline.

    Loads the dataset from Hugging Face, iterates through a sample size,
    extracts token distributions, and saves the resulting DataFrame as a
    Parquet file in the 'data/processed' directory.
    """
    print("1. Loading the lmsys-chat-1m dataset (This may take a few minutes the first time)...")

    # Load a sample (e.g., 100,000 rows) to prevent RAM exhaustion.
    # To process the entire dataset, remove 'streaming=True' and the '.take()' call.
    dataset = load_dataset("lmsys/lmsys-chat-1m", split="train", streaming=True, token=True)
    sample_size = 100000
    dataset_sample = dataset.take(sample_size)

    print("2. Loading Tokenizer (Using GPT-2 as a fast baseline)...")
    # Note: For TCC purposes, GPT-2 provides a good fast approximation of token counts.
    # If edge hardware runs Llama specifically, you can change this to "huggyllama/llama-7b"
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    simulation_data = []

    print("3. Extracting conversations and calculating tokens...")
    for idx, row in enumerate(dataset_sample):
        if idx > 0 and idx % 10000 == 0:
            print(f"Processing row {idx} / {sample_size}...")

        conversations = row.get("conversation", [])
        prompt_tokens, generation_tokens = extract_token_counts(conversations, tokenizer)

        if prompt_tokens is not None and generation_tokens is not None:
            simulation_data.append(
                {
                    "conversation_id": row.get("conversation_id", idx),
                    "model": row.get("model", "unknown"),
                    "prompt_tokens": prompt_tokens,
                    "generation_tokens": generation_tokens,
                }
            )

    # Convert to Pandas DataFrame for easier statistical analysis and export
    df = pd.DataFrame(simulation_data)

    print("\n4. Processed Dataset Statistics:")
    print(df[["prompt_tokens", "generation_tokens"]].describe())

    # Resolve paths relative to this script to ensure correct output directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    processed_dir = os.path.abspath(os.path.join(script_dir, "../../data/processed"))
    raw_dir = os.path.abspath(os.path.join(script_dir, "../../data/raw"))

    # Ensure directories exist
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(raw_dir, exist_ok=True)

    # Save to Parquet (Highly optimized format, much faster/lighter than CSV for the simulator)
    output_path = os.path.join(processed_dir, "lmsys_token_distribution.parquet")
    df.to_parquet(output_path, index=False)

    print(f"\n✅ Success! Data saved to: {output_path}")


if __name__ == "__main__":
    main()
