# finetune.py
#
# A unified script for fine-tuning both Decoder-only (e.g., GPT-2) and
# Encoder-Decoder (e.g., T5) models on the samsum dataset using LoRA
# and 8-bit quantization to fit on an 8GB GPU.
#
# Download the samsum dataset from Hugging Face datasets:
# git clone https://huggingface.co/datasets/knkarthick/samsum/
#
# Required installs:
# pip install transformers datasets peft bitsandbytes accelerate
#
# Usage:
# For GPT-2:
# python finetune.py --model_type gpt2 --epochs 1 --output_dir ./gpt2-samsum
#
# For T5:
# python finetune.py --model_type t5 --epochs 1 --output_dir ./t5-samsum
#

import os
import torch
import argparse
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


# --- Helper Function 1: Print Trainable Parameters ---
def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || "
        f"trainable%: {100 * trainable_params / all_param:.2f}"
    )


# --- Helper Function 2: GPT-2 Prompt Template ---
# This helper is provided to the students
def create_prompt_gpt2(sample):
    """
    Creates the instruction prompt for a given sample (GPT-2 specific format).
    """
    prompt_template = (
        "Human: Summarize the following conversation.\n\n"
        "Conversation:\n{dialogue}\n\n"
        "Assistant:\n{summary}"
    )
    # Return prompt with or without summary, depending on what's available
    return prompt_template.format(
        dialogue=sample["dialogue"],
        summary=sample.get("summary", ""),  # Use .get for inference
    )


# =======================================================================
# == STUDENT TASK SECTION ==
# This function is the core of the assignment.
# Students must fill in the logic for both model types.
# =======================================================================


def get_model_components(model_type, lora_r, lora_alpha, lora_dropout):
    """
    Returns the model-specific components based on the model_type.

    STUDENT ASSIGNMENT: Fill in the logic for both model types.
    """

    if model_type == "gpt2":

        # --- STUDENT TASK 1a: GPT-2 (Decoder-only) Preprocessing ---
        def preprocess_function_gpt2(examples, tokenizer):
            # 1. Create the full instruction prompts (using the 'create_prompt_gpt2' helper)
            #    and add the EOS token (tokenizer.eos_token) to the end of each.
            # 2. Tokenize the prompts. Set 'truncation=True', 'max_length=256',
            #    and 'padding="max_length"'.
            # 3. Create the 'labels'. This MUST be a DEEP COPY of the 'input_ids'.
            #    (Hint: `labels = [l.copy() for l in tokenized_outputs["input_ids"]]`)
            # 4. Find the start of the "summary" (the answer) in each tokenized prompt.
            #    (Hint: You'll need to re-tokenize *without* the summary to find the length).
            # 5. Mask the 'labels' *before* the summary start by setting them to -100.
            #    This is the most critical step! We only want the model to learn
            #    to predict the summary, not the prompt.
            # 6. Return the 'tokenized_outputs' dictionary.
            full_instruction_prompts = [
                create_prompt_gpt2({"dialogue": dialogue, "summary": summary})
                + tokenizer.eos_token
                for dialogue, summary in zip(examples["dialogue"], examples["summary"])
            ]

            tokenized_outputs = tokenizer(
                full_instruction_prompts,
                truncation=True,
                max_length=256,
                padding="max_length",
            )

            labels = [l.copy() for l in tokenized_outputs["input_ids"]]

            prompts_without_summary = [
                create_prompt_gpt2({"dialogue": dialogue})
                for dialogue in examples["dialogue"]
            ]

            tokenized_without_summary = tokenizer(
                prompts_without_summary,
                truncation=True,
                max_length=256,
                padding=False,
            )

            for i, prompt_ids in enumerate(tokenized_without_summary["input_ids"]):
                # set the prompt tokens to -100 as the model should not learn the prompt
                prompt_length = len(prompt_ids)
                labels[i][:prompt_length] = [-100] * prompt_length

                # set the end of text tokens to -100 as the model should not learn the padding
                for j in range(prompt_length, len(labels[i])):
                    if tokenized_outputs["attention_mask"][i][j] == 0:
                        labels[i][j] = -100

            tokenized_outputs["labels"] = labels

            return tokenized_outputs

        # --- STUDENT TASK 2a: GPT-2 (Decoder-only) LoRA Config ---
        # 1. Set 'r', 'lora_alpha', and 'lora_dropout' using the function arguments.
        # 2. Set 'bias' to 'none'.
        # 3. Set the 'task_type' to "CAUSAL_LM" (this is crucial for GPT-2).
        # 4. Set the 'target_modules'. For GPT-2, this is typically ["c_attn"].
        #    (You can find this by printing the model architecture).
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["c_attn"],
        )

        return {
            "model_name": "gpt2",
            "model_class": AutoModelForCausalLM,
            "preprocess_function": preprocess_function_gpt2,
            "lora_config": lora_config,
            "data_collator_class": DataCollatorForSeq2Seq,  # We use this for both
        }

    elif model_type == "t5":

        # --- STUDENT TASK 1b: T5 (Encoder-Decoder) Preprocessing ---
        def preprocess_function_t5(examples, tokenizer):
            # 1. Create the input strings. T5 *requires* a prefix for summarization.
            #    (e.g., "summarize: " + dialogue)
            # 2. Create the label strings (just the summaries).
            # 3. Tokenize the 'inputs'. Set 'truncation=True', 'max_length=512',
            #    and 'padding="max_length"'.
            # 4. Tokenize the 'labels' using the 'text_target=' argument.
            #    Also set 'truncation=True', 'max_length=128', and 'padding="max_length"'.
            # 5. Add the tokenized labels to the 'tokenized_inputs' dictionary
            #    under the key "labels".
            # 6. Return 'tokenized_inputs'.
            for i in range(0, len(examples["dialogue"])):
                examples["dialogue"][i] = "summarize: " + examples["dialogue"][i]

            tokenized_inputs = tokenizer(
                examples["dialogue"],
                truncation=True,
                max_length=512,
                padding="max_length",
            )

            tokenized_labels = tokenizer(
                text_target=examples["summary"],
                truncation=True,
                max_length=128,
                padding="max_length",
            )

            # mask the padding tokens
            #label_ids = tokenized_labels["input_ids"]
            #for i in range(len(label_ids)):
            #    label_ids[i] = [
            #        -100 if token == tokenizer.pad_token_id else token
            #        for token in label_ids[i]
            #    ]

            tokenized_inputs["labels"] = tokenized_labels["input_ids"]

            return tokenized_inputs

        # --- STUDENT TASK 2b: T5 (Encoder-Decoder) LoRA Config ---
        # 1. Set 'r', 'lora_alpha', and 'lora_dropout'.
        # 2. Set 'bias' to 'none'.
        # 3. Set the 'task_type' to "SEQ_2_SEQ_LM" (this is crucial for T5).
        # 4. Set the 'target_modules'. For T5, this is typically ["q", "v"].
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="SEQ_2_SEQ_LM",
            target_modules=["q", "v"],
        )

        return {
            "model_name": "t5-small",
            "model_class": AutoModelForSeq2SeqLM,
            "preprocess_function": preprocess_function_t5,
            "lora_config": lora_config,
            "data_collator_class": DataCollatorForSeq2Seq,
        }

    else:
        raise ValueError(f"Unknown model_type: {model_type}. Choose 'gpt2' or 't5'.")


# =======================================================================
# == MAIN TRAINING AND EVALUATION SCRIPT ==
# This part is generic and should not be modified by students.
# =======================================================================


def main(args):
    print(f"--- Starting fine-tuning for model_type: {args.model_type} ---")

    # 1. Get model-specific components from the "student task" function
    components = get_model_components(
        args.model_type, args.lora_r, args.lora_alpha, args.lora_dropout
    )

    model_name = components["model_name"]
    model_class = components["model_class"]
    preprocess_function = components["preprocess_function"]
    lora_config = components["lora_config"]
    DataCollatorClass = components["data_collator_class"]

    # 2. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if args.model_type == "gpt2" and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3. Load Model
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
    )

    # Load in 8-bit for memory savings
    model = model_class.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",  # Automatically distributes model across GPU/CPU
    )

    # 4. Load and Prepare Dataset
    # This line will automatically download and cache the dataset
    dataset = load_dataset("samsum")
    dataset = dataset["train"].train_test_split(test_size=0.1)

    # Apply the model-specific preprocessing function
    tokenized_datasets = dataset.map(
        lambda examples: preprocess_function(examples, tokenizer),
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    # 5. Configure LoRA (PEFT)
    # Prepare model for 8-bit training
    model = prepare_model_for_kbit_training(model)
    # Apply LoRA config
    model = get_peft_model(model, lora_config)

    print("Model configured with LoRA:")
    print_trainable_parameters(model)

    # 6. Set Training Arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accumulation,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        fp16=False,  # Use mixed precision
        logging_steps=1,
        save_total_limit=2,
        report_to="none",
    )

    # 7. Define Trainer
    # We use DataCollatorForSeq2Seq for both models as it correctly
    # handles pre-computed labels for Causal LM fine-tuning.
    if args.model_type == "gpt2":
        data_collator = DataCollatorClass(tokenizer, model=model, padding="longest")
    else:  # t5
        data_collator = DataCollatorClass(tokenizer, model=model, padding="longest")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        data_collator=data_collator,
    )

    # 8. Train!
    print("Starting training...")
    trainer.train()
    print("Training finished.")

    # 9. Save the LoRA adapter
    adapter_dir = os.path.join(args.output_dir, "final_adapter")
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"Adapter saved to {adapter_dir}")

    # 10. Inference Example
    print("\n--- Running Inference ---")

    # We load the base model again (8-bit) and apply the adapter
    # This is how you would use it in production
    from peft import PeftModel

    base_model = model_class.from_pretrained(
        model_name, quantization_config=bnb_config, device_map="auto"
    )
    # Load the adapter
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.eval()  # Set to evaluation mode

    test_sample = dataset["test"][0]

    # Create the correct prompt format for inference
    if args.model_type == "gpt2":
        prompt = create_prompt_gpt2({"dialogue": test_sample["dialogue"]})
    else:  # t5
        prompt = "summarize: " + test_sample["dialogue"]

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Generate
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        eos_token_id=tokenizer.eos_token_id if args.model_type == "gpt2" else None,
    )

    print(f"**Prompt:**\n{prompt}")

    if args.model_type == "gpt2":
        # For GPT-2, we slice off the prompt
        input_token_count = inputs["input_ids"].shape[1]
        generated_summary = tokenizer.decode(
            outputs[0][input_token_count:], skip_special_tokens=True
        )
        print(f"\n**Generated Summary:**\n{generated_summary}")
    else:
        # For T5, the output is only the summary
        decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\n**Generated Summary:**\n{decoded_output}")

    print(f"\n**Ground Truth Summary:**\n{test_sample['summary']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune a model (GPT-2 or T5) with LoRA."
    )

    # Key argument
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=["gpt2", "t5"],
        help="Type of model to fine-tune ('gpt2' or 't5').",
    )

    # Training parameters
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./lora_finetune",
        help="Directory to save the trained adapter and logs.",
    )
    parser.add_argument(
        "--epochs", type=int, default=1, help="Number of training epochs."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Per-device train batch size (adjust for VRAM).",
    )
    parser.add_argument(
        "--grad_accumulation",
        type=int,
        default=4,
        help="Gradient accumulation steps (effective batch size = batch_size * grad_accumulation).",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=2e-4, help="Learning rate."
    )

    # LoRA parameters
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank (r).")
    parser.add_argument(
        "--lora_alpha", type=int, default=16, help="LoRA alpha (scaling factor)."
    )
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout.")

    args = parser.parse_args()
    main(args)
