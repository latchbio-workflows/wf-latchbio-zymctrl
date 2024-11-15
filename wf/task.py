import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from latch.executions import rename_current_execution
from latch.functions.messages import message
from latch.resources.tasks import large_gpu_task
from latch.types.directory import LatchOutputDir
from latch.types.file import LatchFile

sys.stdout.reconfigure(line_buffering=True)


@large_gpu_task
def zymctrl_task(
    run_name: str,
    ec_numbers: List[str],
    training_fasta: Optional[LatchFile] = None,
    epochs: int = 28,
    num_batches: int = 20,
    num_return_sequences: int = 20,
    validation_split: int = 10,
    output_directory: LatchOutputDir = LatchOutputDir("latch:///ZymCTRL"),
) -> LatchOutputDir:
    rename_current_execution(str(run_name))

    print("-" * 60)
    print("Creating local directories")
    local_output_dir = Path(f"/root/outputs/{run_name}")
    local_output_dir.mkdir(parents=True, exist_ok=True)

    print("-" * 60)
    subprocess.run(["nvidia-smi"], check=True)
    subprocess.run(["nvcc", "--version"], check=True)

    print("-" * 60)
    try:
        if training_fasta:
            if len(ec_numbers) != 1:
                message(
                    "error",
                    {
                        "title": "ZymCTRL Error",
                        "body": "Fine-tuning mode only supports one EC number",
                    },
                )
                raise ValueError(
                    "Fine-tuning mode only supports one EC number at a time"
                )

            print(f"Starting fine-tuning workflow for EC {ec_numbers[0]}")

            dataset_dir = local_output_dir / "dataset"
            model_dir = local_output_dir / "model"
            sequences_dir = local_output_dir / "sequences"

            dataset_dir.mkdir(exist_ok=True, parents=True)
            model_dir.mkdir(exist_ok=True, parents=True)
            sequences_dir.mkdir(exist_ok=True, parents=True)

            print("Preparing training data")
            subprocess.run(
                [
                    "python3.9",
                    "/root/scripts/prep.py",
                    "--input",
                    str(training_fasta.local_path),
                    "--ec_label",
                    ec_numbers[0],
                    "--output_dir",
                    str(dataset_dir),
                    "--validation_split",
                    str(validation_split),
                ],
                check=True,
            )

            print("Fine-tuning model")
            subprocess.run(
                [
                    "python3.9",
                    "/root/scripts/run_clm-post.py",
                    "--tokenizer_name",
                    "AI4PD/ZymCTRL",
                    "--model_name_or_path",
                    "AI4PD/ZymCTRL",
                    "--do_train",
                    "--do_eval",
                    "--output_dir",
                    str(model_dir),
                    "--eval_strategy",
                    "steps",
                    "--eval_steps",
                    "10",
                    "--logging_steps",
                    "5",
                    "--save_steps",
                    "500",
                    "--num_train_epochs",
                    str(epochs),
                    "--per_device_train_batch_size",
                    "1",
                    "--per_device_eval_batch_size",
                    "4",
                    "--cache_dir",
                    str(local_output_dir / "cache"),
                    "--save_total_limit",
                    "2",
                    "--learning_rate",
                    "0.8e-04",
                    "--dataloader_drop_last",
                    "True",
                ],
                check=True,
                cwd=local_output_dir,
            )

            model_path = str(model_dir)
            output_path = str(sequences_dir)

        else:
            print("Starting direct generation workflow")
            model_path = "AI4PD/ZymCTRL"
            output_path = str(local_output_dir)

        # Generate sequences (for both modes)
        print("Generating sequences")
        for ec in ec_numbers:
            ec_dir = Path(output_path) / ec
            ec_dir.mkdir(exist_ok=True)

            print(
                f"Generating {num_batches*num_return_sequences} sequences for EC {ec}"
            )
            subprocess.run(
                [
                    "python3.9",
                    "/root/scripts/generate.py",
                    "--ec_number",
                    str(ec),
                    "--output_dir",
                    str(ec_dir),
                    "--model_path",
                    model_path,
                    "--num_batches",
                    str(num_batches),
                    "--num_return_sequences",
                    str(num_return_sequences),
                ],
                check=True,
            )

    except Exception as e:
        error_msg = f"ZymCTRL {'fine-tuning' if training_fasta else 'generation'} failed: {str(e)}"
        print(f"ERROR: {error_msg}")
        message("error", {"title": "ZymCTRL Error", "body": error_msg})
        raise e

    print("-" * 60)
    print("Returning results")
    return LatchOutputDir(str("/root/outputs"), output_directory.remote_path)
