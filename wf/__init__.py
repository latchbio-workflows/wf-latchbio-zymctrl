from typing import List, Optional

from latch.resources.launch_plan import LaunchPlan
from latch.resources.workflow import workflow
from latch.types.directory import LatchOutputDir
from latch.types.file import LatchFile
from latch.types.metadata import (
    LatchAuthor,
    LatchMetadata,
    LatchParameter,
    LatchRule,
    Params,
    Section,
    Spoiler,
    Text,
)

from wf.task import zymctrl_task

flow = [
    Section(
        "Generation Input",
        Text("Basic inputs required for sequence generation"),
        Params(
            "ec_numbers",
            "num_batches",
            "num_return_sequences",
        ),
    ),
    Section(
        "Fine-tuning",
        Params("training_fasta"),
        Text("""
Fine-tuning allows updating ZymCTRL's weights with new sequences.

While this is not strictly necessary (good generations even for EC classes with only 1-2 representatives in Nature), you might want to fine-tune if you have:

- Internal datasets after protein engineering efforts
- Ancestrally-reconstructed sets
- Sequences from metagenomics databases

The authors recommend using at least 200 sequences for best results, but it can work with fewer.

NOTE: Fine-tuning works with only a single EC number at a time.
            """),
        Spoiler(
            "Advanced Options",
            Params(
                "epochs",
                "validation_split",
            ),
        ),
    ),
    Section(
        "Output",
        Params("run_name"),
        Text("Directory for outputs"),
        Params("output_directory"),
    ),
]

metadata = LatchMetadata(
    display_name="ZymCTRL",
    author=LatchAuthor(
        name="Noelia Ferruz et. al.",
    ),
    repository="https://github.com/latchbio-workflows/wf-latchbio-zymctrl",
    license="Apache-2.0",
    parameters={
        "run_name": LatchParameter(
            display_name="Run Name",
            description="Name of run",
            batch_table_column=True,
            rules=[
                LatchRule(
                    regex=r"^[a-zA-Z0-9_-]+$",
                    message="Run name must contain only letters, digits, underscores, and dashes. No spaces are allowed.",
                )
            ],
        ),
        "ec_numbers": LatchParameter(
            display_name="Enzyme Class (EC) Numbers",
            description="List of EC numbers to generate sequences for (e.g. 1.1.1.1)",
            batch_table_column=True,
        ),
        "num_batches": LatchParameter(
            display_name="Number of Batches",
            description="Number of batches of sequences to generate per EC number",
            batch_table_column=True,
        ),
        "num_return_sequences": LatchParameter(
            display_name="Number of Sequences per Batch",
            description="Number of sequences per batch to generate per EC number",
            batch_table_column=True,
        ),
        "validation_split": LatchParameter(
            display_name="Validation Split",
            description="Split between training and evaluation",
            batch_table_column=True,
        ),
        "training_fasta": LatchParameter(
            display_name="Training FASTA",
            description="FASTA file containing sequences for fine-tuning (only needed for fine-tune mode)",
            batch_table_column=True,
        ),
        "epochs": LatchParameter(
            display_name="Training Epochs",
            description="Number of epochs for fine-tuning (only used in fine-tune mode)",
            batch_table_column=True,
        ),
        "output_directory": LatchParameter(
            display_name="Output Directory",
            description="Directory to write output files",
        ),
    },
    flow=flow,
)


@workflow(metadata)
def zymctrl_workflow(
    run_name: str,
    ec_numbers: List[str],
    training_fasta: Optional[LatchFile] = None,
    epochs: int = 28,
    num_batches: int = 20,
    num_return_sequences: int = 20,
    validation_split: int = 10,
    output_directory: LatchOutputDir = LatchOutputDir("latch:///ZymCTRL"),
) -> LatchOutputDir:
    """
    ZymCTRL - Enzyme Sequence Generator with fine-tuning

    <p align="center">
        <img src=https://neurips.cc/media/PosterPDFs/NeurIPS%202022/59047.png?t=1669864213.082831" alt="ZymCTRL figure" width="800px"/>
    </p>

    <html>
    <p align="center">
    <img src="https://user-images.githubusercontent.com/31255434/182289305-4cc620e3-86ae-480f-9b61-6ca83283caa5.jpg" alt="Latch Verified" width="100">
    </p>

    <p align="center">
    <strong>
    Latch Verified
    </strong>
    </p>

    ## ZymCTRL

    ZymCTRL is a conditional language model designed for sustainable, large-scale enzyme engineering. Leveraging breakthroughs in language models for protein design, it generates custom-tailored enzymes that can accelerate chemical transformations by several orders of magnitude.

    Trained on over 37M UniProt sequences with EC annotations, it creates novel enzyme sequences that are structurally viable yet distinct from natural variants, while maintaining their intended catalytic functionality.

    ### Overview

    This workflow provides:
    - Generation of artificial enzymes from any EC number prompt
    - Optional fine-tuning on custom sequence sets
    - Production of biodegradable nanoscopic catalysts for industrial applications
    - Validation of functionality through orthogonal prediction methods

    ## Model Details

    - Architecture: CTRL Transformer (similar to ChatGPT)
    - Size: 738 million parameters
    - Training: 37M UniProt enzyme sequences
    - Input: EC numbers for targeted generation
    - Output: Novel protein sequences with specified catalytic functions

    ## Finding EC Numbers

    Not sure which EC number to use? Visit the BRENDA EC Explorer to find the right classification for your target enzyme.

    ## Citations

    If you use ZymCTRL in your research, please cite:

    Conditional language models enable the efficient design of proficient enzymes
    Geraldene Munsamy, Ramiro Illanes-Vicioso, Silvia Funcillo, Ioanna T. Nakou, Sebastian Lindner, Gavin Ayres, Lesley S. Sheehan, Steven Moss, Ulrich Eckhard, Philipp Lorenz, Noelia Ferruz
    bioRxiv 2024.05.03.592223; doi: https://doi.org/10.1101/2024.05.03.592223

    """
    return zymctrl_task(
        run_name=run_name,
        ec_numbers=ec_numbers,
        training_fasta=training_fasta,
        epochs=epochs,
        num_batches=num_batches,
        num_return_sequences=num_return_sequences,
        validation_split=validation_split,
        output_directory=output_directory,
    )


LaunchPlan(
    zymctrl_workflow,
    "Sequence Generation",
    {
        "run_name": "Sequence_Generation",
        "ec_numbers": ["2.7.1.1"],
    },
)

LaunchPlan(
    zymctrl_workflow,
    "Finetuning Generation",
    {
        "run_name": "Finetuning_Sequence_Generation",
        "ec_numbers": ["2.7.1.1"],
        "training_fasta": LatchFile(
            "s3://latch-public/proteinengineering/zymctrl/training.fasta"
        ),
    },
)
