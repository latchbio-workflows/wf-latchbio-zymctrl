from typing import List, Optional

from latch.resources.workflow import workflow
from latch.types.directory import LatchOutputDir
from latch.types.file import LatchFile
from latch.types.metadata import LatchAuthor, LatchMetadata, LatchParameter, LatchRule

from wf.task import zymctrl_task

metadata = LatchMetadata(
    display_name="ZymCTRL",
    author=LatchAuthor(
        name="LatchBio",
    ),
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
        "mode": LatchParameter(
            display_name="Mode",
            description="Choose whether to generate sequences directly or fine-tune then generate",
            type=str,
            choices=["generate", "finetune"],
        ),
        "ec_numbers": LatchParameter(
            display_name="EC Numbers",
            description="List of EC numbers to generate sequences for (e.g. 1.1.1.1)",
            batch_table_column=True,
        ),
        "num_sequences": LatchParameter(
            display_name="Number of Sequences",
            description="Number of sequences to generate per EC number",
            batch_table_column=True,
        ),
        "training_fasta": LatchParameter(
            display_name="Training FASTA",
            description="FASTA file containing sequences for fine-tuning (only needed for fine-tune mode)",
            optional=True,
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
)


@workflow(metadata)
def zymctrl_workflow(
    run_name: str,
    mode: str,
    ec_numbers: List[str],
    num_sequences: int = 100,
    training_fasta: Optional[LatchFile] = None,
    epochs: int = 28,
    output_directory: LatchOutputDir = LatchOutputDir("latch:///ZymCTRL"),
) -> LatchOutputDir:
    return zymctrl_task(
        run_name=run_name,
        mode=mode,
        ec_numbers=ec_numbers,
        num_sequences=num_sequences,
        training_fasta=training_fasta,
        epochs=epochs,
        output_directory=output_directory,
    )
