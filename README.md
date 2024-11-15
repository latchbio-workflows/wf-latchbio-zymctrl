# ZymCTRL - Enzyme Sequence Generator with fine-tuning

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

### Model Details

- Architecture: CTRL Transformer (similar to ChatGPT)
- Size: 738 million parameters
- Training: 37M UniProt enzyme sequences
- Input: EC numbers for targeted generation
- Output: Novel protein sequences with specified catalytic functions

### Finding EC Numbers

Not sure which EC number to use? Visit the BRENDA EC Explorer to find the right classification for your target enzyme.

### Citations

If you use ZymCTRL in your research, please cite:

Conditional language models enable the efficient design of proficient enzymes
Geraldene Munsamy, Ramiro Illanes-Vicioso, Silvia Funcillo, Ioanna T. Nakou, Sebastian Lindner, Gavin Ayres, Lesley S. Sheehan, Steven Moss, Ulrich Eckhard, Philipp Lorenz, Noelia Ferruz
bioRxiv 2024.05.03.592223; doi: https://doi.org/10.1101/2024.05.03.592223
