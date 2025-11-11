# AI Photo Editor

## Project Overview

This project demonstrated how <b>Generative AI</b> and <b>Computer Vision</b> can be combined to perform interactive image editing - specifically replacing or modifying a subject or background using natural language prompts.

The app allows the users to:
-   Upload an image.
-   Select a subject by clicking on it (using <b>Meta's SAM</b> for segmentation).
-   Generate an accurate segmentation mask around the subject.
-   Replace either:
    -   the background while keeping the subject intact or
    -   the subject while keeping the background intact.
-   Describe the new content using natural language prompts (e.g. "replace background with a snowy mountain" or "turn the cat into a crocodile")

## Models Used
1. `Segment Anything Model` for the object detection and segmentation via clicks.
2. `Stable Diffusion Model` to generate new image or modifying the existing image given the region and text prompts.
