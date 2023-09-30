#!/bin/bash
git clone https://github.com/CompVis/stable-diffusion
git clone https://github.com/CompVis/taming-transformers
git clone https://github.com/CompVis/latent-diffusion
export PYTORCH_CUDA_ALLOC_CONF="garbage_collection_threshold:0.6, max_split_size_mb:32"
