# DeepSeek Quantization Visualization

A small Streamlit app for exploring the quantization section of the DeepSeek chapter in
`deep_seek_quant_chapter.pdf`.

The app focuses on the core FP8 training ideas:

- number formats and bit layouts
- mixed precision policy
- fine-grained quantization with local scales
- accumulation precision and Tensor Core to CUDA Core promotion
- E4M3 vs E5M2 format tradeoffs
- online quantization with current-tensor scaling

## Run

```bash
streamlit run app.py
```

If using the repo pyenv environment:

```bash
pyenv local deepseek-quantization-visualization-3.13.2
pip install -r requirements.txt
streamlit run app.py
```

## Views

- `Number Formats`: compare FP32, FP16, BF16, INT8, FP8 E4M3, and FP8 E5M2.
- `Pillar 1`: explains DeepSeek's mixed precision framework.
- `Pillar 2`: shows fixed fine-grained blocks with separate local scaling factors.
- `Pillar 3`: summarizes accumulation promotion from Tensor Cores to CUDA Cores.
- `Pillar 4`: visualizes why DeepSeek prefers E4M3 when local scaling handles range.
- `Pillar 5`: explains online quantization versus delayed historical scaling.
