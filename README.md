# diffae-fwd

This is a inference-only version of the original diffae repository: <https://github.com/konpatp/diffae>. \
We use up-to-date versions of Python and Pytorch and try to keep other dependencies as minimal as possible.

## Setup

We recommend [miniforge](https://conda-forge.org/download/) to set up your python environment. \
Then [uv](https://docs.astral.sh/uv/) can be used to install the project dependencies:

```bash
conda create -n $YOUR_ENV_NAME python=3.12
conda activate $YOUR_ENV_NAME
uv pip install -r requirements.txt
```
