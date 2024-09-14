# Fusion3D

[![Join our Discord](https://img.shields.io/badge/Discord-Join%20our%20server-5865F2?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/agora-999382051935506503) [![Subscribe on YouTube](https://img.shields.io/badge/YouTube-Subscribe-red?style=for-the-badge&logo=youtube&logoColor=white)](https://www.youtube.com/@kyegomez3242) [![Connect on LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/kye-g-38759a207/) [![Follow on X.com](https://img.shields.io/badge/X.com-Follow-1DA1F2?style=for-the-badge&logo=x&logoColor=white)](https://x.com/kyegomezb)
g
An implementation of [3D Diffusion Models](https://arxiv.org/abs/2303.01469) in PyTorch.

## Installation

```bash
pip install fusion-threed
```
## Usage

```python
from fusion_threed import Diffusion3DGenerator

model = Diffusion3DGenerator(image_size=64, voxel_size=32, num_timesteps=1000)
```