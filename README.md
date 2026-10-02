# ViT Paper Replication

Implementing the Vision Transformer from scratch in PyTorch, following [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929) (Dosovitskiy et al., 2020), then comparing it against a pretrained ViT-B/16 on the same small dataset.

The architecture is built layer by layer from the paper's four equations rather than imported from `torchvision`. I did this to find out what replicating a research paper actually involves, and most of what I learned is in the implementation notes further down. The result itself is the comparison at the end the same architecture trained from scratch on 450 images against the same architecture with ImageNet-21k pretraining behind it.

## Results

3 classes (apple pie, french fries, ice cream) 450 training images, 150 test images and 10 epochs each. Each row is the epoch with the lowest test loss, with both metrics taken from that same epoch.

| Setup | Trainable params | Test acc | Test loss | Best epoch |
|---|---|---|---|---|
| From scratch, lr 3e-5 | 85.8M | 0.570 | 0.970 | 6 of 10 |
| From scratch, lr 1e-3 | 85.8M | 0.363 | 1.095 | 8 of 10 |
| Pretrained ViT-B/16, head only | 2,307 | **0.954** | **0.111** | 4 of 10 |

Chance level for 3 balanced classes is 0.333 accuracy and ln(3) = 1.0986 cross-entropy.

The from-scratch model learns, but barely. 0.570 against a 0.333 floor is real, and test loss stays below chance in every epoch, but it is nowhere near usable. 85.8M parameters on 450 images works out to roughly 190,000 parameters per training sample and ViT has no convolutional inductive bias to make up the difference.

The 1e-3 run does not learn at all. Its mean test accuracy across the 10 epochs is 0.329, slightly below chance and its mean test loss is 1.136, above ln(3). The paper's 1e-3 is a base rate for batch size 4096. At batch size 32 the linearly scaled equivalent is around 7.8e-6, so 1e-3 is about 128 times too large and there is no warmup or gradient clipping here to absorb that. I kept the run in because it shows what a too-high learning rate looks like when the model does not blow up outright it just never descends.

All three runs peaked before epoch 10, so training longer was making things worse rather than better.

## Architecture

Built from the paper's Table 1 spec for ViT-Base:

| | |
|---|---|
| Layers | 12 |
| Hidden size D | 768 |
| MLP size | 3072 |
| Heads | 12 |
| Patch size | 16x16 |
| Params | 85.8M (for 3 classes) |

Each equation is its own module:

* **Eq 1** `PatchEmbedding` turns a 224x224 image into 196 patches via a strided Conv2d, flattens them to a (196, 768) sequence, prepends a learnable class token and adds learnable 1D position embeddings
* **Eq 2** `MultiHeadSelfAttentionBlock` is LayerNorm then `nn.MultiheadAttention` then dropout, with a residual connection
* **Eq 3** `MLPBlock` is LayerNorm then Linear to 3072, GELU, dropout, Linear back to 768, dropout, with a residual connection
* **Eq 4** the classifier is LayerNorm and a Linear layer on the class token output

As a check, one `TransformerEncoderBlock` has 7,087,872 parameters, matching `nn.TransformerEncoderLayer(d_model=768, nhead=12, dim_feedforward=3072, norm_first=True)` exactly.

## Implementation notes

The paper leaves most of the training setup to its appendix and reference code. These are the details that changed the result, checked against [google-research/vision_transformer](https://github.com/google-research/vision_transformer) and [google-research/big_vision](https://github.com/google-research/big_vision). My first version of this notebook sat at chance for ten epochs because several of them were wrong.

**Decoupled weight decay.** The paper says "Adam with a high weight decay of 0.1", but `big_vision/optax.py` applies it with `optax.add_decayed_weights`, independent of the gradient, and explicitly refuses coupled decay. The PyTorch equivalent is `AdamW`, not `torch.optim.Adam(weight_decay=...)`, which adds an L2 term to the gradient and behaves very differently under Adam's adaptive scaling.

**Decay only on weight matrices.** `wd_mults` defaults to `[(".*/kernel$", 1.0)]`, so biases, LayerNorm parameters, the class token and the position embeddings are never decayed. Passing every parameter to the optimizer shrinks the embeddings toward zero during training. It is measurable: with decay on everything, the position embedding standard deviation fell from 1.0 to 0.887 over 10 epochs. With the parameter groups set correctly it stays at its initialized value of 0.02.

**Embedding initialization.** `models_vit.py` uses `nn.initializers.normal(stddev=0.02)` for position embeddings, commented "from BERT", and `nn.initializers.zeros` for the class token. Initializing position embeddings with unit-variance `torch.randn` swamps the patch projection output, which is on the order of 0.1 to 0.5, so the image content is drowned at the input to the first encoder block.

**Zero-initialized head.** `kernel_init=nn.initializers.zeros` on the final Dense layer makes the initial logits identical, so training starts at exactly ln(num_classes). Without it my epoch 1 training loss was 1.226, above chance, because the model had to unlearn a randomly confident head before it could start.

**Learning rate scales with batch size.** `big_vision/configs/vit_i21k.py` pairs `lr = 0.001` with `batch_size = 4096`, `grad_clip_norm = 1.0` and `warmup_steps = 10_000`. Dropping the batch size without scaling the learning rate, and without the warmup and clipping, produces the 1e-3 result above.

**Dropout after attention.** `Encoder1DBlock` applies dropout to the attention output before the residual add, in addition to the dropout inside the attention itself.

## Setup

```bash
git clone https://github.com/Rohan08-21/ViT-Paper-Replication.git
cd ViT-Paper-Replication

python -m venv env
env\Scripts\activate          # Windows
source env/bin/activate       # macOS or Linux

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt

jupyter lab
```

## Environment

| | |
|---|---|
| Python | 3.14.7 |
| torch | 2.14.0+cu130 |
| torchvision | 0.29.0+cu130 |
| CUDA | 13.0 |
| GPU | NVIDIA RTX 4080 Laptop, 12 GB |

If you are on a different CUDA version, get the right install command from [pytorch.org](https://pytorch.org/get-started/locally/). For CPU only, drop the `--index-url` flag. The notebook is device-agnostic but training the from-scratch model on CPU is impractical.

## Data

A 20% subset of three [Food101](https://pytorch.org/vision/stable/generated/torchvision.datasets.Food101.html) classes: apple pie, french fries, ice cream. 450 training images and 150 test images, so 150 and 50 per class.

Not committed. Build it yourself with `torchvision.datasets.Food101` and save it in the standard `ImageFolder` layout, `train/<class>/` and `test/<class>/` under a folder in `data/`, then point the notebook at it. Any three classes will run, but keep these three and these split sizes if you want the results above to be comparable.

## Structure

```
ViT-Paper-Replication/
├── 07_PyTorch_VisionTransformerPaper.ipynb
├── going_modular/
│   ├── data_setup.py
│   └── engine.py
├── helper_functions.py
├── images/                   # paper figures used in the notebook
├── requirements.txt
└── README.md
```

`data/`, `models/` and `env/` are gitignored. No weights are committed. The pretrained run trains a `Linear(768, 3)` head and takes about a minute, and the from-scratch model is 328 MB and only reaches 0.570, so neither is worth shipping.

## What I took from this

The architecture was the easy part. Building the twelve encoder blocks from the four equations was mechanical once the shapes lined up, and the parameter count matched `nn.TransformerEncoderLayer` exactly. Everything that actually went wrong was in the training setup, and almost none of it is in the main text of the paper. It is in the appendix and in the authors' code.

The thing I did not expect is how much a model that has learned nothing resembles a model that is training badly. My first version sat between 0.29 and 0.50 test accuracy across ten epochs and the loss curves had enough movement in them to look like something was happening. Comparing the loss against ln(3) = 1.0986 instead of reading the plot is what showed it had never once beaten chance.

The other lesson is that a hyperparameter does not travel without the batch size it was tuned at. 1e-3 is the number printed in the paper. At batch 4096 it is correct at batch 32 it is roughly 128 times too large and nothing in the paper warns you because the paper never ran at batch 32.

## Limitations

Single seed per configuration. The 3e-5 and 1e-3 runs share an initialization and a data order, so the learning rate is the only variable between them, but neither is averaged over seeds and the epoch-to-epoch variance on a 150 image test set is substantial.

No warmup or gradient clipping. Both belong inside the training loop in `going_modular/engine.py` and were not added.

Accuracy is reported as an unweighted mean of per-batch accuracies, which is slightly biased when the final batch is short.

## References

* Dosovitskiy et al., [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929), ICLR 2021
* [google-research/vision_transformer](https://github.com/google-research/vision_transformer), the authors' reference implementation
* [google-research/big_vision](https://github.com/google-research/big_vision), the pretraining configs

