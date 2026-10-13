# CS3807 Experiment 8 & 9: Attention in ResNet50

Add attention modules to a pretrained ResNet50 and compare different types (channel, spatial, self-attention) at different positions (early, middle, late stages). Test on flower images.


### Setup
- Use **Google Colab** (free GPU, ~45 min to train all models).
- Open `resnet50_attention_lab_FULL.ipynb` and run cells top to bottom.
- Or use Jupyter locally if you have a GPU.

### Data
The notebook auto-downloads 3,670 flower images (5 types), splits them 70% train / 15% val / 15% test, and resizes to 224×224.

## What is covered

**Attention mechanisms:** Channel attention (what channels matter), spatial attention (what locations matter), self-attention (how distant parts relate).

**Experiments:**
1. Compare five attention types at the same position (middle of network).
2. Insert the same attention at different depths (early, middle, late) and see how effectiveness changes.
3. Use attention at multiple positions and check if it helps.
4. Apply attention layers in different orders.
5. Try different kernel sizes.

**Outputs:** Accuracy, inference speed, visualizations of what the network attends to (Grad-CAM maps and internal attention heatmaps).

## What the Notebook Does

1. **Setup & imports** – Install TensorFlow, Keras, scikit-learn, pandas, matplotlib. Set training parameters (batch size=32, epochs=5+5, learning rates).
2. **Download data** – Get 3,670 flower images, split 70/15/15, preprocess to 224×224.
3. **Define attention layers** – Channel attention, spatial attention, self-attention, ECA, etc.
4. **Build ResNet50 with attention** – Take pretrained ResNet50, insert attention after one or more of its stages (P1, P2, P3, P4).
5. **Train each model** – Freeze backbone first, then fine-tune. Save results (accuracy, F1, parameters, speed) to disk so you can resume.
6. **Run experiments** – Train 15 models total (baseline + various attention configs).
7. **Make tables** – Show results by experiment type (which attention? which position? single or multiple positions?).
8. **Plot graphs** – Accuracy curves, confusion matrices, accuracy vs parameters, speed vs F1.
9. **Visualize attention** – Show what each model pays attention to (heatmaps) and how gradients explain its predictions (Grad-CAM).

## Settings that can be changed

- **`SEEDS = [42]`** – Use one seed for speed, or `[42, 7, 123]` to get mean ± std (takes 3× longer).
- **`BS = 32`** – Batch size. Reduce to 16 or 8 if out of memory.
- **`EP_HEAD, EP_FT = 5, 5`** – Epochs to train (frozen backbone, then fine-tune). Increase to 10 or 20 for your final report.
- **`RUN_P1 = True`** – Include P1 (early stage, high memory). Set `False` if it crashes.
- **`QUICK = False`** – For testing, set to `True` to use tiny data.



**Typical findings:**
- Channel attention is cheap and works well at deep stages (P3, P4).
- Spatial attention helps suppress background.
- Self-attention is powerful but slow.
- Putting attention deeper (later stages, P3/P4) usually works better than early (P1/P2).
- Adding attention at multiple positions helps, but not always enough to justify the extra cost.

## Common Issues

**Out of memory (OOM):**
- Reduce batch size: `BS = 16` or `BS = 8`.
- Skip P1: `RUN_P1 = False`.
- Use fewer epochs: `EP_HEAD, EP_FT = 3, 3`.

**Protobuf error when downloading data:**
```python
!pip install -q -U "protobuf>=6.31.1"
```
Then restart the Colab runtime.

**Data download fails:**
```python
!rm -rf /root/.keras/datasets/flower_photos*
```
Then re-run the data cell.

**Model results missing?**
Results are saved to `results/` folder. If training crashed, just re-run the training cell to complete it.

---

