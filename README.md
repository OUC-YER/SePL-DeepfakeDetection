# Generalizable Deepfake Detection via Separable Prompt Learning

 ### Authors: Enrui Yang, Baoyuan Wu, Yuezun Li*
 
 In this paper, we propose Separable Prompt Learning (SePL) to better exploit the text modality, which further enhances the detection
capacity. Specifically, SePL distills the forgery knowledge from CLIP via two separate learnable prompts, supported by a cross-modality alignment strategy and dedicated objectives.
<img width="2031" height="605" alt="pipeline1" src="https://github.com/user-attachments/assets/b4d0bdfc-92b7-40af-97ba-de501060e508" />


## Comparison with state-of-the-art deepfake detection methods on cross-dataset and cross-method evaluations.

<img width="991" height="353" alt="屏幕截图 2026-04-16 100430" src="https://github.com/user-attachments/assets/fb9f1b88-e886-46eb-8005-5278534d32c9" />

## Model Weight Download

| Model Weight | Download Link |
| :--- | :--- |
| ckpt_best.pth | [Google Drive](https://drive.google.com/file/d/1hIl36E695MDa6ALOGxp2cAb241MzYgCe/view?usp=sharing) |

## Environment

Our environment meets the following requirements:

-  **Python** 3.10.18
-  **PyTorch** 1.11.0+cu113
-  **CUDA** 11.3
-  **GPU** NVIDIA GeForce RTX 3090
---

> The dataset downloading and processing procedures can be referred to the implementation provided in [**DeepfakeBench**](https://github.com/SCLBD/DeepfakeBench).

### If you want to reproduce the results of our paper, you can follow the detailed procedure in [Effort](https://github.com/YZY-stack/Effort-AIGI-Detection).

## Training

Make sure to modify the relevant configurations in the `train.yaml` file before training.

Start training with the following command:

```bash
python3 training/train.py --detector_path ./training/config/detector/sepl.yaml --train_dataset FaceForensics++ --test_dataset Celeb-DF-v1 Celeb-DF-v2 
```

---

## Testing

Make sure to modify the relevant configurations in the `test.yaml` file before testing.

To test the model, you can directly load our pre-trained weights and run a command like the following:

```bash
python3 training/test.py --detector_path ./training/config/detector/sepl.yaml --test_dataset Celeb-DF-v1 Celeb-DF-v2 DFD DFDC DFDCP WDF --weights_path ./training/weights/ckpt_best.pth
```

---

## Citations

If this repo helps your research, please cite it as
```bash
@article{sepl2026generalizable,
  author  = {Enrui Yang, Baoyuan Wu, Yuezun Li}, 
  title   = {Generalizable Face Forgery Detection via Separable Prompt Learning},
  journal = {arXiv preprint arXiv:2604.17307},
  year    = {2026}
}
```


---

## 📄 License

This project is released for research purposes. Please refer to the repository for licensing details.
