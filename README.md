# Generalizable Deepfake Detection via Separable Prompt Learning

Pipeline of SePL. TThe upper part illustrates how instance-specific prompts are generated and encoded into content and artifact text embeddings. The
lower part shows the text-guided feature decoupling process and the cross-modal alignment objective.
<img width="1714" height="1060" alt="p1" src="https://github.com/user-attachments/assets/05de163e-f3b7-4016-851d-c7a8961a4744" />

[Model Weight] Download : [Google Drive](https://drive.google.com/file/d/1hIl36E695MDa6ALOGxp2cAb241MzYgCe/view?usp=sharing)
## Environment

Our environment meets the following requirements:

-  **Python** 3.10.18
-  **PyTorch** 1.11.0+cu113
-  **CUDA** 11.3
-  **GPU** NVIDIA GeForce RTX 3090
---

> The dataset downloading and processing procedures can be referred to the implementation provided in [**DeepfakeBench**](https://github.com/SCLBD/DeepfakeBench).

---

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

## 📄 License

This project is released for research purposes. Please refer to the repository for licensing details.
