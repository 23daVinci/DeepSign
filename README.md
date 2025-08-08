# 🖋️ DeepSign

This project implements a **production-ready pipeline** to train a **Siamese Neural Network** for **offline signature verification** using the [IDCAR signature dataset](https://www.researchgate.net/publication/335391564_IDCAR_A_New_Indian_Dataset_for_Offline_Signature_Verification).

---

## 🚀 Features

- 🔁 Pair generation (genuine-genuine, genuine-forged)
- 🧼 Image preprocessing (grayscale, resize, normalize)
- 💾 Efficient data serialization using **TFRecord**
- 🧠 Siamese Network with custom distance-based similarity learning
- 📊 Evaluation using ROC, EER, FAR/FRR
- ⚙️ Production-quality training pipeline using `tf.data` and `Keras`
- 🛠️ Inference-ready API skeleton using FastAPI *(optional)*

---

## 📦 Installation

```bash
git clone https://github.com/your-username/signature-verification-idcar.git
cd signature-verification-idcar
pip install -r requirements.txt
