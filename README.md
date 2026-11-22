# 🥗 NutriVision AI — Food Image Analyzer & Calorie Counter

> **Advanced AI-Powered Nutrition & Calorie Estimation Platform**  
> Real-data trained MobileNetV2 · OpenCV multi-food segmentation · 504+ verified nutrition entries · Streamlit web app

---

## 📋 Table of Contents
1. [Executive Summary & Workflow](#-executive-summary--workflow)
2. [Key Features & Improvements](#-key-features--improvements)
3. [Deep Learning Architecture](#-deep-learning-architecture)
4. [Smart Nutrition Database (504+ Foods)](#-smart-nutrition-database-504-foods)
5. [OpenCV Multi-Food Region Segmentation](#-opencv-multi-food-region-segmentation)
6. [Daily Calorie & Diet Calculator](#-daily-calorie--diet-calculator)
7. [Running the Application](#-running-the-application)
8. [Model Evaluation & Real Metrics](#-model-evaluation--real-metrics)
9. [Viva Examination Q&A](#-viva-examination-qa)

---

## 🌟 Executive Summary & Workflow

NutriVision AI revolves around the primary real-world clinical nutrition workflow:

$$\text{User Upload / Camera} \longrightarrow \text{OpenCV Segmentation} \longrightarrow \text{Deep Learning Classification} \longrightarrow \text{Portion Estimation} \longrightarrow \text{Nutrition Calculation} \longrightarrow \text{Daily Food Diary}$$

```
📸 User Uploads / Snaps Food Image
     │
     ├──> Preprocessing (LANCZOS resize to 224x224, float32 normalization)
     │
     ├──> OpenCV Multi-Food Contour Segmentation (detects multiple items on plate)
     │
     ├──> MobileNetV2 (ImageNet Transfer Learning) & Custom 4-Block CNN
     │      └── Softmax Probabilities + Top-3 Confidences
     │
     ├──> Serving Weight Calibration (foreground pixel area + verified database serving)
     │
     ├──> Nutrition Calculation Engine:
     │      • Calories = (Weight / 100) × Cal/100g
     │      • Protein, Carbs, Fat, Fiber, Sugar, Sodium
     │
     ├──> Instant Interactive Recalculation on Portion Slider Adjust
     │
     └──> Persistent SQLite Daily Diary (grouped by Breakfast, Lunch, Dinner, Snack)
```

---

## 🚀 Key Features & Improvements

- **Minimal native Streamlit UI**: Calm typography, native metrics, progress bars, and expanders — no dashboard clutter.
- **504+ Verified Food Database**: Sourced from USDA FoodData Central and Indian Food Composition Tables (IFCT) covering Indian main dishes, South Indian specialties, street food, Western meals, fruits, vegetables, beverages, and desserts.
- **Multi-Food Region Detection**: OpenCV-based plate contour analysis that isolates distinct items (e.g. Dosa + Sambar + Chutney or Burger + Fries), estimating individual portions and calculating grand meal totals.
- **Non-Food Rejection Gate**: The classifier compares each photo against the training feature bank (mean top-5 cosine similarity ≥ 0.57). Faces, landscapes and objects are flagged as “not one of the 12 supported foods” instead of receiving invented calories.
- **Micro & Macro Nutritional Tracking**: Tracks Calories, Protein, Carbohydrates, Fat, Dietary Fiber, Sugars, and Sodium per item and per meal.
- **Daily Budget & RDA Gauges**: Daily budget tracker (`Daily Goal - Food Intake = Remaining Calories`) with progress bars against RDA targets.
- **Persistent SQLite Food Diary**: Stores logged meals with timestamps, meal types, itemized breakdowns, and deletion capabilities.

---

## 🧠 Deep Learning Architecture

### 1. MobileNetV2 Transfer Learning (`training/train_mobilenet_linear.py`)
- **Backbone**: MobileNetV2 pre-trained on 1.28M ImageNet images (frozen).
- **Production classifier** (best result on this dataset size):
  $$\text{Input}(224,224,3) \rightarrow \text{MobileNetV2} \rightarrow \text{GAP}(1280) \rightarrow \text{StandardScaler} \rightarrow \text{LogisticRegression}(12) \rightarrow \text{Softmax}$$
  The scaler and logistic-regression weights are folded into a Keras `Dense`
  layer, so the deployed artefact is one self-contained `.keras` model.
- An optional fine-tuned-head variant is available in
  `models/transfer_learning.py`, but it scored lower on the small real dataset.

### 2. Custom 4-Block Convolutional Neural Network (`models/cnn_model.py`)
- **Architecture**:
  - In-model data augmentation (`RandomFlip`, `RandomRotation(±20°)`, `RandomZoom(±20%)`)
  - 4 Sequential Conv blocks: `Conv2D(32, 64, 128, 256)` with `BatchNormalization`, `ReLU`, and `MaxPooling2D(2×2)`
  - Fully Connected layers: `Dense(512, Dropout=0.5)` $\rightarrow$ `Dense(256, Dropout=0.3)` $\rightarrow$ `Dense(12, Softmax)`
  - Total parameters: 26,215,372
  - CPU training is slow (several minutes per epoch on Apple Silicon); it is
    optional. The app automatically falls back to MobileNetV2 when the
    checkpoint is absent.

### 3. Tabular Nutritional Regression ANN (`models/ann_model.py`)
- Demonstrates tabular regression on structured data independently of image inputs:
  $$\text{Input}(13\text{ features: 12-dim one-hot class} + \text{normalised weight}) \rightarrow \text{Dense}(64) \rightarrow \text{Dense}(128) \rightarrow \text{Dense}(64) \rightarrow \text{Dense}(4, \text{Linear})$$
- Achieved Test MAE = 9.06 kcal.

---

## 📊 Smart Nutrition Database (504+ Foods)

Stored locally in `data/nutrition_database.json` and `data/nutrition.csv` for 100% offline functionality.

| Category | Representative Food Items |
|---|---|
| **Indian Main Course** (60) | Butter Chicken, Chicken Tikka Masala, Rogan Josh, Shahi Paneer, Palak Paneer, Dal Makhani, Dal Tadka, Rajma, Chole |
| **Indian Breads & Rice** (50) | Roti, Naan, Garlic Naan, Aloo Paratha, Paneer Paratha, Hyderabadi Chicken Biryani, Mutton Biryani, Veg Pulao, Khichdi |
| **South Indian** (40) | Masala Dosa, Rava Dosa, Steamed Idli, Medu Vada, Uttapam, Sambar, Coconut Chutney, Appam, Ven Pongal, Poha |
| **Street Food & Snacks** (45) | Samosa, Pav Bhaji, Chole Bhature, Pani Puri, Bhel Puri, Sev Puri, Aloo Tikki, Paneer Pakora, Vada Pav, Momos |
| **Indian Sweets** (40) | Gulab Jamun, Rasgulla, Rasmalai, Jalebi, Kheer, Gajar Halwa, Moong Dal Halwa, Kaju Katli, Besan Ladoo, Shrikhand |
| **Western & Fast Food** (55) | Margherita Pizza, Pepperoni Pizza, Cheeseburger, French Fries, Club Sandwich, Pasta Alfredo, Lasagna, Grilled Chicken |
| **Asian Cuisine** (35) | Chicken Fried Rice, Hakka Noodles, Pad Thai, California Roll, Salmon Nigiri, Ramen, Dim Sum, Tom Yum, Chilli Paneer |
| **Fruits** (45) | Apple, Banana, Alphonso Mango, Orange, Grapes, Watermelon, Papaya, Pineapple, Strawberry, Pomegranate, Avocado |
| **Vegetables** (45) | Broccoli, Carrot, Cucumber, Tomato, Potato, Spinach, Cauliflower, Cabbage, Capsicum, Green Peas, Sweet Corn |
| **Beverages** (40) | Whole Milk, Soy Milk, Almond Milk, Black Coffee, Cappuccino, Masala Chai, Green Tea, Fresh Juices, Lassi, Chaas |
| **Desserts & Bakery** (40) | Ice Cream, Chocolate Cake, Brownie, Cookies, Glazed Donut, Croissant, Dark Chocolate, Almonds, Walnuts, Protein Bar |
| **Grains & Dairy** (35) | Rolled Oats, Quinoa, Boiled Chickpeas, Lentils, Tofu, Paneer, Cheddar Cheese, Greek Yoghurt, Raita, Hummus |

---

## 📈 Model Evaluation (Real Held-Out Photos)

The production MobileNetV2 linear-probe model is evaluated on **562 real
photographs** (47 per class) that were never used for training or validation:

| Metric | Score |
|---|---|
| Top-1 accuracy | **91.1%** |
| Top-3 accuracy | **98.6%** |
| Macro precision | 0.913 |
| Macro recall | 0.911 |
| Macro F1 | 0.911 |

Data split: **2,353 train / 470 validation / 562 test** images across 12 classes.
The official dataset test splits are used (Food-101 `validation`, and the
official test splits of the Indian/Fruits datasets).

Full per-class report: [`results/mobilenet_classification_report.txt`](results/mobilenet_classification_report.txt)  
Confusion matrix: [`results/mobilenet_confusion_matrix.png`](results/mobilenet_confusion_matrix.png)  
Machine-readable metrics: [`results/mobilenet_metrics.json`](results/mobilenet_metrics.json)

The non-food gate uses `saved_models/ood_calibration.npz` (training feature
bank + threshold), which `training/train_mobilenet_linear.py` regenerates. If
the file is absent, the app falls back to plain 12-class classification.

> **Note on the original scores:** the first version of this repo trained on
> `make_synthetic_dataset()`, which generates random-noise images. On random
> noise, 1-in-12 chance accuracy is ~8.3% — exactly what the checked-in
> "~6%" metrics showed. Those models scored **8.0% on real photos**. The
> synthetic path is now restricted to pipeline smoke tests, and the training
> and evaluation defaults require real photographs.

---

## 💻 Running the Application

### Windows (recommended)
Double-click **`launch.bat`**, or follow the step-by-step
**[Windows setup guide](WINDOWS.md)**. The trained MobileNetV2 classifier,
non-food calibration, ANN, sample photos, and nutrition database are included
in the repository, so **you do not need the image dataset just to run the app**.
No `.env` file, API key, or external service is required.

### macOS / Linux

#### 1. Install dependencies
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows manually:
```bat
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

#### 2. Download real training data — **only for retraining** (one time, ~600 MB)
The image classifiers are trained on real photographs assembled from public
datasets (Food-101, Indian Foods, Indian Food Images, Fruits-360):
```bash
python training/download_dataset.py
```
This writes `data/food_images/{train,val,test}/<class>/` and twelve
`assets/samples/*.jpg` quick-test photos. Skip this step if you only want to
run the app.

#### 3. Train the models — **only for retraining**
```bash
# Primary model (frozen MobileNetV2 features + linear classifier)
python training/train_mobilenet_linear.py

# Optional comparison model (custom 4-block CNN; slow on CPU)
python training/train_cnn.py --model cnn --dataset real --epochs 5

# Tabular nutrition ANN
python training/train_ann.py
```

`train_mobilenet_linear.py` caches the extracted 1280-D features in
`data/features/`, so retraining the classifier takes seconds. Use
`--force-features` after changing the image dataset.

#### 4. Evaluate on the held-out test split
```bash
python training/evaluate.py --model all --dataset real
```

#### 5. Launch the app
```bash
python -m streamlit run app.py
```
The app opens at **http://localhost:8501**.

> **Why the checked-in metrics used to be ~6%:** the original `bootstrap_models.py`
> trained on `make_synthetic_dataset()`, which generates random-noise images.
> Random noise has no class signal, so 1-in-12 chance accuracy (~8.3%) is the
> expected result. The evaluation script also defaulted to the same noise data.
> Both defaults have been fixed; real photographs are now required, and
> synthetic data is only available for pipeline smoke tests.


---

## 🎓 Viva Examination Preparation Guide

### Q1: How does your system estimate food quantity from a single 2D image?
**Answer:** A standard 2D RGB image contains no metric depth scale. Our system uses a dual-stage heuristic:
1. **Contour & Foreground Ratio**: OpenCV HSV/LAB segmentation extracts the foreground food region area relative to the plate frame.
2. **Nutritional Density & Standard Calibration**: Typical serving weights from our 504-food database are scaled proportionally by the area coverage. The UI explicitly labels weight as an approximation and provides an interactive slider for user correction.

### Q2: Why use MobileNetV2 over traditional CNNs like VGG16?
**Answer:** VGG16 contains over 138 million parameters and heavy dense layers, making CPU inference slow. MobileNetV2 uses depthwise separable convolutions (which separate 3×3 spatial filtering from 1×1 pointwise channel combination), reducing computational complexity by 8–9× (only 3.4M parameters) while maintaining high feature transferability from ImageNet.

### Q3: What is the purpose of having both a CNN and an ANN?
**Answer:** They solve two distinct paradigms of deep learning:
- **CNN**: Computer Vision classification operating on spatial tensors ($224 \times 224 \times 3$) to identify food categories.
- **ANN**: Tabular multi-output regression operating on discrete attributes (one-hot category vector + portion weight) to predict continuous nutritional variables (Calories, Protein, Carbs, Fat) with an evaluation MAE of 9.06 kcal.
