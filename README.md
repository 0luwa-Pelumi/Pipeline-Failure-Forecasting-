

## 🎯 **Project Goal**
Build machine learning models to detect and classify pipeline faults using SCADA (Supervisory Control and Data Acquisition) data.

## 📊 **Dataset Structure**
- **Source**: `scada_pipeline.csv` with 1000 records
- **Target**: Binary classification (0 = Normal, 1 = Faulty)
- **Features**: 8 operational parameters including pressure, flow rate, temperature, valve status, pump state, pump speed, compressor state, and energy consumption

## 🏗️ **Two-Stage Modeling Approach**

### **Model 1: Fault Detection (Binary Classification)**
- **Goal**: Predict if pipeline is faulty (1) or normal (0)
- **Algorithm**: RandomForestClassifier
- **Performance**: 
  - Accuracy: 97.5%
  - ROC AUC: 96.4%
  - Excellent precision/recall for both classes

### **Model 2: Fault Type Classification (Multi-class)**
- **Goal**: Classify the type of fault (only on faulty data)
- **Classes**: blockage (0), degradation (1), leak (2), surge (3)
- **Algorithms Tested**:
  - RandomForest: 95.2% accuracy
  - GradientBoosting: 95.2% accuracy
- **Best Model**: RandomForest with tuned hyperparameters

## 🔧 **Key Technical Details**

### **Data Exploration Insights**
- **Class Distribution**: 694 normal vs 306 faulty cases
- **Fault Types**: degradation (135), leak (65), surge (61), blockage (45)
- **Alarm System**: 216 alarms triggered for faults, but 90 faults had no alarms (important finding!)

### **Feature Engineering**
- Used LabelEncoder for fault type classification
- Feature correlation analysis with target variable
- Stratified train-test splits to maintain class distribution

### **Model Optimization**
- Used GridSearchCV for hyperparameter tuning
- Class weight balancing for imbalanced data
- Cross-validation for robust performance evaluation

## 💡 **Key Findings**
1. **Pressure and energy consumption** are strongly correlated with faults
2. **Flow rate** decreases during faults (negative correlation)
3. Models achieve **high accuracy (>95%)** in both detection and classification tasks

## 📁 **Output Models**
- `m1_fault_detection.plk` - Binary fault detection
- `m2_fault_type.plk` - Fault type classification (RandomForest)
- `m3_fault_type.pkl` - Fault type classification (GradientBoosting)
