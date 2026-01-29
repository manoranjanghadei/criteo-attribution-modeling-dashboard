# CRITEO ATTRIBUTION MODELING - PG-DBDA PROJECT

## Quick Start Guide

### 📋 **What This Project Does**
Predicts which online advertising impressions will generate revenue for Criteo using machine learning.

### **How to Run (3 Simple Steps)**

1. **Explore the Data**
   ```bash
   python 1_eda_analysis.py
   ```
   - Analyzes the dataset
   - Shows attribution patterns
   - Creates visualizations

2. **Prepare Features**
   ```bash
   python 2_feature_engineering_production_ready.py
   ```
   - Creates meaningful features
   - Prepares data for modeling
   - Saves processed dataset

3. **Train Models**
   ```bash
   python criteo_final_4models.py
   ```
   - Trains 4 industry-standard models
   - Compares performance
   - Selects best model

### **Results Summary**
- **Best Model**: LightGBM
- **Performance**: 95.35% ROC-AUC
- **Accuracy**: 63.8% precision
- **Speed**: Under 1 second training

### **Key Files**
- `PROJECT_DOCUMENTATION.md` - Complete project documentation
- `1_eda_analysis.py` - Data exploration
- `2_feature_engineering_production_ready.py` - Feature creation
- `criteo_final_4models.py` - Model training
- `pcb_dataset_final.csv` - Original dataset

### **For Your PG-DBDA Project**
This project demonstrates:
- Complete ML pipeline
- Real-world dataset handling
- Industry-standard algorithms
- Business problem solving
- Professional documentation

### **Business Impact**
- Optimize advertising spend
- Predict revenue opportunities
- Improve bidding strategies
- Reduce wasted ad budget

---

**Ready for your PG-DBDA submission! **
