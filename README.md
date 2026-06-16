# 🧠 Biochar Stabilized Resilient Modulus
Resilient Modulus prediction using Physics-Informed Machine Learning (PI-MS) for biochar-stabilized soil.
This repository provides a graphical user interface (GUI) for estimating resilient modulus (MR) of biochar-enhanced soils using laboratory-based soil parameters and physics-informed machine learning techniques.


# Running the Application

## Method 1: With Python Environment

### 1. Place the following files in the same directory:

- app.py  
- PINN_MS_model.h5  
- physics_model.pkl  
- physics_stats.pkl  
- scaler_X1.pkl  
- scaler_X2.pkl  
- scaler_X3.pkl  
- Final_SHAP_Background.pkl  
- requirements.txt  


### 2. Open PowerShell

Press:

Win + R  
Type:

powershell  

Press Enter.


### 3. Navigate to the directory containing the files

Use the command below:

cd "Your downloaded path"

Example:

cd "D:\Biochar_Stabilized_Resilient_Modulus"

Replace the example path with your actual folder location.


### 4. Run the application

Execute the following command:

streamlit run app.py


### 5. Launching the Interface

Wait approximately **5–20 seconds** (depending on system performance).

The application will automatically open in your browser at:

http://localhost:8501


# Input Parameters

The graphical interface requires the following input parameters:

- **w** — Moisture Content (%)  
- **gd** — Dry Unit Weight (kN/m³)  
- **NFT** — Number of Freeze–Thaw Cycles  
- **wPI** — Plasticity Index (%)  
- **sc** — Confining Stress (kPa)  
- **sd** — Deviator Stress (kPa)  


# Output

The system generates:

- Predicted **Resilient Modulus (MR)**  
- Strength Classification Gauge  
- SHAP-based Feature Contribution Visualization  

These outputs assist in understanding the relative importance of each input parameter in the prediction process.


# Notes

- Ensure all model files (`.pkl`, `.h5`) are placed in the same folder as `app.py`.  
- Recommended Python version: **3.9 – 3.11**  
- The first run may take slightly longer due to model initialization.  


# Required Software Installation Links

If the required software is not installed, download using the links below:

Python installation:  
https://www.python.org/downloads/

Anaconda distribution (optional):  
https://www.anaconda.com/download

Visual Studio Code (recommended editor):  
https://code.visualstudio.com/

PyCharm IDE:  
https://www.jetbrains.com/pycharm/

Streamlit documentation:  
https://docs.streamlit.io/

If you are using **PyCharm** or **VS Code**, the application can also be executed directly from the IDE terminal.


# About

This project presents a Physics-Informed Machine Learning (PI-MS) based graphical system developed for predicting the Resilient Modulus (MR) of biochar-stabilized soil.

The framework integrates machine learning models, physics-based soil relationships, and explainable artificial intelligence (SHAP) to provide reliable and interpretable predictions suitable for research and engineering applications.

# Contact

For technical support, queries, or research collaboration:

**Aman Mishra**  
Geotechnical Engineering  
 Raipur, India  

📧 Email: amanm2620@gmail.com

