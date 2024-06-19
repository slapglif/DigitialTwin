# 🩸 Trauma Model in PyTorch 🧬

## Introduction ✨

This repository contains a refactored implementation of a trauma model originally written in a domain-specific language for biological modeling. The model simulates the complex interplay of hemodynamics, coagulation, and inflammation following a traumatic injury. 

This PyTorch version offers significant advantages over the original implementation, including:

- **🚀 Enhanced Performance:**  Leveraging PyTorch's tensor operations and potential for GPU acceleration, this version runs significantly faster, enabling more efficient simulations and analysis.
- **🧱 Improved Modularity:**  The code is organized into separate modules for hemodynamics, coagulation, inflammation, and utility functions, promoting code reusability, readability, and maintainability.
-  **⚙️ Centralized Configuration:**  Model parameters and settings are defined in a `config.yaml` file, making it easy to modify and experiment with different scenarios.
-  **📝 Comprehensive Documentation:**  Detailed docstrings and inline comments provide clear explanations of the code, making it easier to understand and extend.

## Getting Started 🚀

### 1. Installation 🧰

- Make sure you have Python 3.7 or higher installed.
- Install the required packages:

   ```bash
   pip install torch torchdiffeq pyyaml loguru tqdm matplotlib pandas
   ```

### 2. Configuration ⚙️

- Open the `config.yaml` file and adjust the model parameters and settings to match your desired scenario. 
- The file is well-commented, explaining each parameter and its role in the model.

### 3. Running the Simulation 🧬

- Execute the `main.py` script:

   ```bash
   python main.py
   ```

- The script will:
   - Load the configuration from `config.yaml`.
   - Initialize the model and set the initial conditions.
   - Solve the differential equations over the specified time span.
   - Save the simulation results to a CSV file in the `output` directory.
   - Generate visualizations of key results and save them in the `output/figures` directory.
   - Log important information and progress updates to the console and a log file in the `logs` directory.

### 4. Exploring the Results 📊

- The simulation results are saved as a CSV file (`simulation_results.csv`) in the `output` directory. You can open this file in a spreadsheet program or use Pandas in Python to analyze the data.
- Visualizations of blood volume/pressure and cytokine levels are saved as PNG files in the `output/figures` directory. These plots provide a visual representation of the model's behavior over time.
- The log file in the `logs` directory contains detailed information about the simulation process, including parameter values, progress updates, and any warnings or errors encountered.
