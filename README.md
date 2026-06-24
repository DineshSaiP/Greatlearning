# Project Overview

This repository contains a collection of Python scripts and Jupyter notebooks for performing Exploratory Data Analysis (EDA) and Simple Linear Regression (SL‑Reg) across multiple days. Each notebook (`*.ipynb`) is paired with a corresponding Python script (`*.py`) that demonstrates the same analysis in a scriptable format.

## Table of Contents

- [Project Overview](#project-overview)
- [Installation](#installation)
- [Usage](#usage)
  - [Running the notebooks](#running-the-notebooks)
  - [Running the Python scripts](#running-the-python-scripts)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```

2. **Create a virtual environment (optional but recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows use `venv\Scripts\activate`
   ```

3. **Install the required packages**
   ```bash
   pip install -r requirements.txt
   ```
   If a `requirements.txt` file is not present, you can install the common dependencies manually:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn jupyter
   ```

## Usage

### Running the notebooks

Open any of the Jupyter notebooks with:
```bash
jupyter notebook "EDA Day1.ipynb"
```
This will launch the Jupyter interface in your browser where you can step through the analysis.

### Running the Python scripts

Each notebook has an equivalent `.py` script that can be executed directly from the command line:
```bash
python "EDA Day1.py"
```
The scripts output plots and results to the console and save figures in the current directory.

## Contributing

Contributions are welcome! Please follow these steps:

1. **Fork the repository**
2. **Create a new branch** for your feature or bug fix:
   ```bash
   git checkout -b my-feature-branch
   ```
3. **Make your changes** and ensure they follow the existing coding style.
4. **Add tests** if applicable and run the existing test suite.
5. **Commit your changes** with a clear commit message:
   ```bash
   git commit -m "Add feature: ..."
   ```
6. **Push to your fork** and open a Pull Request against the `main` branch.

Please make sure your code is well‑documented and that the README is updated if you add new functionality.

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.
