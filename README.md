<p align="center">
  <img src="https://cdn-icons-png.flaticon.com/512/6295/6295417.png" width="100" />
</p>
<p align="center">
    <h1 align="center">GENDER-CLASSIFICATION-USING-ARABIC-NAMES</h1>
</p>
<p align="center">
    <em>Predict Gender By Analyzing The Arabic First Name</em>
</p>
<p align="center">
	<img src="https://img.shields.io/github/license/Faris-Alanazi/Gender-Classification-Using-Arabic-Names?style=flat&color=0080ff" alt="license">
	<img src="https://img.shields.io/github/last-commit/Faris-Alanazi/Gender-Classification-Using-Arabic-Names?style=flat&color=0080ff" alt="last-commit">
	<img src="https://img.shields.io/github/languages/top/Faris-Alanazi/Gender-Classification-Using-Arabic-Names?style=flat&color=0080ff" alt="repo-top-language">
	<img src="https://img.shields.io/github/languages/count/Faris-Alanazi/Gender-Classification-Using-Arabic-Names?style=flat&color=0080ff" alt="repo-language-count">
<p>
<p align="center">
		<em>Developed with the software and tools below.</em>
</p>
<p align="center">
	<img src="https://img.shields.io/badge/TensorFlow-FF6F00.svg?style=flat&logo=TensorFlow&logoColor=white" alt="TensorFlow">
	<img src="https://img.shields.io/badge/scikitlearn-F7931E.svg?style=flat&logo=scikit-learn&logoColor=white" alt="scikitlearn">
	<img src="https://img.shields.io/badge/Jupyter-F37626.svg?style=flat&logo=Jupyter&logoColor=white" alt="Jupyter">
	<img src="https://img.shields.io/badge/Python-3776AB.svg?style=flat&logo=Python&logoColor=white" alt="Python">
	<img src="https://img.shields.io/badge/pandas-150458.svg?style=flat&logo=pandas&logoColor=white" alt="pandas">
	<img src="https://img.shields.io/badge/MLflow-0194E2.svg?style=flat&logo=MLflow&logoColor=white" alt="MLflow">
	<img src="https://img.shields.io/badge/NumPy-013243.svg?style=flat&logo=NumPy&logoColor=white" alt="NumPy">
</p>
<hr>

## 🔗 Quick Links

> - [📍 Overview](#-overview)
> - [📂 Repository Structure](#-repository-structure)
> - [🚀 Getting Started](#-getting-started)
>   - [⚙️ Installation](#️-installation)
>   - [🤖 Running Gender-Classification-Using-Arabic-Names](#-running-Gender-Classification-Using-Arabic-Names)
>   - [🥈 Run](#-tests)
> - [🤝 Contributing](#-contributing)

---

## 📍 Overview

The Gender-Classification-Using-Arabic-Names project utilizes machine learning to predict gender from Arabic names. Employing Jupyter Notebooks for preprocessing and modeling, the workflow includes cleaning the dataset, exploring data, and training classifiers.
the project provides a practical tool for gender inference using Arabic Names

---

## 📂 Repository Structure

```sh
└── Gender-Classification-Using-Arabic-Names/
    ├── _1_Exploratory_Data_Analysis.ipynb
    ├── _2_Preporcessing.ipynb
    ├── _3_BiLSTM_Model.ipynb
    ├── _4_Xgboost_Model.ipynb
    ├── app.py
    ├── requirements.txt
    ├── saved_models
    │   ├── XGBoost.bin
    └── temp.ipynb
```


---

## 🧩 Modules

<details closed><summary>.</summary>

| File                                                                                                                                                           | Summary                                                                                                                                                                                                                                                                                                                                    |
| ---                                                                                                                                                            | ---                                                                                                                                                                                                                                                                                                                                        |
| [_2_Preporcessing.ipynb](https://github.com/Faris-Alanazi/Gender-Classification-Using-Arabic-Names/blob/master/_2_Preporcessing.ipynb)                         | This preprocessing notebook enhances the gender classification dataset, transforming Arabic names into numerical features for ML models within the project's pipeline. It includes feature engineering to extract linguistic attributes, implements feature encoding for categorical data, and finalizes by storing the preprocessed data. |
| [_1_Exploratory_Data_Analysis.ipynb](https://github.com/Faris-Alanazi/Gender-Classification-Using-Arabic-Names/blob/master/_1_Exploratory_Data_Analysis.ipynb) | This notebook conducts the initial data exploration and preprocessing for Arabic names gender classification, encompassing data loading, visualization, null handling, alphabet filtering, and dataset cleaning.                                                                                                                           |
| [temp.ipynb](https://github.com/Faris-Alanazi/Gender-Classification-Using-Arabic-Names/blob/master/temp.ipynb)                                                 | This code snippet is integral for determining gender classification within a larger system focused on identity analysis.                                                                                                                                                                                                                   |
| [_4_Xgboost_Model.ipynb](https://github.com/Faris-Alanazi/Gender-Classification-Using-Arabic-Names/blob/master/_4_Xgboost_Model.ipynb)                         | The provided text is incomplete. Please provide the full repository structure or details of the code snippet for an accurate summary.                                                                                                                                                                                                      |
| [_3_BiLSTM_Model.ipynb](https://github.com/Faris-Alanazi/Gender-Classification-Using-Arabic-Names/blob/master/_3_BiLSTM_Model.ipynb)                           | Assuming the provided information is incomplete:The snippet orchestrates a key component within the repository, interfacing with other modules to maintain cohesive functionality.                                                                                                                                                         |
| [requirements.txt](https://github.com/Faris-Alanazi/Gender-Classification-Using-Arabic-Names/blob/master/requirements.txt)                                     | The `requirements.txt` file specifies dependencies for a web application that predicts gender from Arabic names using BiLSTM and XGBoost models, as part of the larger machine learning project.                                                                                                                                           |
| [app.py](https://github.com/Faris-Alanazi/Gender-Classification-Using-Arabic-Names/blob/master/app.py)                                                         | The `app.py` serves as a Streamlit web interface for gender prediction using Arabic names, utilizing a pre-trained XGBoost model and pre-processing with tokenization and feature scaling.                                                                                                                                                 |

</details>

---

## 🚀 Getting Started


### ⚙️ Installation

1. Clone the Gender-Classification-Using-Arabic-Names repository:

```sh
git clone https://github.com/Faris-Alanazi/Gender-Classification-Using-Arabic-Names
```

2. Change to the project directory:

```sh
cd Gender-Classification-Using-Arabic-Names
```

3. Install the dependencies:

```sh
pip install -r requirements.txt
```

### 🥈 Run

To execute run the streamlit Application, run:

```sh
streamlit run app.py
```

---

## 🤝 Contributing

Contributions are welcome! Here are several ways you can contribute:

- **[Submit Pull Requests](https://github/Faris-Alanazi/Gender-Classification-Using-Arabic-Names/blob/main/CONTRIBUTING.md)**: Review open PRs, and submit your own PRs.
- **[Join the Discussions](https://github/Faris-Alanazi/Gender-Classification-Using-Arabic-Names/discussions)**: Share your insights, provide feedback, or ask questions.
- **[Report Issues](https://github/Faris-Alanazi/Gender-Classification-Using-Arabic-Names/issues)**: Submit bugs found or log feature requests for Gender-classification-using-arabic-names.

<details closed>
    <summary>Contributing Guidelines</summary>

1. **Fork the Repository**: Start by forking the project repository to your GitHub account.
2. **Clone Locally**: Clone the forked repository to your local machine using a Git client.
   ```sh
   git clone https://github.com/Faris-Alanazi/Gender-Classification-Using-Arabic-Names
   ```
3. **Create a New Branch**: Always work on a new branch, giving it a descriptive name.
   ```sh
   git checkout -b new-feature-x
   ```
4. **Make Your Changes**: Develop and test your changes locally.
5. **Commit Your Changes**: Commit with a clear message describing your updates.
   ```sh
   git commit -m 'Implemented new feature x.'
   ```
6. **Push to GitHub**: Push the changes to your forked repository.
   ```sh
   git push origin new-feature-x
   ```
7. **Submit a Pull Request**: Create a PR against the original project repository. Clearly describe the changes and their motivations.

Once your PR is reviewed and approved, it will be merged into the main branch.

</details>

[**Return**](#-quick-links)

---
