# Enterprise Insider Threat Detection System

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

A comprehensive machine learning-based system to detect and predict insider threats or corporate espionage using employee behavioral data.

## 📋 Table of Contents
- [Features](#-features)
- [Prerequisites](#-prerequisites)
- [Tech Stack](#️-tech-stack)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Usage](#-usage)
- [Examples](#-examples)
- [Project Structure](#project-structure)
- [API Documentation](#api-documentation)
- [Contributing](#contributing)
- [Security](#security)
- [License](#license)
- [Contact](#contact)

## 🔧 Features

- Real-time monitoring dashboard (Streamlit)
- Employee behavior analysis
- Risk scoring and assessment
- Anomaly detection using Isolation Forest
- Interactive visualizations
- Trend analysis
- Advanced analytics with PCA
- 3D PCA risk analysis
- Employee profile visualizations
- Simulated live activity feed

## 📋 Prerequisites

Before you begin, ensure you have the following installed:
- Python 3.8 or higher
- pip (Python package manager)
- Git
- A modern web browser
- At least 4GB RAM
- 1GB free disk space

## 🛠️ Tech Stack

- Python 3.x
- Streamlit
- Scikit-learn
- Pandas, NumPy
- Matplotlib, Seaborn, Plotly

## 📦 Installation

1. Clone the repository:
```bash
git clone https://github.com/SandeepSanthaKumar04/insider-threat-detection.git
cd insider-threat-detection
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Unix or MacOS
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
streamlit run app.py
```

## ⚙️ Configuration

The application can be configured through environment variables or a config file:

```python
# config.py
API_KEY = "your_api_key"
MODEL_PATH = "models/"
DATA_PATH = "data/"
LOG_LEVEL = "INFO"
```

## 🚀 Usage

1. Launch the application using the command above
2. Access the web interface through your browser (default: http://localhost:8501)
3. Navigate through different sections:
   - Dashboard
   - Employee Profiles
   - Risk Analysis
   - Anomaly Detection
   - Trend Analysis
   - Advanced Analytics

## 📝 Examples

### Basic Usage
```python
import streamlit as st
from insider_threat import ThreatDetector

detector = ThreatDetector()
results = detector.analyze_employee_data(employee_id="EMP001")
st.write(results)
```

### Custom Configuration
```python
from insider_threat import ThreatDetector

detector = ThreatDetector(
    risk_threshold=0.8,
    model_type="isolation_forest",
    sensitivity="high"
)
```

## Project Structure

```
insider-threat-detection/
├── app.py              # Main application file
├── requirements.txt    # Python dependencies
├── README.md          # Project documentation
├── data/              # Data directory
│   └── sample_data.csv
├── docs/              # Documentation
│   └── ARCHITECTURE.md
├── tests/             # Test files
├── models/            # Trained models
└── config/            # Configuration files
```

## 📚 API Documentation

For detailed API documentation, please visit our [Wiki](https://github.com/SandeepSanthaKumar04/insider-threat-detection/wiki).

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## 🔒 Security

Please report any security issues to sandeepsanthakumar04@gmail.com. For more information, see our [Security Policy](SECURITY.md).

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact

- **Author**: Sandeep Santhakumar
- **Email**: sandeepsanthakumar04@gmail.com
- **GitHub**: [@SandeepSanthaKumar04](https://github.com/SandeepSanthaKumar04)
- **Project Link**: [https://github.com/SandeepSanthaKumar04/insider-threat-detection](https://github.com/SandeepSanthaKumar04/insider-threat-detection)
