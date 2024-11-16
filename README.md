# Twitter Sentiment Analysis with LSTM

## Overview
This project is a Streamlit-based web application designed to perform sentiment analysis on Twitter text data. It leverages a pre-trained LSTM (Long Short-Term Memory) model to classify input messages as either **positive** or **not positive**. The application provides a user-friendly interface for real-time sentiment prediction.

## Features
- **Text Cleaning:** Automatically preprocesses input text by removing unwanted characters, URLs, HTML tags, and more.
- **Sentiment Prediction:** Utilizes a trained LSTM model to classify text sentiment.
- **Interactive Interface:** Easy-to-use interface built with Streamlit.

## Installation

### Prerequisites
- Python 3.8 or higher
- Virtual environment (optional but recommended)

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/vineethsaivs/twitter-sentiment-analysis-lstm.git
   cd twitter-sentiment-analysis-lstm
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # For Linux/Mac
   venv\Scripts\activate   # For Windows
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download the required `model.h5` and `tokenizer.pickle` files if they are not included in the repository. These files should be placed in the project directory.

5. Run the application:
   ```bash
   streamlit run app.py
   ```

## File Structure
- **`app.py`**: The main Streamlit application script.
- **`model.h5`**: The pre-trained LSTM model for sentiment classification.
- **`tokenizer.pickle`**: Tokenizer used for text preprocessing.
- **`.gitattributes`**: Configuration file for Git LFS.
- **`.gitignore`**: Excludes unnecessary files (e.g., virtual environment) from the repository.
- **`requirements.txt`**: List of Python dependencies.

## Usage
1. Open the application in your web browser by following the URL provided by Streamlit.
2. Enter a Twitter message in the text area.
3. Click on the **Predict** button to analyze the sentiment.
4. The app will display whether the message is positive or not positive.

## Text Preprocessing
The app applies the following preprocessing steps to the input text:
- Converts text to lowercase.
- Removes URLs, HTML tags, punctuation, and numeric characters.
- Removes stopwords (common words that do not add significant meaning).
- Applies stemming to reduce words to their base form.

## Requirements
Below are the main dependencies for this project:
- `streamlit`
- `pandas`
- `numpy`
- `keras`
- `nltk`

Refer to `requirements.txt` for the full list of dependencies.

## Known Issues
- Ensure that the `model.h5` and `tokenizer.pickle` files are in the project directory.
- The `model.h5` file uses custom objects, which are handled with a custom mapping in `app.py`.

## Contributions
Contributions to this project are welcome! Feel free to fork the repository and create a pull request with your improvements.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Contact
For questions or feedback, please open an issue on the [GitHub repository](https://github.com/vineethsaivs/twitter-sentiment-analysis-lstm).

