# TheSpot - Drink Sales Predictor

TheSpot is a web application that predicts drink sales based on historical sales data, weather forecasts, and event information. The application uses machine learning to provide accurate sales predictions for the upcoming days.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Routes](#routes)
- [Contributing](#contributing)
- [License](#license)

## Features

- Upload CSV files containing historical sales data.
- Fetch historical weather data and merge it with sales data.
- Train a machine learning model (Random Forest Regressor) to predict future sales.
- Set up predictions by selecting future dates and events.
- Fetch weather forecasts for future dates.
- Display predicted sales for each drink and a summary of total sales.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/Germuu/TheSpot.git
    cd TheSpot
    ```

2. Install dependencies using Poetry:
    ```sh
    poetry install
    ```

3. Set up environment variables:
    - Create a `.env` file in the root directory.
    - Add the following environment variables:
        ```
        FLASK_APP=app.py
        FLASK_ENV=development
        SECRET_KEY=your_secret_key
        ```

4. Run the application:
    ```sh
    poetry run flask run
    ```

## Usage

1. Open your web browser and navigate to `http://127.0.0.1:5000/`.
2. Upload a CSV file containing historical sales data.
3. Set up predictions by selecting future dates and events.
4. View the predicted sales for each drink and a summary of total sales.

## Project Structure

```
├── .gitignore
├── app.py
├── config.py
├── poetry.lock
├── pyproject.toml
├── README.md
├── validate_weather.py
├── __pycache__/
├── src/
│   ├── models/
│   │   ├── predictor.py
│   │   ├── train_data.csv
│   │   └── __pycache__/
│   ├── services/
│   │   ├── file_handler.py
│   │   └── __pycache__/
├── static/
│   └── style.css
├── templates/
│   ├── index.html
│   ├── predictions.html
│   ├── setup_prediction.html
│   └── weather_input.html
└── uploads/
    ├── data (2).csv
    ├── data_without_weather.csv
    └── synthetic_data.csv
```

## Routes

- `/` (GET): Home page. Upload sales data CSV file.
- `/upload` (POST): Upload CSV file and train the model.
- `/setup_prediction` (GET, POST): Set up predictions by selecting future dates and events.
- `/predict` (GET, POST): Display predicted sales for each drink and a summary of total sales.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.