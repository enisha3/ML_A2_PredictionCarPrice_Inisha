# A2 - Predicting Car Prices: Web-Based Application with Machine Learning Model

This project is a web app that uses a machine learning model with Dash by Plotly. It is designed to have a visually appealing interface while staying functional. If users skip any form fields, the system automatically fills in the missing values.

## Features

- **Machine Learning Integration**: A pre-trained machine learning model is incorporated into the application for predictions.
- **Dash by Plotly**: The application is built using Dash, providing interactive visualizations and seamless UI integration.
- **Form Imputation**: If users skip any fields in the form, the application automatically fills them using an imputation technique.
- **Responsive Design**: The layout is optimized for various screen sizes, ensuring accessibility on mobile and desktop devices.
- **Data Transformation**: Includes Min-Max scaling and imputation techniques for handling missing data.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/enisha3/ML_A2_PredictionCarPrice_Inisha.git

2. Navigate into the project directory:
   ```bash
   cd <yourproject_dir>

3. Run Dockerfile:
   ```bash
   docker compose up

4. Run the application:
   ```bash
    python3 app/app.py

## Project Structure

- `A2_PredictingCarPrice`: Consists of all the .ipynb and all related to this project.
- `app.py`: Main application file.
- `app/`: Folder containing pages UI.
- `model/`: Folder containing the machine learning model.
- `LinearRegression.py`: Script for handling classes.
- `.Dockerfile`: Installs list of dependencies needed to run the application.
- `docker-compose.yaml`: Needed to run the dockerize the container.
- `Screenshots`: Consists of images of mlflow and Ui Image that is runned in ml server.
- `README.md`: Project documentation.

## Usage

Once the app is running, users can enter data through the web interface. If they leave some fields empty, the system fills in the missing values automatically and makes predictions using the built-in machine learning model.

## RUNNING APPLICATION:
local mlflow logs: http://127.0.0.1:5000/<experiment_name: st125563-Inisha>   
docker hub image: https://hub.docker.com/repositories/inisha5563
brain lab server: https://inishacarpricepredictor-st125563.ml.brain.cs.ait.ac.th/

## Application running on server images:
![Imageone](Screenshots/UI_with_default_values_for_predictingcarprice.png)

![Imagetwo](Screenshots/UI_after_inserting_values_for_predictingcarprice.png)
