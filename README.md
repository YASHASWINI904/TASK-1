# NestEstimate — Smart Home Price Prediction

A custom website for a housing-price estimation internship project. It uses **HTML, CSS and JavaScript** for the frontend, **Python Flask** for the backend, and a **linear regression model implemented from scratch with NumPy**. It does not use Streamlit or scikit-learn.

## Features
- Premium, minimal visual design with sage-green accents, a compact top header, and a dedicated left navigation sidebar
- Distinct dashboard workspaces with active left-sidebar navigation and a compact utility header: Overview, Price Predictor, Model Insights, Saved Properties, and About
- Responsive dashboard with property form and instant predictions
- Features: built-up area, bedrooms, location/locality score (1–10), house age, distance to important facilities
- 600-row synthetic sample dataset in `data/housing_data.csv`
- Deterministic 80/20 train/test split
- MSE, RMSE, R², actual-vs-predicted scatter plot, model coefficients
- Username/password registration and login with Flask sessions\n- Dedicated My Account page with editable full name, email, and phone number; read-only username and account creation date\n- Account details saved in SQLite and retained between sign-ins
- Account-specific saved properties stored in SQLite; favourites persist across sign-outs and logins and are isolated between users
- Sign out securely clears the login session
- Save property estimates, see the saved count in the sidebar, and remove saved entries
- Download a CSV report of saved properties
- Download sample dataset from the website
- Input validation and JSON prediction endpoint

## Requirements
- Python 3.10 or newer recommended
- Internet connection for the optional Chart.js and remote hero image. Core backend/model works locally.

## Run on Windows (VS Code)
1. Extract the ZIP.
2. Open the extracted `NestEstimate_Web` folder in VS Code.
3. Open **Terminal → New Terminal**.
4. Run:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```

5. Open `http://127.0.0.1:5000` in your browser.

If PowerShell blocks activation, run this in the same terminal:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```
Or use Command Prompt:
```bat
.venv\Scripts\activate.bat
```

## Model details
- `model.py` implements standardization and batch gradient descent using NumPy.
- The data is shuffled with a fixed seed for a repeatable 80/20 train/test split.
- MSE = average squared prediction error; RMSE = square root of MSE; R² measures the fraction of variance explained by the model.
- The included dataset is synthetic, created for demonstration and testing. It is not a verified real-estate market dataset and predictions must not be presented as certified property valuations.

## Project structure
```text
NestEstimate_Web/
├── app.py
├── nestestimate.db  (created automatically on first run)
├── model.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   └── housing_data.csv
├── templates/
│   └── index.html
└── static/
    └── style.css
```

## Quick checks
- `http://127.0.0.1:5000/health` returns basic app/model status JSON.
- `http://127.0.0.1:5000/dataset.csv` downloads the sample CSV.

## GitHub / deployment
You can push this folder to a GitHub repository. For a deployment platform that supports Python web apps, configure the start command as `gunicorn app:app` (add `gunicorn` to requirements first) and set Python dependencies from `requirements.txt`. Do not commit `.venv/`.

## Saved properties and reports

Use **Save this property** after making a prediction to store the estimate in the current browser. The Saved Properties section lets you remove entries and download a CSV report. Saved estimates are stored in the local SQLite database (`nestestimate.db`) and associated with the signed-in account, so they persist across sessions on this installation. Back up this database to preserve accounts and saved properties. The generated housing dataset is synthetic, so reports are for demonstration only.


Compatibility note: This version does not depend on Pandas. Housing CSV data is read with Python’s built-in csv module to avoid Pandas parser DLL loading issues on some Windows-managed systems.

## Account settings

After signing in, open **Settings** from the left navigation. Users can save appearance, currency and area-unit preferences, a default city, and preferred property criteria. The page also supports changing the account password and deleting the account with confirmation. Saved properties remain scoped to the signed-in user and are deleted with that account.

Note: currency and area-unit choices are stored preferences; the current prediction model still calculates in price lakhs and square feet. Live currency conversion and automatic property filtering are not implemented. For deployment, set a strong `NESTESTIMATE_SECRET_KEY` environment variable and use HTTPS.
