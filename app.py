from pathlib import Path
import os
import io
import csv
import re
import numpy as np
import sqlite3
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash
from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for, session, flash
from model import FEATURES, LinearRegressionScratch, train_test_split, regression_metrics

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "housing_data.csv"
app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("NESTESTIMATE_SECRET_KEY", "change-this-secret-before-deployment")
DB_PATH = ROOT / "nestestimate.db"

def db_connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with db_connect() as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT NOT NULL UNIQUE, password_hash TEXT NOT NULL, full_name TEXT NOT NULL DEFAULT '', email TEXT NOT NULL DEFAULT '', phone TEXT NOT NULL DEFAULT '', theme TEXT NOT NULL DEFAULT 'light', currency TEXT NOT NULL DEFAULT 'INR', area_unit TEXT NOT NULL DEFAULT 'sq ft', default_city TEXT NOT NULL DEFAULT '', property_type TEXT NOT NULL DEFAULT 'Any', budget_min REAL NOT NULL DEFAULT 0, budget_max REAL NOT NULL DEFAULT 0, preferred_bedrooms INTEGER NOT NULL DEFAULT 3, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)")
        # Migrate existing installations without deleting registered accounts.
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(users)").fetchall()}
        for name in ("full_name", "email", "phone", "theme", "currency", "area_unit", "default_city", "property_type", "budget_min", "budget_max", "preferred_bedrooms"):
            if name not in columns:
                conn.execute(f"ALTER TABLE users ADD COLUMN {name} TEXT NOT NULL DEFAULT ''")
        conn.execute("""CREATE TABLE IF NOT EXISTS saved_properties (
            id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL,
            city TEXT NOT NULL, area REAL NOT NULL, bedrooms INTEGER NOT NULL,
            locality_score INTEGER NOT NULL, age REAL NOT NULL, distance REAL NOT NULL,
            price REAL NOT NULL, saved_at TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE)""")

def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
        return view(*args, **kwargs)
    return wrapped

init_db()

def load_and_train():
    # Read the CSV with Python's built-in csv module so Pandas is not required.
    # This avoids Windows application-control blocks of Pandas compiled extensions.
    with DATA_PATH.open("r", newline="", encoding="utf-8-sig") as csv_file:
        data = list(csv.DictReader(csv_file))
    if not data:
        raise RuntimeError(f"No housing records found in {DATA_PATH}")
    X = np.array([[float(row[feature]) for feature in FEATURES] for row in data], dtype=float)
    y = np.array([float(row["price_lakhs"]) for row in data], dtype=float)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegressionScratch().fit(X_train, y_train)
    metrics = regression_metrics(y_test, model.predict(X_test))
    predicted = model.predict(X_test)
    coef, intercept = model.coefficients_original_scale()
    chart = [
        {"actual": round(float(a), 2), "predicted": round(float(p), 2)}
        for a, p in zip(y_test, predicted)
    ]
    coeffs = [
        {"feature": "Built-up Area (sq. ft.)", "value": float(coef[0]), "display": f"{coef[0]:.4f}"},
        {"feature": "Bedrooms", "value": float(coef[1]), "display": f"{coef[1]:.4f}"},
        {"feature": "Location / Locality Score", "value": float(coef[2]), "display": f"{coef[2]:.4f}"},
        {"feature": "House Age (years)", "value": float(coef[3]), "display": f"{coef[3]:.4f}"},
        {"feature": "Distance to Facilities (km)", "value": float(coef[4]), "display": f"{coef[4]:.4f}"},
        {"feature": "Intercept", "value": float(intercept), "display": f"{intercept:.4f}"},
    ]
    return data, model, metrics, chart, coeffs

DATA, MODEL, METRICS, CHART, COEFFICIENTS = load_and_train()

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        if len(username) < 3 or len(password) < 8:
            return render_template("auth.html", mode="register", error="Use a username with at least 3 characters and a password with at least 8 characters.")
        try:
            with db_connect() as conn:
                cur = conn.execute("INSERT INTO users(username, password_hash) VALUES (?, ?)", (username, generate_password_hash(password)))
                session.clear()
                session["user_id"] = cur.lastrowid
                session["username"] = username
            return redirect(url_for("home"))
        except sqlite3.IntegrityError:
            return render_template("auth.html", mode="register", error="That username already exists. Please choose another.")
    return render_template("auth.html", mode="register", error=None)

@app.route("/login", methods=["GET", "POST"])
def login():
    if "user_id" in session:
        return redirect(url_for("home"))
    error = None
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        with db_connect() as conn:
            user = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        if user and check_password_hash(user["password_hash"], password):
            session.clear()
            session["user_id"] = user["id"]
            session["username"] = user["username"]
            return redirect(url_for("home"))
        error = "Incorrect username or password. Please try again."
    return render_template("auth.html", mode="login", error=error)

@app.route("/logout", methods=["POST"])
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route("/account", methods=["GET", "POST"])
@login_required
def account():
    user_id = session["user_id"]
    error = None
    success = None
    with db_connect() as conn:
        user = conn.execute("SELECT id, username, full_name, email, phone, created_at FROM users WHERE id = ?", (user_id,)).fetchone()
    if request.method == "POST":
        full_name = request.form.get("full_name", "").strip()
        email = request.form.get("email", "").strip()
        phone = request.form.get("phone", "").strip()
        if len(full_name) > 100 or len(email) > 254 or len(phone) > 25:
            error = "Please check the length of the details you entered."
        elif email and not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", email):
            error = "Enter a valid email address."
        elif phone and not re.fullmatch(r"[+0-9() .-]{7,25}", phone):
            error = "Enter a valid phone number."
        else:
            with db_connect() as conn:
                conn.execute("UPDATE users SET full_name = ?, email = ?, phone = ? WHERE id = ?",
                             (full_name, email, phone, user_id))
            success = "Your account details have been updated."
            with db_connect() as conn:
                user = conn.execute("SELECT id, username, full_name, email, phone, created_at FROM users WHERE id = ?", (user_id,)).fetchone()
    return render_template("account.html", user=user, error=error, success=success,
                           username=session.get("username", "User"))

@app.route("/settings", methods=["GET"])
@login_required
def settings():
    return redirect(url_for("settings_section", section="overview"))

@app.route("/settings/<section>", methods=["GET", "POST"])
@login_required
def settings_section(section):
    allowed_sections = {"overview", "general", "property", "security", "privacy"}
    if section not in allowed_sections:
        return redirect(url_for("settings_section", section="privacy"))
    user_id = session["user_id"]
    error = None
    success = None
    with db_connect() as conn:
        user = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    if request.method == "POST":
        action = request.form.get("action", "")
        if action == "preferences":
            theme = request.form.get("theme", "light")
            currency = request.form.get("currency", "INR")
            area_unit = request.form.get("area_unit", "sq ft")
            default_city = request.form.get("default_city", "").strip()
            if theme not in ("light", "dark") or currency not in ("INR", "USD") or area_unit not in ("sq ft", "sq yd") or len(default_city) > 100:
                error = "Please check your general preferences."
            else:
                with db_connect() as conn:
                    conn.execute("UPDATE users SET theme=?, currency=?, area_unit=?, default_city=? WHERE id=?", (theme, currency, area_unit, default_city, user_id))
                success = "General preferences saved successfully."
        elif action == "property_preferences":
            property_type = request.form.get("property_type", "Any")
            try:
                budget_min = float(request.form.get("budget_min", "0") or 0)
                budget_max = float(request.form.get("budget_max", "0") or 0)
                preferred_bedrooms = int(request.form.get("preferred_bedrooms", "3"))
                if property_type not in ("Any", "Apartment", "Independent house", "Villa") or not (0 <= budget_min <= 1000000 and 0 <= budget_max <= 1000000) or not (1 <= preferred_bedrooms <= 10):
                    raise ValueError
                if budget_min and budget_max and budget_min > budget_max:
                    error = "Minimum budget cannot be greater than maximum budget."
                else:
                    with db_connect() as conn:
                        conn.execute("UPDATE users SET property_type=?, budget_min=?, budget_max=?, preferred_bedrooms=? WHERE id=?", (property_type, budget_min, budget_max, preferred_bedrooms, user_id))
                    success = "Property preferences saved successfully."
            except (ValueError, TypeError):
                error = "Please enter valid property preferences."
        elif action == "password":
            current_password = request.form.get("current_password", "")
            new_password = request.form.get("new_password", "")
            confirm_password = request.form.get("confirm_password", "")
            if not check_password_hash(user["password_hash"], current_password):
                error = "Your current password is incorrect."
            elif len(new_password) < 8:
                error = "Your new password must contain at least 8 characters."
            elif new_password != confirm_password:
                error = "The new passwords do not match."
            else:
                with db_connect() as conn:
                    conn.execute("UPDATE users SET password_hash=? WHERE id=?", (generate_password_hash(new_password), user_id))
                success = "Password changed successfully."
        with db_connect() as conn:
            user = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    cities = sorted({row["city"] for row in DATA})
    return render_template("settings.html", user=user, cities=cities, error=error, success=success, section=section, username=session.get("username", "User"))

@app.route("/account/delete", methods=["POST"])
@login_required
def delete_account():
    if request.form.get("confirm_delete") != "DELETE":
        flash("Please confirm account deletion.")
        return redirect(url_for("settings_section", section="privacy"))
    user_id = session["user_id"]
    with db_connect() as conn:
        conn.execute("DELETE FROM saved_properties WHERE user_id = ?", (user_id,))
        conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
    session.clear()
    return redirect(url_for("register"))

def get_user_theme(user_id):
    with db_connect() as conn:
        row = conn.execute("SELECT theme FROM users WHERE id = ?", (user_id,)).fetchone()
    return row["theme"] if row else "light"

@app.route("/")
@login_required
def home():
    defaults = {"area": 1200, "bedrooms": 3, "locality_score": 8, "age": 5, "distance": 2.0}
    default_prediction = float(MODEL.predict([[defaults["area"], defaults["bedrooms"], defaults["locality_score"], defaults["age"], defaults["distance"]]])[0])
    return render_template(
        "index.html", metrics=METRICS, record_count=len(DATA),
        train_count=int(len(DATA)*0.8), prediction=max(0, default_prediction),
        defaults=defaults, coefficients=COEFFICIENTS,
        chart_data=CHART, cities=sorted({row["city"] for row in DATA}), username=session.get("username", "User"), theme=get_user_theme(session["user_id"])
    )

@app.route("/predict", methods=["POST"])
@login_required
def predict():
    try:
        payload = request.get_json(force=True)
        area = float(payload["area"])
        bedrooms = int(payload["bedrooms"])
        locality_score = int(payload["locality_score"])
        age = float(payload["age"])
        distance = float(payload["distance"])
        if not (200 <= area <= 10000):
            raise ValueError("Built-up area must be between 200 and 10,000 sq. ft.")
        if not (1 <= bedrooms <= 10):
            raise ValueError("Bedrooms must be between 1 and 10.")
        if not (1 <= locality_score <= 10):
            raise ValueError("Location / Locality Score must be between 1 and 10.")
        if not (0 <= age <= 150):
            raise ValueError("House age must be between 0 and 150 years.")
        if not (0 <= distance <= 100):
            raise ValueError("Distance to facilities must be between 0 and 100 km.")
        value = float(MODEL.predict([[area, bedrooms, locality_score, age, distance]])[0])
        return jsonify({"price": round(max(0, value), 2), "city": payload.get("city", "Not specified"),
                        "area": area, "bedrooms": bedrooms, "locality_score": locality_score,
                        "age": age, "distance": distance})
    except (ValueError, KeyError, TypeError) as exc:
        return jsonify({"error": str(exc)}), 400

@app.route("/dataset.csv")
@login_required
def dataset():
    return send_file(DATA_PATH, as_attachment=True, download_name="housing_data.csv", mimetype="text/csv")

@app.route("/api/saved-properties", methods=["GET", "POST"])
@login_required
def saved_properties():
    user_id = session["user_id"]
    if request.method == "GET":
        with db_connect() as conn:
            rows = conn.execute("SELECT id, city, area, bedrooms, locality_score, age, distance, price, saved_at AS savedAt FROM saved_properties WHERE user_id = ? ORDER BY id DESC", (user_id,)).fetchall()
        return jsonify([dict(row) for row in rows])
    try:
        p = request.get_json(force=True)
        city = str(p.get("city", "")).strip()
        area = float(p["area"]); bedrooms = int(p["bedrooms"])
        locality = int(p["locality_score"]); age = float(p["age"])
        distance = float(p["distance"]); price = float(p["price"])
        if not city or not (200 <= area <= 10000 and 1 <= bedrooms <= 10 and 1 <= locality <= 10 and 0 <= age <= 150 and 0 <= distance <= 100 and price >= 0):
            raise ValueError("Property details are outside the allowed range.")
        from datetime import datetime
        saved_at = datetime.now().strftime("%Y-%m-%d")
        with db_connect() as conn:
            cur = conn.execute("INSERT INTO saved_properties(user_id, city, area, bedrooms, locality_score, age, distance, price, saved_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", (user_id, city, area, bedrooms, locality, age, distance, price, saved_at))
        return jsonify({"id": cur.lastrowid, "savedAt": saved_at}), 201
    except (ValueError, KeyError, TypeError) as exc:
        return jsonify({"error": str(exc)}), 400

@app.route("/api/saved-properties/<int:property_id>", methods=["DELETE"])
@login_required
def delete_saved_property(property_id):
    with db_connect() as conn:
        cur = conn.execute("DELETE FROM saved_properties WHERE id = ? AND user_id = ?", (property_id, session["user_id"]))
    if cur.rowcount == 0:
        return jsonify({"error": "Saved property not found."}), 404
    return jsonify({"ok": True})

@app.route("/health")
def health():
    return jsonify({"status": "ok", "records": len(DATA), "model": "linear regression from scratch"})

if __name__ == "__main__":
    app.run(debug=True)
