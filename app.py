import numpy as np
from flask import Flask, request, jsonify, session
from flask_session import Session
import flask
import sqlite3
import pandas as pd
import pickle
from prepare_data import prep_for_modeling

with open("best_lgbm_claim_classifier.pkl", "rb") as f:
   model = pickle.load(f)

app = Flask(__name__)
app.config.from_object(__name__)

@app.route("/", methods=["GET", "POST"]) 
def home():
    return flask.render_template("homepage.html")

@app.route("/return_query", methods=["POST", "GET"]) 
def return_query():
    conn = sqlite3.connect("car_insurance_claim.sqlite")
    query_string = request.form["query"]
    query_string.replace("SELECT ", "SELECT ID, ")
    df = pd.read_sql(query_string, conn)
    policy_ids = df["ID"].values.copy()
    session['user_policy_ids'] = policy_ids
    sample_to_display = df.iloc[:10]
    html_display = sample_to_display.to_html(header=True)
    
    return flask.render_template("query_options.html", html_table = html_display)

@app.route("/classify_policys", methods=["POST", "GET"])
def classify_policys():
    policy_ids = session.get("user_policy_ids", None)
    policy_ids = tuple(policy_ids)
    conn = sqlite3.connect("car_insurance_claim.sqlite")
    df = pd.read_sql(f"SELECT * FROM car_insurance_claims_classification WHERE ID in {policy_ids}", conn)
    X = prep_for_modeling(df)
    preds = model.predict_proba(X)
    preds_df = pd.DataFrame(preds)
    policy_ids_series = pd.Series(policy_ids)
    df_preds = pd.concat([policy_ids_series, preds_df], axis=1)
    df_preds.columns = ["Policy_ID ", " Probability No Claim ", " Probability Claim "]
    df_preds = df_preds.round(3)
    preds_html = df_preds.to_html(header=True)
    return flask.render_template("predictions.html", preds_html = preds_html)

    


if __name__=="__main__":
    app.secret_key="jeffs_key"
    app.config['SESSION_TYPE'] = 'filesystem'
    sess = Session(app)
    sess.init_app(app)
    app.run(port=5000, debug=True)
