import numpy as np
from flask import Flask, request, jsonify, session
from flask_session import Session
from flask_mail import Mail, Message
import flask
import sqlite3
import pandas as pd
import pickle
from prepare_data import prep_for_modeling
import os

with open("best_lgbm_claim_classifier.pkl", "rb") as f:
   model = pickle.load(f)

app = Flask(__name__)
app.config.from_object(__name__)

mail_settings = {
    "MAIL_SERVER": 'smtp.gmail.com',
    "MAIL_PORT": 465,
    "MAIL_USE_TLS": False,
    "MAIL_USE_SSL": True,
    "MAIL_USERNAME": os.environ['EMAIL_USER'],
    "MAIL_PASSWORD": os.environ['EMAIL_PASSWORD']
}

app.config.update(mail_settings)
mail = Mail(app)


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
    html_display = sample_to_display.to_html(header=True, classes="blueTable")
    
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
    df_preds_sample = df_preds.sample(10)
    preds_html = df_preds_sample.to_html(header=True, classes="blueTable")
    df_preds.to_csv("predictions.csv")
    return flask.render_template("predictions.html", preds_html = preds_html)

@app.route("/send_email", methods=["POST","GET"])
def sent_email():
    msg = Message("Insurance Policy Claim Predictions", sender=app.config.get("MAIL_USERNAME"),recipients=["sethweiland@g.ucla.edu"], body="Here are the insurnace policy claim predictions you requested")
    with app.open_resource("predictions.csv") as predicts:
        msg.attach("predictions.csv","text/csv", predicts.read())
    mail.send(msg) 
    return "Message Sent" 


if __name__=="__main__":
    app.secret_key="jeffs_key"
    app.config['SESSION_TYPE'] = 'filesystem'
    sess = Session(app)
    sess.init_app(app)
    app.run(port=5000, debug=True)
