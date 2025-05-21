import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
import shutil
from flask import Flask, render_template_string
import webbrowser
import threading

# Load your CSV file
file_path = "C:\\Users\\DELL\\Downloads\\gambling_data_enhanced.csv"
df = pd.read_csv(file_path)

# Encode categorical values
le = LabelEncoder()
df['game_type_encoded'] = le.fit_transform(df['game_type'])

# KMeans Clustering to identify behavior clusters
features_kmeans = df[['age', 'gambling_frequency', 'bet_amount', 'previous_wins', 'previous_losses']]
kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(features_kmeans)

# Linear Regression to predict chance to win
features = df[['age', 'monthly_income', 'gambling_frequency', 'bet_amount', 'previous_wins', 'previous_losses', 'game_type_encoded']]
target = df['win']
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
df['predicted_win_prob'] = model.predict(features)

# Analyze age group most addicted
age_addiction = df.groupby('age')['gambling_frequency'].mean().reset_index()

# Save results to a new CSV file
output_path = "C:\\Users\\DELL\\Downloads\\gambling_analysis_results.csv"
df.to_csv(output_path, index=False)

# Generate and save plots
plt.figure(figsize=(10, 5))
sns.lineplot(data=age_addiction, x='age', y='gambling_frequency')
plt.title("Addiction Trend by Age")
plt.xlabel("Age")
plt.ylabel("Average Gambling Frequency")
plt.tight_layout()
plt.savefig("age_addiction_trend.png")
plt.close()

sns.scatterplot(data=df, x='monthly_income', y='bet_amount', hue='cluster')
plt.title("Cluster Behavior by Income vs Bet Amount")
plt.tight_layout()
plt.savefig("cluster_income_bet.png")
plt.close()

sns.boxplot(data=df, x='game_type', y='gambling_frequency')
plt.title("Gambling Frequency by Game Type")
plt.tight_layout()
plt.savefig("frequency_by_game.png")
plt.close()

# Flask app to serve charts in Chrome
app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Gambling Prediction Dashboard</title>
</head>
<body style="font-family:sans-serif; text-align:center;">
    <h1>📊 Gambling Analysis Dashboard</h1>
    <h2>Addiction Trend by Age</h2>
    <img src="/static/age_addiction_trend.png" width="600"><br><br>

    <h2>Cluster Behavior by Income vs Bet Amount</h2>
    <img src="/static/cluster_income_bet.png" width="600"><br><br>

    <h2>Gambling Frequency by Game Type</h2>
    <img src="/static/frequency_by_game.png" width="600">
</body>
</html>
"""

@app.route("/")
def dashboard():
    return render_template_string(HTML_TEMPLATE)

def open_browser():
    webbrowser.open("http://127.0.0.1:5000")

def run_flask():
    static_path = os.path.join(os.getcwd(), "static")
    os.makedirs(static_path, exist_ok=True)

    # Helper to move and overwrite if file exists
    def move_overwrite(src, dst):
        if os.path.exists(dst):
            os.remove(dst)  # remove existing file first
        shutil.move(src, dst)

    move_overwrite("age_addiction_trend.png", os.path.join(static_path, "age_addiction_trend.png"))
    move_overwrite("cluster_income_bet.png", os.path.join(static_path, "cluster_income_bet.png"))
    move_overwrite("frequency_by_game.png", os.path.join(static_path, "frequency_by_game.png"))

    threading.Timer(1.0, open_browser).start()
    app.run()

# Run the web app
run_flask()
