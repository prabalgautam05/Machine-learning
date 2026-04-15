import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time

st.set_page_config(page_title="Gradient Descent Visualizer", layout="wide")

# ---------------- TITLE ----------------
st.title("📉 Gradient Descent Visualizer (Interactive)")

# ---------------- EXPLANATION ----------------
with st.expander("📘 What is Gradient Descent? (Hinglish Explanation)"):
    st.write("""
    Gradient Descent ek optimization algorithm hai jo model ke error (loss) ko minimize karta hai.

    👉 Simple samjho:
    Tum ek pahadi ke upar khade ho aur neeche valley tak jaana hai.
    Har step pe tum slope dekhte ho aur neeche ki taraf move karte ho.

    👉 ML me:
    - Gradient = slope (error ka direction)
    - Descent = neeche jaana (error kam karna)

    Formula:
    New Weight = Old Weight - Learning Rate × Gradient

    🔥 Goal: Loss ko minimum karna
    """)

# ---------------- CONTROLS ----------------
st.sidebar.header("⚙️ Controls")

learning_rate = st.sidebar.slider("Learning Rate", 0.001, 0.1, 0.01)
epochs = st.sidebar.slider("Epochs", 10, 200, 50)
speed = st.sidebar.slider("Animation Speed", 0.01, 0.5, 0.1)

# ---------------- DATA ----------------
X = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 6, 8, 10])

# ---------------- INIT ----------------
m = 0
b = 0
n = len(X)

plot_placeholder = st.empty()
losses = []

# ---------------- ANIMATION ----------------
st.subheader("🎬 Training Animation")

for i in range(epochs):
    y_pred = m * X + b
    loss = np.mean((y - y_pred) ** 2)
    losses.append(loss)

    dm = (-2/n) * np.sum(X * (y - y_pred))
    db = (-2/n) * np.sum(y - y_pred)

    m -= learning_rate * dm
    b -= learning_rate * db

    fig, ax = plt.subplots()
    ax.scatter(X, y)
    ax.plot(X, m*X + b)
    ax.set_title(f"Epoch {i+1} | Loss: {loss:.4f}")

    plot_placeholder.pyplot(fig)
    time.sleep(speed)

# ---------------- RESULTS ----------------
st.success("Training Complete ✅")

col1, col2 = st.columns(2)
col1.metric("Final Slope (m)", f"{m:.4f}")
col2.metric("Final Intercept (b)", f"{b:.4f}")

# ---------------- LOSS GRAPH ----------------
st.subheader("📊 Loss vs Epochs")

fig2, ax2 = plt.subplots()
ax2.plot(losses)
ax2.set_xlabel("Epochs")
ax2.set_ylabel("Loss")
ax2.set_title("Loss Decreasing Over Time")

st.pyplot(fig2)

# ---------------- INTERVIEW NOTES ----------------
with st.expander("💡 Interview Explanation"):
    st.write("""
    Gradient Descent ek iterative optimization algorithm hai jo loss function ko minimize karta hai.

    Key points:
    - Direction: negative gradient
    - Step size: learning rate
    - Goal: global minimum (ya local minimum)

    Types:
    - Batch GD
    - Stochastic GD
    - Mini-batch GD
    """)
