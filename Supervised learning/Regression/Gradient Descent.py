import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time

st.title("🎬 Gradient Descent Animation (Step-by-Step)")

# Controls
learning_rate = st.slider("Learning Rate", 0.001, 0.1, 0.01)
epochs = st.slider("Epochs", 10, 200, 50)
speed = st.slider("Animation Speed (seconds)", 0.01, 0.5, 0.1)

# Sample Data
X = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 6, 8, 10])

# Initialize parameters
m = 0
b = 0
n = len(X)

# Placeholder for dynamic plot
plot_placeholder = st.empty()

losses = []

# Animation loop
for i in range(epochs):
    y_pred = m * X + b
    loss = np.mean((y - y_pred)**2)
    losses.append(loss)

    # Gradients
    dm = (-2/n) * np.sum(X * (y - y_pred))
    db = (-2/n) * np.sum(y - y_pred)

    # Update
    m -= learning_rate * dm
    b -= learning_rate * db

    # Plot current state
    fig, ax = plt.subplots()
    ax.scatter(X, y)
    ax.plot(X, m*X + b)
    ax.set_title(f"Epoch {i+1} | Loss: {loss:.4f}")

    plot_placeholder.pyplot(fig)

    time.sleep(speed)

# Final Output
st.success("Training Complete ✅")
st.write(f"Final Slope: {m:.4f}")
st.write(f"Final Intercept: {b:.4f}")

# Loss curve
fig2, ax2 = plt.subplots()
ax2.plot(losses)
ax2.set_title("Loss vs Epochs")
st.pyplot(fig2)
