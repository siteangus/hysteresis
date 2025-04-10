import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time

st.set_page_config(page_title="Preisach Hysteresis Model", layout="centered")
st.title(" Preisach Hysteresis Model ")

st.markdown("""
This interactive model shows a simplified version of the **Preisach hysteresis model**.
Adjust parameters on the sidebar and optionally animate the input waveform.
""")

# Sidebar controls
H_max = st.sidebar.slider("Max Magnetic Field (H_max)", 0.1, 2.0, 1.0, step=0.1)
num_hysterons = st.sidebar.slider("Number of Hysterons", 10, 500, 100, step=10)
saturation_magnetisation = st.sidebar.slider("Saturation Magnetisation (Ms)", 0.1, 2.0, 1.0, step=0.1)
coercive_mean = st.sidebar.slider("Mean Coercive Field", 0.01, 1.0, 0.3, step=0.01)
coercive_spread = st.sidebar.slider("Spread of Coercive Field", 0.01, 1.0, 0.2, step=0.01)
animate = st.sidebar.button("▶️ Animate Input Waveform")

# Generate triangular input field
H_up = np.linspace(-H_max, H_max, 300)
H_down = np.linspace(H_max, -H_max, 300)
H_input = np.concatenate([H_up, H_down])

# Preisach model hysterons (alphas, betas)
np.random.seed(0)
coercive_fields = np.random.normal(loc=coercive_mean, scale=coercive_spread, size=num_hysterons)
coercive_fields = np.clip(coercive_fields, 0.01, H_max)
alphas = coercive_fields
betas = -coercive_fields

states = -np.ones(num_hysterons)

# Clip the magnetization to saturation levels
saturation_magnetisation_pos = saturation_magnetisation
saturation_magnetisation_neg = -saturation_magnetisation

magnetisation = []  # To store magnetization values

if animate:
    chart = st.line_chart()
    temp_magnetisation = []
    for H in H_input:
        for i in range(num_hysterons):
            if H >= alphas[i]:
                states[i] = 1
            elif H <= betas[i]:
                states[i] = -1
        M = np.sum(states) / num_hysterons * saturation_magnetisation
        # Clip magnetization to saturation limits
        M = np.clip(M, saturation_magnetisation_neg, saturation_magnetisation_pos)
        temp_magnetisation.append(M)
        chart.add_rows({"Magnetisation": [M]})
        time.sleep(0.01)
    magnetisation = temp_magnetisation
else:
    for H in H_input:
        for i in range(num_hysterons):
            if H >= alphas[i]:
                states[i] = 1
            elif H <= betas[i]:
                states[i] = -1
        M = np.sum(states) / num_hysterons * saturation_magnetisation
        # Clip magnetization to saturation limits
        M = np.clip(M, saturation_magnetisation_neg, saturation_magnetisation_pos)
        magnetisation.append(M)

# Calculate coercivity and remanence
magnetisation_array = np.array(magnetisation)

# Coercivity - where magnetisation crosses zero
H_zero_crossing = H_input[np.argmin(np.abs(magnetisation_array))]

# Remanence - magnetisation at H = 0
M_at_H_zero = magnetisation_array[np.abs(H_input).argmin()]

# Plotting
fig, ax = plt.subplots()
ax.plot(H_input, magnetisation, color='blue', label="Magnetisation")
ax.axvline(x=H_zero_crossing, color='red', linestyle='--', label=f"Coercivity ≈ {H_zero_crossing:.2f}")
ax.axhline(y=M_at_H_zero, color='green', linestyle='--', label=f"Remanence ≈ {M_at_H_zero:.2f}")

ax.set_xlabel("Magnetic Field H")
ax.set_ylabel("Magnetisation M")
ax.set_title("Preisach Hysteresis Loop")
ax.grid(True)
ax.legend()

st.pyplot(fig)

st.markdown("### Notes")
st.markdown(f"""
- **Coercivity (Hc)**: The field where magnetisation crosses zero ≈ `{H_zero_crossing:.2f}`
- **Remanence (Mr)**: The residual magnetisation at `H = 0` ≈ `{M_at_H_zero:.2f}`
""")
