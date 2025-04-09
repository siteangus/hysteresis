{\rtf1\ansi\ansicpg1252\cocoartf2822
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 import streamlit as st\
import numpy as np\
import matplotlib.pyplot as plt\
\
st.set_page_config(page_title="Preisach Hysteresis Model", layout="centered")\
st.title("\uc0\u55357 \u56521  Preisach Hysteresis Model Explorer")\
\
st.markdown("""\
This interactive model shows a simplified version of the **Preisach hysteresis model**.  \
You can adjust magnetic parameters and see how the loop changes in real time.\
""")\
\
# Sidebar controls\
H_max = st.sidebar.slider("Max Magnetic Field (H_max)", 0.1, 2.0, 1.0, step=0.1)\
num_hysterons = st.sidebar.slider("Number of Hysterons", 10, 500, 100, step=10)\
saturation_magnetisation = st.sidebar.slider("Saturation Magnetisation (Ms)", 0.1, 1.0, 1.0, step=0.1)\
\
# Generate triangular input field\
H_up = np.linspace(-H_max, H_max, 200)\
H_down = np.linspace(H_max, -H_max, 200)\
H_input = np.concatenate([H_up, H_down])\
\
# Create coercive field distribution (symmetrical)\
thresholds = np.linspace(-H_max, H_max, num_hysterons)\
states = -np.ones(num_hysterons)\
magnetisation = []\
\
# Preisach loop\
for H in H_input:\
    for i, h in enumerate(thresholds):\
        if H >= h:\
            states[i] = 1\
        elif H <= -h:\
            states[i] = -1\
    M = np.sum(states) / num_hysterons * saturation_magnetisation\
    magnetisation.append(M)\
\
# Calculate coercivity and remanence\
H_zero_crossing = H_input[np.abs(np.array(magnetisation)) == np.min(np.abs(magnetisation))][0]\
M_at_H_zero = np.interp(0, H_input, magnetisation)\
\
# Plotting\
fig, ax = plt.subplots()\
ax.plot(H_input, magnetisation, color='blue', label="Magnetisation")\
ax.axvline(x=H_zero_crossing, color='red', linestyle='--', label=f"Coercivity \uc0\u8776  \{H_zero_crossing:.2f\}")\
ax.axhline(y=M_at_H_zero, color='green', linestyle='--', label=f"Remanence \uc0\u8776  \{M_at_H_zero:.2f\}")\
ax.set_xlabel("Magnetic Field H")\
ax.set_ylabel("Magnetisation M")\
ax.set_title("Preisach Hysteresis Loop")\
ax.grid(True)\
ax.legend()\
\
st.pyplot(fig)\
\
st.markdown("### \uc0\u8505 \u65039  Notes")\
st.markdown(f"""\
- **Coercivity (Hc)**: The field where magnetisation crosses zero \uc0\u8776  `\{H_zero_crossing:.2f\}`\
- **Remanence (Mr)**: The residual magnetisation at `H = 0` \uc0\u8776  `\{M_at_H_zero:.2f\}`\
""")\
}