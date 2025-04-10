import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time

st.set_page_config(page_title="Preisach Hysteresis Model Explorer", layout="centered")
st.title("Preisach Hysteresis Model Explorer")
st.markdown("""
This interactive model shows a simplified version of the **Preisach hysteresis model**. Adjust parameters on the sidebar and optionally animate the input waveform.
""")

# Sidebar controls
H_max = st.sidebar.slider("Max Magnetic Field (H_max)", 0.1, 2.0, 2.0, step=0.01)
num_hysterons = st.sidebar.slider("Number of Hysterons", 10, 500, 480, step=10)
saturation_magnetisation = st.sidebar.slider("Saturation Magnetisation (Ms)", 0.1, 2.0, 1.0, step=0.01)
coercive_mean = st.sidebar.slider("Mean Coercive Field", 0.01, 1.0, 0.93, step=0.01)
coercive_spread = st.sidebar.slider("Spread of Coercive Field", 0.01, 1.0, 0.39, step=0.01)
animate = st.sidebar.button("▶️ Animate Input Waveform")

# Generate a more detailed input field waveform for smoother curves
num_points = 400
H_up = np.linspace(-H_max, H_max, num_points)
H_down = np.linspace(H_max, -H_max, num_points)
H_cycle = np.concatenate([H_up, H_down])

# Run multiple cycles to reach steady state (major loop)
num_cycles = 2
H_input = np.tile(H_cycle, num_cycles)

# Generate the Preisach distribution of hysterons with appropriate weighting
# Create a more realistic distribution for magnetic materials
np.random.seed(42)  # Fixed seed for reproducibility

# Instead of simple normal distribution, use a distribution that ensures positive coercivity
# and provides better control over the shape of the hysteresis loop
def generate_preisach_distribution(num_hysterons, mean_coercivity, spread, h_max):
    # Generate coercive fields with a clipped Gaussian distribution
    coercive_fields = np.abs(np.random.normal(loc=mean_coercivity, 
                                           scale=spread, 
                                           size=num_hysterons))
    # Ensure values are within reasonable range
    coercive_fields = np.clip(coercive_fields, 0.01, h_max * 0.95)
    
    # Create alpha and beta values (switching thresholds)
    alphas = coercive_fields
    betas = -coercive_fields
    
    # Create weights based on distance from the mean
    weights = np.exp(-0.5 * ((coercive_fields - mean_coercivity) / spread) ** 2)
    weights = weights / np.sum(weights)  # Normalize weights
    
    return alphas, betas, weights

# Generate the hysteron distribution
alphas, betas, weights = generate_preisach_distribution(num_hysterons, 
                                                      coercive_mean, 
                                                      coercive_spread, 
                                                      H_max)

# Initialize hysteron states
# Use None initial state to properly initialize based on the first field value
states = None

# Function to compute magnetization
def compute_magnetization(H, prev_states=None):
    if prev_states is None:
        # Initial states are set based on the first field value
        # If first H is negative, initialize all to -1, otherwise to +1
        prev_states = np.ones(num_hysterons) * (-1 if H < 0 else 1)
    
    # Copy the previous states
    states = prev_states.copy()
    
    # Update states based on input field and hysteron thresholds
    for i in range(num_hysterons):
        if H >= alphas[i]:
            states[i] = 1
        elif H <= betas[i]:
            states[i] = -1
    
    # Calculate weighted magnetization
    M = np.sum(states * weights) * saturation_magnetisation
    
    # Apply saturation limits
    M = np.clip(M, -saturation_magnetisation, saturation_magnetisation)
    
    return M, states

# Compute magnetization for the input waveform
magnetisation = []
states = None

if animate:
    # Set up progress display
    progress_bar = st.progress(0)
    chart = st.line_chart()
    
    for i, H in enumerate(H_input):
        # Compute magnetization and update states
        M, states = compute_magnetization(H, states)
        magnetisation.append(M)
        
        # Only display the final cycle for animation
        if i >= len(H_input) - 2 * num_points:
            chart_index = i - (len(H_input) - 2 * num_points)
            chart.add_rows({"Magnetisation": [M]})
        
        # Update progress
        progress_bar.progress(i / len(H_input))
        time.sleep(0.01)
else:
    # Compute without animation
    for H in H_input:
        M, states = compute_magnetization(H, states)
        magnetisation.append(M)

# Only use the final cycle for analysis and display
start_idx = len(H_input) - 2 * num_points
final_H = H_input[start_idx:]
final_M = magnetisation[start_idx:]

# Calculate magnetic properties more accurately
# Find zero-crossing indices in the H field (where H changes sign)
zero_crossings = np.where(np.diff(np.signbit(final_H)))[0]

# Calculate coercivity
# When M crosses zero from positive to negative and vice versa
M_zero_crossings = np.where(np.diff(np.signbit(final_M)))[0]

if len(M_zero_crossings) >= 2:
    # Calculate the interpolated H values where M crosses zero
    pos_to_neg_idx = M_zero_crossings[0]
    neg_to_pos_idx = M_zero_crossings[1]
    
    # Linear interpolation for more accurate coercivity
    h1, h2 = final_H[pos_to_neg_idx], final_H[pos_to_neg_idx + 1]
    m1, m2 = final_M[pos_to_neg_idx], final_M[pos_to_neg_idx + 1]
    coercivity_pos = h1 - m1 * (h2 - h1) / (m2 - m1)
    
    h1, h2 = final_H[neg_to_pos_idx], final_H[neg_to_pos_idx + 1]
    m1, m2 = final_M[neg_to_pos_idx], final_M[neg_to_pos_idx + 1]
    coercivity_neg = h1 - m1 * (h2 - h1) / (m2 - m1)
    
    # Average the absolute values for more accurate coercivity
    coercivity = (abs(coercivity_pos) + abs(coercivity_neg)) / 2
else:
    coercivity = None

# Calculate remanence - magnetization when H = 0
if len(zero_crossings) >= 2:
    # Indices where H crosses zero
    pos_to_neg_idx = zero_crossings[0]  # H goes from positive to negative
    neg_to_pos_idx = zero_crossings[1] if len(zero_crossings) > 1 else None  # H goes from negative to positive
    
    # Linear interpolation for more accurate remanence
    if pos_to_neg_idx is not None:
        h1, h2 = final_H[pos_to_neg_idx], final_H[pos_to_neg_idx + 1]
        m1, m2 = final_M[pos_to_neg_idx], final_M[pos_to_neg_idx + 1]
        # Interpolate to find M at H=0
        remanence_pos = m1 + (0 - h1) * (m2 - m1) / (h2 - h1)
    else:
        remanence_pos = None
        
    if neg_to_pos_idx is not None:
        h1, h2 = final_H[neg_to_pos_idx], final_H[neg_to_pos_idx + 1]
        m1, m2 = final_M[neg_to_pos_idx], final_M[neg_to_pos_idx + 1]
        # Interpolate to find M at H=0
        remanence_neg = m1 + (0 - h1) * (m2 - m1) / (h2 - h1)
    else:
        remanence_neg = None
    
    # Use both remanence values for more accurate calculation
    if remanence_pos is not None and remanence_neg is not None:
        remanence = (remanence_pos - remanence_neg) / 2
    elif remanence_pos is not None:
        remanence = remanence_pos
    elif remanence_neg is not None:
        remanence = remanence_neg
    else:
        remanence = None
else:
    remanence = None

# Plotting
fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(final_H, final_M, color='blue', linewidth=2, label="Magnetisation")

# Add coercivity and remanence lines
if coercivity is not None:
    ax.axvline(x=coercivity, color='red', linestyle='--', label=f"Coercivity ≈ {coercivity:.2f}")
if remanence is not None:
    ax.axhline(y=remanence, color='green', linestyle='--', label=f"Remanence ≈ {remanence:.2f}")

ax.set_xlabel("Magnetic Field H", fontsize=12)
ax.set_ylabel("Magnetisation M", fontsize=12)
ax.set_title("Preisach Hysteresis Loop", fontsize=14)
ax.grid(True)
ax.legend(fontsize=10)
ax.set_xlim(-H_max, H_max)
ax.set_ylim(-saturation_magnetisation, saturation_magnetisation)

st.pyplot(fig)

# Display calculated properties
st.markdown("### Magnetic Properties")
st.markdown(f"""
- **Coercivity (Hc)**: The magnetic field required to demagnetize the material ≈ `{coercivity:.2f}` (if calculated)
- **Remanence (Mr)**: The magnetization that remains when the field is removed ≈ `{remanence:.2f}` (if calculated)
- **Saturation Magnetisation (Ms)**: The maximum possible magnetization ≈ `{saturation_magnetisation:.2f}`
- **Squareness Ratio (Mr/Ms)**: The ratio of remanence to saturation ≈ `{abs(remanence/saturation_magnetisation):.2f}` (if calculated)
""")

# Add model explanation
with st.expander("About the Preisach Model"):
    st.markdown("""
    ### Preisach Hysteresis Model
    
    The Preisach model represents a magnetic material as a collection of elementary magnetic units called "hysterons." 
    Each hysteron is a simple bistable unit with two switching thresholds (α and β).
    
    Key components of this implementation:
    
    1. **Hysterons**: Each hysteron switches up at field H ≥ α and down at field H ≤ β.
    2. **Distribution**: The distribution of hysterons creates the shape of the hysteresis loop.
    3. **Weighting**: Hysterons are weighted to match real material behavior.
    4. **Multiple Cycles**: The model runs multiple cycles to reach the stable major loop.
    
    This model can be adjusted to represent different magnetic materials by changing:
    - The mean coercive field (affects loop width)
    - The spread of coercive fields (affects loop squareness)
    - The saturation magnetization (affects loop height)
    """)
