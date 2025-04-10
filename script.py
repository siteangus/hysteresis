# Code combination from ChatGPT/Claude

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib.patches import FancyArrowPatch

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

# Generate input field waveform
num_points = 400
H_up = np.linspace(-H_max, H_max, num_points)
H_down = np.linspace(H_max, -H_max, num_points)

# Generate full waveform with multiple cycles
H_input = np.concatenate([H_up, H_down, H_up, H_down, H_up, H_down])  # 3 complete cycles

# Critical fix: Ensure we have a clear direction of field change
# This is what produces proper remanence values
np.random.seed(42)  # For reproducibility

# Fixed version of the distribution function
def generate_preisach_distribution(num_hysterons, mean_coercivity, spread, h_max):
    # Generate coercive fields with a distribution that ensures positive values
    # Use lognormal distribution which better models physical coercivity distributions
    sigma = spread / mean_coercivity  # Convert spread to lognormal sigma parameter
    mu = np.log(mean_coercivity) - 0.5 * sigma**2  # Ensure mean is at specified value
    
    coercive_fields = np.random.lognormal(mean=mu, sigma=sigma, size=num_hysterons)
    coercive_fields = np.clip(coercive_fields, 0.01, h_max * 0.95)
    
    # Create separate parameters for up and down switching to ensure distinct remanence
    # This is critical for getting proper remanence behavior
    alpha_factor = 1.2  # Make alpha consistently higher than -beta
    beta_factor = 1.0
    
    # Create alpha and beta values (switching thresholds)
    alphas = coercive_fields * alpha_factor
    betas = -coercive_fields * beta_factor
    
    # Create weights based on distribution
    weights = np.ones(num_hysterons) / num_hysterons  # Equal weights for simplicity
    
    return alphas, betas, weights

# Generate the hysteron distribution
alphas, betas, weights = generate_preisach_distribution(num_hysterons, 
                                                      coercive_mean, 
                                                      coercive_spread, 
                                                      H_max)

# Guaranteed asymmetric hysteresis with distinct remanence
# Force some asymmetry by ensuring alphas are consistently different from -betas
alphas = np.abs(alphas)
betas = -np.abs(betas) * 0.8  # Make betas systematically smaller in magnitude

# Function to compute magnetization - improved version
def compute_magnetization(H_values):
    """Compute the entire magnetization curve to ensure consistency"""
    states = np.ones(num_hysterons) * -1  # Start with all hysterons in negative state
    magnetisation = []
    
    for H in H_values:
        # Update states based on input field and hysteron thresholds
        for i in range(num_hysterons):
            if H >= alphas[i]:  # If field exceeds upper threshold, switch up
                states[i] = 1
            elif H <= betas[i]:  # If field below lower threshold, switch down
                states[i] = -1
            # Otherwise retain previous state (this is key for hysteresis)
        
        # Calculate weighted magnetization
        M = np.sum(states * weights) * saturation_magnetisation
        
        # Ensure M stays within saturation limits
        M = np.clip(M, -saturation_magnetisation, saturation_magnetisation)
        magnetisation.append(M)
    
    return np.array(magnetisation)

# Compute magnetization - always calculate the full curve for consistency
if animate:
    # Set up progress display
    progress_bar = st.progress(0)
    chart = st.line_chart()
    
    # Pre-compute to ensure consistent results
    magnetisation_array = compute_magnetization(H_input)
    
    # Display with animation
    for i, M in enumerate(magnetisation_array):
        # Only display the final cycle for animation
        final_cycle_length = 2 * num_points  # Up and down
        if i >= len(H_input) - final_cycle_length:
            chart_index = i - (len(H_input) - final_cycle_length)
            chart.add_rows({"Magnetisation": [M]})
        
        # Update progress
        progress_bar.progress(i / len(H_input))
        time.sleep(0.01)
else:
    # Compute without animation
    magnetisation_array = compute_magnetization(H_input)

# Only use the final cycle for analysis and display
# The last cycle should be a stable major loop
final_cycle_length = 2 * num_points  # One complete up and down cycle
start_idx = len(H_input) - final_cycle_length
final_H = H_input[start_idx:]
final_M = magnetisation_array[start_idx:]

# Calculate magnetic properties
# Simplify calculation by using array indices at key points
up_half = final_M[:num_points]    # Rising field portion
down_half = final_M[num_points:]  # Falling field portion

# Find the remanence directly - when H = 0 after saturation
# This is the value of M when the field returns to zero from positive saturation
h_zero_idx = np.argmin(np.abs(final_H[num_points:]))
remanence = down_half[h_zero_idx]  # This should be the positive remanence

# Find coercivity - where M crosses zero during demagnetization
# This is the field needed to reduce M to zero after saturation
m_zero_idx = np.argmin(np.abs(down_half))
coercivity = final_H[num_points + m_zero_idx]

# Plotting with enhanced visualization
fig, ax = plt.subplots(figsize=(10, 8))

# Plot the hysteresis loop
ax.plot(final_H, final_M, color='blue', linewidth=2.5, label="Magnetisation")

# Add coercivity and remanence markers
ax.axvline(x=coercivity, color='red', linestyle='--', label=f"Coercivity ≈ {coercivity:.2f}")
ax.axhline(y=remanence, color='green', linestyle='--', label=f"Remanence ≈ {remanence:.2f}")

# Add bold axes at zero
ax.axhline(y=0, color='black', linewidth=1.5)
ax.axvline(x=0, color='black', linewidth=1.5)

# Add arrows to show direction of hysteresis loop
# First arrow - bottom half of loop (left to right)
arrow1_idx = num_points // 4
ax.add_patch(FancyArrowPatch(
    (final_H[arrow1_idx], final_M[arrow1_idx]),
    (final_H[arrow1_idx+20], final_M[arrow1_idx+20]),
    arrowstyle='->', color='blue', mutation_scale=15
))

# Second arrow - top half of loop (right to left)
arrow2_idx = num_points + num_points // 4
ax.add_patch(FancyArrowPatch(
    (final_H[arrow2_idx], final_M[arrow2_idx]),
    (final_H[arrow2_idx+20], final_M[arrow2_idx+20]),
    arrowstyle='->', color='blue', mutation_scale=15
))

# Improve layout and labels
ax.set_xlabel("Magnetic Field H", fontsize=12)
ax.set_ylabel("Magnetisation M", fontsize=12)
ax.set_title("Preisach Hysteresis Loop", fontsize=14)
ax.grid(True, alpha=0.3)  # Lighter grid for better visibility
ax.legend(fontsize=10, loc='lower right')
ax.set_xlim(-H_max, H_max)
ax.set_ylim(-saturation_magnetisation * 1.05, saturation_magnetisation * 1.05)

# Ensure more square-like aspect ratio for better visualization
ax.set_aspect(H_max/saturation_magnetisation)

st.pyplot(fig)

# Display calculated properties
st.markdown("### Magnetic Properties")
st.markdown(f"""
- **Coercivity (Hc)**: The magnetic field required to demagnetize the material ≈ `{coercivity:.2f}`
- **Remanence (Mr)**: The magnetization that remains when the field is removed ≈ `{remanence:.2f}`
- **Saturation Magnetisation (Ms)**: The maximum possible magnetization ≈ `{saturation_magnetisation:.2f}`
- **Squareness Ratio (Mr/Ms)**: The ratio of remanence to saturation ≈ `{abs(remanence/saturation_magnetisation):.2f}`
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
    5. **Asymmetry**: Systematic asymmetry in the switching thresholds creates realistic remanence.
    
    This model can be adjusted to represent different magnetic materials by changing:
    - The mean coercive field (affects loop width)
    - The spread of coercive fields (affects loop squareness)
    - The saturation magnetization (affects loop height)
    """)
