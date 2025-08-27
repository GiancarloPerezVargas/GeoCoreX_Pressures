import streamlit as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# ================== CONFIGURACIÓN ==================
st.set_page_config(
    page_title="GeoCoreX | Análisis Geotécnico",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================== ESTILOS ==================
st.markdown("""
<style>
    .main-header { background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%); padding: 1.5rem;
                    border-radius: 10px; color: white; margin-bottom: 2rem; text-align: center;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);}
    .metric-container {background: #f8f9fa; padding:1rem; border-radius:8px; border-left: 4px solid #2a5298; margin:0.5rem 0;}
</style>
""", unsafe_allow_html=True)

# ================== ENCABEZADO ==================
st.markdown("""
<div class="main-header">
<h1>⚡ GeoCoreX | Análisis de Presiones Laterales</h1>
<p>Automatización de cálculos de presiones y empuje total considerando nivel freático</p>
</div>
""", unsafe_allow_html=True)

# ================== FUNCIONES ==================
def create_metric_card(title, value, unit, icon):
    st.markdown(f"<div class='metric-container'><h4>{icon} {title}</h4><h2>{value} {unit}</h2></div>", unsafe_allow_html=True)

def calculate_k0_jacky(phi_deg):
    return 1 - np.sin(np.deg2rad(phi_deg))

def calculate_pressure_distribution(Hs, gammas, phis, q_kN, z_water_top, gamma_w):
    K0s = [calculate_k0_jacky(phi) for phi in phis]
    z_top_list, z_bot_list, p_top_list, p_bot_list, A_list, zc_list = [], [], [], [], [], []
    z_current = 0
    for H, gamma, K0 in zip(Hs, gammas, K0s):
        z_top, z_bot = z_current, z_current + H
        p_top = K0 * q_kN
        dry_h = max(0, min(z_bot, z_water_top) - z_top)
        sub_h = H - dry_h
        p_bot = p_top + K0 * (gamma * dry_h + (gamma - gamma_w) * sub_h)
        A = 0.5 * (p_top + p_bot) * H
        zc = z_top + H * (p_top + 2 * p_bot) / (3 * (p_top + p_bot)) if (p_top + p_bot) != 0 else z_top + H/2
        z_top_list.append(z_top)
        z_bot_list.append(z_bot)
        p_top_list.append(p_top)
        p_bot_list.append(p_bot)
        A_list.append(A)
        zc_list.append(zc)
        z_current = z_bot
    h_water = max(0, sum(Hs) - z_water_top)
    A_water = 0.5 * gamma_w * h_water**2
    zc_water = z_water_top + 2/3 * h_water if h_water > 0 else 0
    P_total = sum(A_list) + A_water
    z_bar = (sum([a*z for a,z in zip(A_list, zc_list)]) + A_water*zc_water) / P_total if P_total > 0 else 0
    return {
        'K0s': K0s, 'z_top_list': z_top_list, 'z_bot_list': z_bot_list,
        'p_top_list': p_top_list, 'p_bot_list': p_bot_list,
        'A_list': A_list, 'zc_list': zc_list,
        'h_water': h_water, 'A_water': A_water, 'zc_water': zc_water,
        'P_total': P_total, 'z_bar': z_bar
    }

# ================== SIDEBAR ==================
st.sidebar.header("🎛️ Parámetros de Entrada")

q_unit = st.sidebar.selectbox("Unidad sobrecarga q:", ["kN/m²", "t/m²"])
q_input = st.sidebar.number_input("Valor de q:", value=147.15 if q_unit=="kN/m²" else 15.0, step=0.01)
q_kN = q_input if q_unit=="kN/m²" else q_input*9.81

n = st.sidebar.number_input("Número de estratos:", min_value=1, max_value=4, value=2)

default_values = [
    {'H': 2.4, 'gamma': 16.5, 'phi': 25},
    {'H': 3.6, 'gamma': 19.3, 'phi': 30},
    {'H': 2.0, 'gamma': 18.0, 'phi': 28},
    {'H': 2.5, 'gamma': 20.0, 'phi': 32}
]

Hs, gammas, phis = [], [], []
for i in range(int(n)):
    expander = st.sidebar.expander(f"Estrato {i+1}", expanded=True)
    defaults = default_values[i] if i < len(default_values) else default_values[0]
    H = expander.number_input(f"H (m):", value=defaults['H'], step=0.01, key=f"H{i}")
    gamma = expander.number_input(f"γ (kN/m³):", value=defaults['gamma'], step=0.01, key=f"gamma{i}")
    phi = expander.number_input(f"φ (°):", value=defaults['phi'], step=0.1, key=f"phi{i}")
    Hs.append(H)
    gammas.append(gamma)
    phis.append(phi)

z_water_top = st.sidebar.number_input("Profundidad NAF (m):", value=2.4, step=0.01)
gamma_w = st.sidebar.number_input("γw (kN/m³):", value=9.81, step=0.01)

# ================== CÁLCULOS ==================
results = calculate_pressure_distribution(Hs, gammas, phis, q_kN, z_water_top, gamma_w)

# ================== RESULTADOS ==================
st.subheader("📈 Resultados")
col1, col2, col3 = st.columns(3)
create_metric_card("Empuje Total (t/m)", f"{results['P_total']:.2f}", "t/m", "⚡")
create_metric_card("Empuje Total (kN/m)", f"{results['P_total']*9.81:.1f}", "kN/m", "🔧")
create_metric_card("Centro de Presiones", f"{results['z_bar']:.3f}", "m", "🎯")

# ================== DIAGRAMA ==================
st.subheader("📊 Diagrama de Presiones Laterales")
fig, ax = plt.subplots(figsize=(10,6))
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
for i in range(int(n)):
    z_vals = [results['z_top_list'][i], results['z_bot_list'][i]]
    p_vals = [results['p_top_list'][i]*9.81, results['p_bot_list'][i]*9.81]
    ax.fill_betweenx(z_vals, 0, p_vals, alpha=0.3, color=colors[i % len(colors)])
    ax.plot(p_vals, z_vals, linewidth=2.5, color=colors[i % len(colors)])
ax.axhline(results['z_bar'], color='red', linestyle='--', linewidth=2, label="Centro de Presiones")
ax.set_xlabel("Presión (kN/m)")
ax.set_ylabel("Altura (m)")
ax.invert_yaxis()
ax.grid(True)
ax.legend()
st.pyplot(fig)