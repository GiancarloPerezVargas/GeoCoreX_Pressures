import streamlit as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# ================== CONFIGURACIÓN Y ESTILOS ==================
st.set_page_config(
    page_title="GeoCoreX | Análisis Geotécnico", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado para mejorar el aspecto visual
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-container {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2a5298;
        margin: 0.5rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
        background-color: #f1f3f4;
        border-radius: 5px 5px 0px 0px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2a5298;
        color: white;
    }
    .sidebar-section {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border: 1px solid #e9ecef;
    }
</style>
""", unsafe_allow_html=True)

# ================== ENCABEZADO PRINCIPAL ==================
st.markdown("""
<div class="main-header">
    <h1>⚡ GeoCoreX | Análisis de Presiones Laterales</h1>
    <p>Cálculo automatizado mediante método geométrico con consideración de nivel freático y desfase en interfaces</p>
</div>
""", unsafe_allow_html=True)

# ================== FUNCIONES AUXILIARES ==================
def create_metric_card(title, value, unit, icon):
    """Crear tarjetas de métricas personalizadas"""
    st.markdown(f"""
    <div class="metric-container">
        <h4>{icon} {title}</h4>
        <h2>{value} {unit}</h2>
    </div>
    """, unsafe_allow_html=True)

def calculate_k0_jacky(phi_degrees):
    """Calcular K0 usando la fórmula de Jacky"""
    return 1.0 - np.sin(np.deg2rad(phi_degrees))

def calculate_pressure_distribution(Hs, gammas_kN, phis, q_kN, z_water_top, gamma_w_kN):
    """Calcular distribución de presiones por estratos"""
    # Conversiones
    gammas_t = [g / 9.81 for g in gammas_kN]
    gamma_w_t = gamma_w_kN / 9.81
    q_t = q_kN / 9.81
    
    # Cálculo de K0 por estrato
    K0s = [calculate_k0_jacky(phi) for phi in phis]
    
    # Inicialización de variables
    H_total = sum(Hs)
    n = len(Hs)
    
    # Cálculo de pesos acumulados
    cum_weight_above = []
    cum = 0.0
    for H, gamma_t in zip(Hs, gammas_t):
        cum_weight_above.append(cum)
        cum += gamma_t * H
    
    # Cálculo por estratos
    z_top_list, z_bot_list = [], []
    p_top_t_list, p_bot_t_list = [], []
    A_list_t, zc_list = [], []
    
    z_current = 0.0
    for i in range(n):
        H, gamma_t, K0 = Hs[i], gammas_t[i], K0s[i]
        ztop, zbot = z_current, z_current + H
        weight_above = cum_weight_above[i]
        
        # Presión en tope del estrato
        p_top_t = K0 * (q_t + weight_above)
        
        # Consideración del nivel freático
        dry_h = max(0.0, min(zbot, z_water_top) - ztop)
        sub_h = H - dry_h
        incr = gamma_t * dry_h + (gamma_t - gamma_w_t) * sub_h
        
        # Presión en base del estrato
        p_bot_t = p_top_t + K0 * incr
        
        # Área del diagrama trapezoidal
        A_t = 0.5 * (p_top_t + p_bot_t) * H
        
        # Centroide
        if (p_top_t + p_bot_t) != 0:
            y_local = H * (p_top_t + 2.0 * p_bot_t) / (3.0 * (p_top_t + p_bot_t))
        else:
            y_local = H / 2
        zc = ztop + y_local
        
        # Almacenar resultados
        z_top_list.append(ztop)
        z_bot_list.append(zbot)
        p_top_t_list.append(p_top_t)
        p_bot_t_list.append(p_bot_t)
        A_list_t.append(A_t)
        zc_list.append(zc)
        z_current = zbot
    
    # Cálculo de presión hidrostática
    h_water = max(0.0, H_total - z_water_top)
    A_water_t = 0.5 * gamma_w_t * h_water**2
    zc_water = z_water_top + 2.0/3.0 * h_water if h_water > 0 else 0.0
    
    # Resultados finales
    P_total_t = sum(A_list_t) + A_water_t
    M_total = sum([A * zc for A, zc in zip(A_list_t, zc_list)]) + A_water_t * zc_water
    z_bar = M_total / P_total_t if P_total_t > 0 else 0.0
    
    return {
        'K0s': K0s, 'z_top_list': z_top_list, 'z_bot_list': z_bot_list,
        'p_top_t_list': p_top_t_list, 'p_bot_t_list': p_bot_t_list,
        'A_list_t': A_list_t, 'zc_list': zc_list, 'h_water': h_water,
        'A_water_t': A_water_t, 'zc_water': zc_water, 'P_total_t': P_total_t,
        'z_bar': z_bar, 'H_total': H_total, 'gamma_w_t': gamma_w_t,
        'gamma_w_kN': gamma_w_kN
    }

# ================== BARRA LATERAL ==================
with st.sidebar:
    st.markdown("### 🎛️ Parámetros de Entrada")
    
    # Sección de sobrecarga
    st.markdown("#### 📊 Sobrecarga Superficial")
    q_unit = st.selectbox("Unidad:", ["kN/m²", "t/m²"])
    
    if q_unit == "kN/m²":
        q_kN = st.number_input("Valor de q:", value=147.15, step=0.01, format="%.2f")
    else:
        q_t_input = st.number_input("Valor de q:", value=15.00, step=0.01, format="%.2f")
        q_kN = q_t_input * 9.81
    
    st.markdown("---")
    
    # Sección de estratos
    st.markdown("#### 🏗️ Definición de Estratos")
    n = st.number_input("Número de estratos:", min_value=1, max_value=4, value=2)
    
    # Valores por defecto mejorados
    default_values = [
        {'H': 2.40, 'gamma': 16.50, 'phi': 24.78},
        {'H': 3.60, 'gamma': 19.30, 'phi': 30.00},
        {'H': 2.00, 'gamma': 18.00, 'phi': 28.00},
        {'H': 2.50, 'gamma': 20.00, 'phi': 32.00}
    ]
    
    Hs, gammas_kN, phis = [], [], []
    
    for i in range(int(n)):
        with st.expander(f"📋 Estrato {i+1}", expanded=True):
            defaults = default_values[i] if i < len(default_values) else default_values[0]
            
            col1, col2 = st.columns(2)
            with col1:
                H = st.number_input(f"H (m):", value=defaults['H'], step=0.01, key=f"H{i}")
                phi = st.number_input(f"φ (°):", value=defaults['phi'], step=0.1, key=f"phi{i}")
            with col2:
                gamma = st.number_input(f"γ (kN/m³):", value=defaults['gamma'], step=0.01, key=f"gamma{i}")
                
            # Mostrar K0 calculado
            k0_calc = calculate_k0_jacky(phi)
            st.info(f"K₀ = {k0_calc:.3f}")
            
            Hs.append(H)
            gammas_kN.append(gamma)
            phis.append(phi)
    
    st.markdown("---")
    
    # Sección de nivel freático
    st.markdown("#### 💧 Condiciones Hidrogeológicas")
    z_water_top = st.number_input("Profundidad NAF (m):", value=2.40, step=0.01)
    gamma_w_kN = st.number_input("γw (kN/m³):", value=9.81, step=0.01)

# ================== CÁLCULOS PRINCIPALES ==================
results = calculate_pressure_distribution(
    Hs, gammas_kN, phis, q_kN, z_water_top, gamma_w_kN
)

# ================== RESULTADOS PRINCIPALES ==================
st.markdown("### 📈 Resultados del Análisis")

col1, col2, col3 = st.columns(3)
with col1:
    create_metric_card(
        "Empuje Total", 
        f"{results['P_total_t']:.2f}", 
        "t/m",
        "⚡"
    )

with col2:
    create_metric_card(
        "Empuje Total", 
        f"{results['P_total_t'] * 9.81:.1f}", 
        "kN/m",
        "🔧"
    )

with col3:
    create_metric_card(
        "Centro de Presiones", 
        f"{results['z_bar']:.3f}", 
        "m",
        "🎯"
    )

# ================== PESTAÑAS DE CONTENIDO ==================
tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Tabla Resumen", 
    "📊 Diagrama", 
    "📖 Metodología", 
    "⭕ Círculos de Mohr"
])

# PESTAÑA 1: Tabla de resultados
with tab1:
    st.markdown("#### 📋 Resumen Detallado por Estrato")
    
    # Preparar datos para la tabla
    table_data = []
    for i in range(int(n)):
        table_data.append({
            "Estrato": f"E-{i+1}",
            "Espesor (m)": f"{Hs[i]:.2f}",
            "γ (kN/m³)": f"{gammas_kN[i]:.1f}",
            "φ (°)": f"{phis[i]:.1f}",
            "K₀": f"{results['K0s'][i]:.3f}",
            "P_sup (kN/m)": f"{results['p_top_t_list'][i] * 9.81:.1f}",
            "P_inf (kN/m)": f"{results['p_bot_t_list'][i] * 9.81:.1f}",
            "Área (kN·m/m)": f"{results['A_list_t'][i] * 9.81:.1f}",
            "Centroide (m)": f"{results['zc_list'][i]:.2f}"
        })
    
    # Agregar fila de agua si existe
    if results['h_water'] > 0:
        table_data.append({
            "Estrato": "AGUA",
            "Espesor (m)": f"{results['h_water']:.2f}",
            "γ (kN/m³)": f"{results['gamma_w_kN']:.1f}",
            "φ (°)": "-",
            "K₀": "-",
            "P_sup (kN/m)": "0.0",
            "P_inf (kN/m)": f"{results['gamma_w_kN'] * results['h_water']:.1f}",
            "Área (kN·m/m)": f"{results['A_water_t'] * 9.81:.1f}",
            "Centroide (m)": f"{results['zc_water']:.2f}"
        })
    
    df = pd.DataFrame(table_data)
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Botón de descarga
    csv_data = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇️ Exportar a CSV",
        data=csv_data,
        file_name=f"analisis_presiones_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
        help="Descargar tabla de resultados en formato CSV"
    )

# PESTAÑA 2: Gráficas
with tab2:
    st.markdown("#### 📊 Diagrama de Distribución de Presiones")
    
    # Crear figura
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Configurar colores
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    
    # Dibujar estratos
    for i in range(int(n)):
        z_vals = [results['z_top_list'][i], results['z_bot_list'][i]]
        p_vals = [results['p_top_t_list'][i] * 9.81, results['p_bot_t_list'][i] * 9.81]
        
        # Área rellena
        ax.fill_betweenx(z_vals, 0, p_vals, alpha=0.3, color=colors[i % len(colors)], 
                        label=f'Estrato {i+1}')
        
        # Línea del diagrama
        ax.plot(p_vals, z_vals, linewidth=2.5, color=colors[i % len(colors)])
        
        # Etiquetas
        mid_z = (z_vals[0] + z_vals[1]) / 2
        mid_p = (p_vals[0] + p_vals[1]) / 2
        ax.annotate(f'E{i+1}', xy=(mid_p, mid_z), xytext=(mid_p + 20, mid_z),
                   fontsize=10, fontweight='bold', ha='center')
    
    # Dibujar presión de agua
    if results['h_water'] > 0:
        z_water = [z_water_top, z_water_top + results['h_water']]
        p_water = [0.0, results['gamma_w_kN'] * results['h_water']]
        
        ax.fill_betweenx(z_water, 0, p_water, alpha=0.2, color='lightblue', 
                        label='Presión hidrostática')
        ax.plot(p_water, z_water, linestyle='--', color='blue', linewidth=2)
    
    # Líneas de referencia
    for i, zt in enumerate(results['z_top_list'][1:], 1):
        ax.axhline(zt, linestyle=":", color="gray", alpha=0.7, linewidth=1)
        ax.text(ax.get_xlim()[1] * 0.05, zt, f'Interface E{i}-E{i+1}', 
               va='bottom', fontsize=8, color='gray')
    
    # NAF
    if 0.0 <= z_water_top <= results['H_total']:
        ax.axhline(z_water_top, linestyle="-", color="cyan", linewidth=2)
        ax.text(ax.get_xlim()[1] * 0.7, z_water_top, 'NAF', 
               va='bottom', fontsize=10, fontweight='bold', color='cyan')
    
    # Centro de presiones
    ax.axhline(results['z_bar'], linestyle="-", color="red", linewidth=2, alpha=0.8)
    ax.text(ax.get_xlim()[1] * 0.7, results['z_bar'], 
           f"Centro: {results['z_bar']:.2f} m", 
           va='top', fontsize=10, fontweight='bold', color='red',
           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="red", alpha=0.8))
    
    # Configuración final
    ax.set_xlabel("Presión Lateral (kN/m)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Profundidad (m)", fontsize=12, fontweight='bold')
    ax.set_title("Distribución de Presiones Laterales", fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.invert_yaxis()
    
    # Mejorar apariencia
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    
    st.pyplot(fig)

# PESTAÑA 3: Metodología
with tab3:
    st.markdown("#### 📖 Fundamentos Teóricos")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### 🧮 Fórmulas Principales")
        st.latex(r"K_0 = 1 - \sin(\varphi)")
        st.latex(r"P_{total} = \sum_{i=1}^{n} A_i + A_{agua}")
        st.latex(r"\bar{z} = \frac{\sum A_i \cdot z_i}{P_{total}}")
        st.latex(r"A_i = \frac{1}{2}(p_{sup} + p_{inf}) \cdot H_i")
    
    with col2:
        st.markdown("##### ⚙️ Consideraciones de Cálculo")
        st.markdown("""
        - **Método de Jacky**: Para calcular el valor del coeficiente de empuje en reposo k0
        - **Peso volumétrico**: Natural sobre NAF, sumergido bajo NAF
        - **Presión hidrostática**: Considerada independientemente
        - **Distribución**: Lineal por estrato (trapezoidal)
        - **Centro de presiones**: Método de momentos estáticos
        """)
    
    st.markdown("##### 🎯 Aplicabilidad del Método")
    st.info("""
    **Válido para:**
    - Suelos granulares y cohesivos normalmente consolidados
    - Muros de retención en condiciones estáticas
    - Análisis preliminar de empujes laterales
    
    **Limitaciones:**
    - No considera sobreconsolidación (OCR > 1)
    - No incluye efectos de cohesión en K₀
    - Aplicable solo a condiciones de reposo
    """)

# PESTAÑA 4: Círculos de Mohr
with tab4:
    st.markdown("#### ⭕ Análisis de Estados de Esfuerzo - Círculos de Mohr")
    
    st.markdown("##### 📊 Ingreso de Estados de Esfuerzo")
    st.caption("Ingrese hasta 4 pares de esfuerzos principales en kg/cm²")
    
    # Crear columnas para entrada de datos
    cols = st.columns(2)
    mohr_data = []
    
    for i in range(4):
        with st.container():
            subcols = st.columns([1, 2, 2, 1])
            with subcols[1]:
                sigma1 = st.number_input(f"σ₁ - Estado {i+1}", value=0.0, step=0.01, 
                                       key=f"sigma1_{i}", format="%.2f")
            with subcols[2]:
                sigma3 = st.number_input(f"σ₃ - Estado {i+1}", value=0.0, step=0.01, 
                                       key=f"sigma3_{i}", format="%.2f")
            
            if sigma1 > 0 and sigma3 > 0 and sigma1 > sigma3:
                mohr_data.append((sigma1, sigma3))
    
    # Parámetros del suelo
    col1, col2 = st.columns(2)
    with col1:
        phi_mohr = st.number_input("Ángulo de fricción φ (°)", value=30.0, step=0.1)
    with col2:
        cohesion = st.number_input("Cohesión c (kg/cm²)", value=0.0, step=0.01)
    
    if len(mohr_data) > 0:
        # Crear gráfica de círculos de Mohr
        fig_mohr, ax_mohr = plt.subplots(figsize=(10, 8))
        
        colors_mohr = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12']
        
        max_sigma = 0
        
        # Dibujar círculos
        for idx, (s1, s3) in enumerate(mohr_data):
            center = (s1 + s3) / 2
            radius = (s1 - s3) / 2
            max_sigma = max(max_sigma, s1)
            
            # Círculo
            theta = np.linspace(0, 2*np.pi, 400)
            x_circle = center + radius * np.cos(theta)
            y_circle = radius * np.sin(theta)
            
            color = colors_mohr[idx % len(colors_mohr)]
            ax_mohr.plot(x_circle, y_circle, linewidth=2.5, color=color, 
                        label=f'Estado {idx+1}: σ₁={s1:.2f}, σ₃={s3:.2f}')
            
            # Puntos principales
            ax_mohr.plot([s1, s3], [0, 0], 'o', color=color, markersize=8, markeredgecolor='black')
            
            # Etiquetas
            ax_mohr.annotate(f'σ₁', xy=(s1, 0), xytext=(s1, -radius*0.3), 
                           ha='center', fontsize=10, color=color, fontweight='bold')
            ax_mohr.annotate(f'σ₃', xy=(s3, 0), xytext=(s3, -radius*0.3), 
                           ha='center', fontsize=10, color=color, fontweight='bold')
        
        # Envolvente de falla
        phi_rad = np.deg2rad(phi_mohr)
        max_sigma_plot = max_sigma * 1.3
        x_envelope = np.linspace(0, max_sigma_plot, 200)
        y_envelope = cohesion / np.cos(phi_rad) + x_envelope * np.tan(phi_rad)
        
        ax_mohr.plot(x_envelope, y_envelope, 'r--', linewidth=3, 
                    label=f'Envolvente: τ = {cohesion:.2f} + σ·tan({phi_mohr:.1f}°)')
        
        # Configuración de la gráfica
        ax_mohr.set_xlabel('Esfuerzo Normal σ (kg/cm²)', fontsize=12, fontweight='bold')
        ax_mohr.set_ylabel('Esfuerzo Cortante τ (kg/cm²)', fontsize=12, fontweight='bold')
        ax_mohr.set_title('Círculos de Mohr y Criterio de Falla de Coulomb', 
                         fontsize=14, fontweight='bold', pad=20)
        ax_mohr.grid(True, linestyle=':', alpha=0.6)
        ax_mohr.legend(loc='upper left', framealpha=0.9)
        ax_mohr.set_aspect('equal', adjustable='box')
        
        # Mejorar límites
        ax_mohr.set_xlim(-max_sigma_plot*0.05, max_sigma_plot)
        ax_mohr.set_ylim(-max_sigma_plot*0.1, max_sigma_plot*0.6)
        
        # Mejorar apariencia
        ax_mohr.spines['top'].set_visible(False)
        ax_mohr.spines['right'].set_visible(False)
        ax_mohr.spines['left'].set_linewidth(1.5)
        ax_mohr.spines['bottom'].set_linewidth(1.5)
        
        st.pyplot(fig_mohr)
        
        # Tabla de análisis
        st.markdown("##### 📊 Análisis de Estados de Esfuerzo")
        analysis_data = []
        for idx, (s1, s3) in enumerate(mohr_data):
            tau_max = (s1 - s3) / 2
            sigma_oct = (s1 + s3) / 2
            analysis_data.append({
                "Estado": f"E-{idx+1}",
                "σ₁ (kg/cm²)": f"{s1:.2f}",
                "σ₃ (kg/cm²)": f"{s3:.2f}",
                "τ_max (kg/cm²)": f"{tau_max:.2f}",
                "σ_oct (kg/cm²)": f"{sigma_oct:.2f}",
                "Relación σ₁/σ₃": f"{s1/s3:.2f}"
            })
        
        df_analysis = pd.DataFrame(analysis_data)
        st.dataframe(df_analysis, use_container_width=True, hide_index=True)
        
    else:
        st.info("👆 Ingrese al menos un par válido de esfuerzos principales para generar el análisis")

# ================== PIE DE PÁGINA ==================
st.markdown("---")
col1, col2, col3 = st.columns([2, 1, 2])

with col1:
    st.markdown("#### 🔗 Contacto Profesional")
    st.markdown("""
    - [LinkedIn](https://www.linkedin.com/in/giancarlo-p%C3%A9rez-vargas-a04a3026b/)
    - [Instagram](https://www.instagram.com/giancarlo_perez_vargas/)
    - geocorex@gmail.com
    """)

with col3:
    st.markdown("#### ⚖️ Información Legal")
    st.caption("© 2025 GeoCoreX - Todos los derechos reservados")
    st.caption("Software desarrollado para análisis geotécnico preliminar")
    st.caption("Versión 2.0 - Actualizado en 2025")