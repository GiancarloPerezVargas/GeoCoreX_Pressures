# GeoCoreX_Pressures

**Aplicación web en Python + Streamlit para cálculo de presiones laterales y empuje total en suelos.**  

GeoCoreX_Pressures forma parte de la suite de software **GeoCoreX**, enfocada en soluciones geotécnicas inteligentes para ingeniería civil.

## ✨ Características principales
- Cálculo de **K₀** mediante el método de Jacky.  
- Diagrama de presiones laterales en función de estratos del suelo.  
- Cálculo de **empuje total** y **punto de aplicación (z̄)**.  
- Consideración de **nivel freático** y presión de agua.  
- Visualización gráfica del diagrama.  
- Interfaz simple y profesional mediante **Streamlit**.

## Tecnologías utilizadas
- **Python**  
- **Streamlit**  
- **NumPy**  
- **Pandas**  
- **Matplotlib**  

---

## Instalación y ejecución
1. Clonar el repositorio:
```bash
git clone https://github.com/TU_USUARIO/EMPUJES_APP.git

2. Entrar a la carpeta del proyecto:
cd EMPUJES_APP

3. Instalar dependencias:
pip install -r requirements.txt

4. Ejecutar la aplicación:
streamlit run app.py

5. Fórmulas principales usadas

Coeficiente de empuje en reposo (Jacky):
K0​=1−sin(φ)

Área de cada estrato:
Ai​=21​(psup​+pinf​)⋅Hi​

Empuje total:
Ptotal​=∑Ai​+Aagua​

Altura del punto de aplicación (resultante):
ˉ=Ptotal​∑(Ai​⋅zi​)​

 Aplicaciones

Diseño de muros de contención.

Estudio de empujes en excavaciones.

Análisis preliminar de estabilidad geotécnica.

Herramienta de soporte para ingenieros civiles y geotécnicos.

## Licencia

Este proyecto está licenciado bajo la MIT License.
Ver el archivo LICENSE
 para más detalles.

## Contacto

Giancarlo Pérez Vargas – Fundador de GeoCoreX

🌐 LinkedIn: https://www.linkedin.com/in/giancarlo-p%C3%A9rez-vargas-a04a3026b/

✉️ Correo: geocorex@gmail.com