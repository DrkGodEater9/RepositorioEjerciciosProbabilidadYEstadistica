import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import poisson, norm
from scipy.special import factorial

print("=" * 80)
print("TALLER: DISTRIBUCIONES DE POISSON Y NORMAL")
print("Alexander Morales")
print("=" * 80)

# ============================================================================
# EJERCICIO POISSON 1: Llamadas Telefónicas por Hora
# ============================================================================

print("\n" + "=" * 80)
print("EJERCICIO POISSON 1: Llamadas Telefónicas")
print("=" * 80)

print("\nUn centro de llamados recibe en promedio 10 llamadas por hora.")
print("¿P(X ≤ llamadas en 1 hora)?")
print("\nDatos: λ=10 (Promedio de llamadas por hora)")
print("       X ≥ 7 (Llamados que desean calcular cierta)")

# Parámetro lambda
lambda_1 = 10

# Fórmula de Poisson: P(X=x) = (λ^x * e^(-λ)) / x!
print(f"\nFórmula: P(X=x) = (λ^x * e^(-λ)) / x!")

# Calcular P(X≤5)
x_val = 5
prob_x5 = poisson.cdf(x_val, lambda_1)
print(f"\nP(X≤{x_val}) = {prob_x5:.4f} ≈ 6.71%")

# Verificación manual
print(f"\nVerificación: P(X≤5) = Σ P(X=x) para x=0,1,2,3,4,5")
prob_manual = sum(poisson.pmf(k, lambda_1) for k in range(x_val + 1))
print(f"P(X≤5) = {prob_manual:.4f}")

# Razón del cálculo
print(f"\nRazón: P(X≤5) = {prob_x5:.4f} ≈ 0.0671 = 6.71%")

# ============================================================================
# EJERCICIO POISSON 2: Defectos en Máquina
# ============================================================================

print("\n" + "=" * 80)
print("EJERCICIO POISSON 2: Defectos en Máquina")
print("=" * 80)

print("\nEn una fábrica de chocolate se producen 3 defectos en una")
print("máquina por cada hora de trabajo.")
print("E(X) ≤ 5 defectos en 1 hora?")
print("\nDatos: λ=3 (Promedio de defectos por hora)")
print("       X ≤ 5 (Defectos a calcular por hora)")

lambda_2 = 3

# Calcular probabilidades
print("\n" + "-" * 70)
print("DISTRIBUCIÓN DE PROBABILIDADES:")
print("-" * 70)

for x in range(6):
    prob = poisson.pmf(x, lambda_2)
    print(f"P(X={x}) = {prob:.4f} = {prob*100:.2f}%")

# P(X≤5)
prob_x5_ej2 = poisson.cdf(5, lambda_2)
print("-" * 70)
print(f"P(X≤5) = {prob_x5_ej2:.4f} = {prob_x5_ej2*100:.2f}%")

# ============================================================================
# EJERCICIO NORMAL 1: Botellas de Agua (Distribución Normal)
# ============================================================================

print("\n" + "=" * 80)
print("EJERCICIO NORMAL 1: Botellas de Agua - Marca Sigue una Distribución")
print("=" * 80)

print("\nLa función de los botellas de cada marca sigue una distribución")
print("normal con media de 1700 horas y desviación estándar 100 horas.")
print("E.P(X) ≤ Villa Vale más de 1356 y 4")
print("\nDatos: μ = 1700 h (Media)")
print("       σ = 100 h (Desviación estándar)")

mu_1 = 1700
sigma_1 = 100

# Datos del ejercicio
print("\nDatos del problema:")
print("• Media μ = 1700 h")
print("• Desviación estándar σ = 100 h")  
print("• Queremos P(X > 1330)")

# Convertir a valores z:
x_value = 1330
z = (x_value - mu_1) / sigma_1

print(f"\nConvertir a valor z:")
print(f"z = (X - μ) / σ = ({x_value} - {mu_1}) / {sigma_1}")
print(f"z = {z:.2f}")

# Calcular probabilidad
prob_normal_1 = 1 - norm.cdf(x_value, mu_1, sigma_1)
print(f"\nP(X > {x_value}) = 1 - P(X ≤ {x_value})")
print(f"P(X > {x_value}) = 1 - Φ(z)")
print(f"P(X > {x_value}) = 1 - Φ({z:.2f})")
print(f"P(X > {x_value}) = {prob_normal_1:.4f} = {prob_normal_1*100:.2f}%")

# ============================================================================
# EJERCICIO NORMAL 2: Calificaciones de Examen
# ============================================================================

print("\n" + "=" * 80)
print("EJERCICIO NORMAL 2: Calificaciones de un Examen")
print("=" * 80)

print("\nLas calificaciones de un examen se distribuyen normalmente")
print("con media de 70 y desviación estándar de 8")
print("¿P(X) ≤ estimula obtenga una calificación entre 60 y 85?")
print("\nDatos: μ = 70")
print("       σ = 8")

mu_2 = 70
sigma_2 = 8

# Calcular P(60 < X < 85)
x1 = 60
x2 = 85

# Convertir a z
z1 = (x1 - mu_2) / sigma_2
z2 = (x2 - mu_2) / sigma_2

print(f"\nConvertir a valores z:")
print(f"z₁ = ({x1} - {mu_2}) / {sigma_2} = {z1:.2f}")
print(f"z₂ = ({x2} - {mu_2}) / {sigma_2} = {z2:.2f}")

# Calcular probabilidades
prob_z1 = norm.cdf(x1, mu_2, sigma_2)
prob_z2 = norm.cdf(x2, mu_2, sigma_2)
prob_between = prob_z2 - prob_z1

print(f"\nP({x1} < X < {x2}) = Φ(z₂) - Φ(z₁)")
print(f"P({x1} < X < {x2}) = Φ({z2:.2f}) - Φ({z1:.2f})")
print(f"P({x1} < X < {x2}) = {prob_z2:.4f} - {prob_z1:.4f}")
print(f"P({x1} < X < {x2}) = {prob_between:.4f} = {prob_between*100:.2f}%")

# ============================================================================
# VISUALIZACIONES
# ============================================================================

fig = plt.figure(figsize=(18, 12))
fig.suptitle('DISTRIBUCIONES DE POISSON Y NORMAL', fontsize=16, fontweight='bold', y=0.995)

# ========== POISSON 1 ==========

# Gráfico 1: Distribución Poisson λ=10
ax1 = plt.subplot(3, 3, 1)
x_poisson_1 = np.arange(0, 25)
y_poisson_1 = poisson.pmf(x_poisson_1, lambda_1)

bars = ax1.bar(x_poisson_1, y_poisson_1, width=0.8, alpha=0.7, 
            color='steelblue', edgecolor='black', linewidth=1.5)

# Destacar x≤5
for i in range(6):
    bars[i].set_color('orange')
    bars[i].set_alpha(0.9)

ax1.set_xlabel('x (Número de llamadas)', fontsize=10, fontweight='bold')
ax1.set_ylabel('P(X=x)', fontsize=10, fontweight='bold')
ax1.set_title(f'Poisson 1: λ={lambda_1} llamadas/hora', fontsize=11, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)
ax1.axvline(x=5.5, color='red', linestyle='--', linewidth=2, label=f'P(X≤5)={prob_x5:.4f}')
ax1.legend(fontsize=9)

# Gráfico 2: Función acumulada Poisson 1
ax2 = plt.subplot(3, 3, 2)
x_range = np.arange(0, 25)
cdf_vals = [poisson.cdf(x, lambda_1) for x in x_range]

ax2.step(x_range, cdf_vals, where='post', linewidth=2.5, color='darkblue')
ax2.scatter(x_range, cdf_vals, s=50, color='darkblue', zorder=5, edgecolors='black')
ax2.axhline(y=prob_x5, color='red', linestyle='--', linewidth=2, alpha=0.7)
ax2.axvline(x=5, color='red', linestyle='--', linewidth=2, alpha=0.7)
ax2.scatter([5], [prob_x5], s=200, color='red', marker='*', 
        edgecolors='black', linewidth=2, zorder=10)

ax2.set_xlabel('x', fontsize=10, fontweight='bold')
ax2.set_ylabel('F(x) = P(X≤x)', fontsize=10, fontweight='bold')
ax2.set_title('Función Acumulada', fontsize=11, fontweight='bold')
ax2.grid(True, alpha=0.3)

# Gráfico 3: Tabla Poisson 1
ax3 = plt.subplot(3, 3, 3)
ax3.axis('tight')
ax3.axis('off')

table_data = [['x', 'P(X=x)', 'P(X≤x)']]
for x in range(8):
    pmf = poisson.pmf(x, lambda_1)
    cdf = poisson.cdf(x, lambda_1)
    table_data.append([f'{x}', f'{pmf:.4f}', f'{cdf:.4f}'])

table = ax3.table(cellText=table_data, cellLoc='center', loc='center',
                colWidths=[0.2, 0.4, 0.4])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2)

for i in range(3):
    table[(0, i)].set_facecolor('#FF6347')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Destacar fila x=5
table[(6, 0)].set_facecolor('#FFD700')
table[(6, 1)].set_facecolor('#FFD700')
table[(6, 2)].set_facecolor('#FFD700')

ax3.set_title('Poisson 1: Probabilidades', fontsize=11, fontweight='bold', pad=20)

# ========== POISSON 2 ==========

# Gráfico 4: Distribución Poisson λ=3
ax4 = plt.subplot(3, 3, 4)
x_poisson_2 = np.arange(0, 15)
y_poisson_2 = poisson.pmf(x_poisson_2, lambda_2)

bars2 = ax4.bar(x_poisson_2, y_poisson_2, width=0.8, alpha=0.7, 
                color='coral', edgecolor='black', linewidth=1.5)

for i in range(6):
    bars2[i].set_color('purple')
    bars2[i].set_alpha(0.9)

ax4.set_xlabel('x (Número de defectos)', fontsize=10, fontweight='bold')
ax4.set_ylabel('P(X=x)', fontsize=10, fontweight='bold')
ax4.set_title(f'Poisson 2: λ={lambda_2} defectos/hora', fontsize=11, fontweight='bold')
ax4.grid(axis='y', alpha=0.3)
ax4.axvline(x=5.5, color='red', linestyle='--', linewidth=2, 
        label=f'P(X≤5)={prob_x5_ej2:.4f}')
ax4.legend(fontsize=9)

# Gráfico 5: Comparación Poisson
ax5 = plt.subplot(3, 3, 5)
x_comp = np.arange(0, 20)
y_comp_1 = poisson.pmf(x_comp, lambda_1)
y_comp_2 = poisson.pmf(x_comp, lambda_2)

ax5.plot(x_comp, y_comp_1, 'o-', linewidth=2, markersize=6, 
        label=f'λ={lambda_1}', color='steelblue')
ax5.plot(x_comp, y_comp_2, 's-', linewidth=2, markersize=6, 
        label=f'λ={lambda_2}', color='coral')

ax5.set_xlabel('x', fontsize=10, fontweight='bold')
ax5.set_ylabel('P(X=x)', fontsize=10, fontweight='bold')
ax5.set_title('Comparación Poisson λ=10 vs λ=3', fontsize=11, fontweight='bold')
ax5.legend(fontsize=9)
ax5.grid(True, alpha=0.3)

# Gráfico 6: Info Poisson
ax6 = plt.subplot(3, 3, 6)
ax6.axis('off')

info_poisson = f"""
DISTRIBUCIÓN DE POISSON

Fórmula:
P(X=x) = (λˣ · e⁻λ) / x!

Ejercicio 1:
• λ = {lambda_1} llamadas/hora
• P(X≤5) = {prob_x5:.4f}
• Media: E[X] = λ = {lambda_1}
• Varianza: Var(X) = λ = {lambda_1}

Ejercicio 2:
• λ = {lambda_2} defectos/hora
• P(X≤5) = {prob_x5_ej2:.4f}
• Media: E[X] = λ = {lambda_2}
• Varianza: Var(X) = λ = {lambda_2}

Propiedades:
• Modelo de eventos raros
• E[X] = Var(X) = λ
"""

ax6.text(0.1, 0.95, info_poisson, transform=ax6.transAxes,
        fontsize=9, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

# ========== NORMAL 1 ==========

# Gráfico 7: Distribución Normal μ=1700, σ=100
ax7 = plt.subplot(3, 3, 7)
x_normal_1 = np.linspace(mu_1 - 4*sigma_1, mu_1 + 4*sigma_1, 1000)
y_normal_1 = norm.pdf(x_normal_1, mu_1, sigma_1)

ax7.plot(x_normal_1, y_normal_1, 'b-', linewidth=2.5)
ax7.fill_between(x_normal_1, y_normal_1, alpha=0.3, color='blue')

# Área x > 1330
x_fill = x_normal_1[x_normal_1 >= x_value]
y_fill = norm.pdf(x_fill, mu_1, sigma_1)
ax7.fill_between(x_fill, y_fill, alpha=0.6, color='orange', 
                label=f'P(X>{x_value})={prob_normal_1:.4f}')

ax7.axvline(mu_1, color='red', linestyle='--', linewidth=2, label=f'μ={mu_1}')
ax7.axvline(x_value, color='green', linestyle='--', linewidth=2, label=f'x={x_value}')

ax7.set_xlabel('Horas', fontsize=10, fontweight='bold')
ax7.set_ylabel('Densidad', fontsize=10, fontweight='bold')
ax7.set_title(f'Normal 1: μ={mu_1}, σ={sigma_1}', fontsize=11, fontweight='bold')
ax7.legend(fontsize=8)
ax7.grid(True, alpha=0.3)

# ========== NORMAL 2 ==========

# Gráfico 8: Distribución Normal μ=70, σ=8
ax8 = plt.subplot(3, 3, 8)
x_normal_2 = np.linspace(mu_2 - 4*sigma_2, mu_2 + 4*sigma_2, 1000)
y_normal_2 = norm.pdf(x_normal_2, mu_2, sigma_2)

ax8.plot(x_normal_2, y_normal_2, 'g-', linewidth=2.5)
ax8.fill_between(x_normal_2, y_normal_2, alpha=0.3, color='green')

# Área entre 60 y 85
x_fill2 = x_normal_2[(x_normal_2 >= x1) & (x_normal_2 <= x2)]
y_fill2 = norm.pdf(x_fill2, mu_2, sigma_2)
ax8.fill_between(x_fill2, y_fill2, alpha=0.6, color='gold',
                label=f'P({x1}<X<{x2})={prob_between:.4f}')

ax8.axvline(mu_2, color='red', linestyle='--', linewidth=2, label=f'μ={mu_2}')
ax8.axvline(x1, color='blue', linestyle='--', linewidth=1.5, alpha=0.7)
ax8.axvline(x2, color='blue', linestyle='--', linewidth=1.5, alpha=0.7)

ax8.set_xlabel('Calificación', fontsize=10, fontweight='bold')
ax8.set_ylabel('Densidad', fontsize=10, fontweight='bold')
ax8.set_title(f'Normal 2: μ={mu_2}, σ={sigma_2}', fontsize=11, fontweight='bold')
ax8.legend(fontsize=8)
ax8.grid(True, alpha=0.3)

# Gráfico 9: Resumen
ax9 = plt.subplot(3, 3, 9)
ax9.axis('tight')
ax9.axis('off')

summary_data = [
    ['Ejercicio', 'Distribución', 'Resultado'],
    ['Poisson 1', f'λ={lambda_1}', f'P(X≤5)={prob_x5:.4f}'],
    ['', 'Llamadas', f'{prob_x5*100:.2f}%'],
    ['Poisson 2', f'λ={lambda_2}', f'P(X≤5)={prob_x5_ej2:.4f}'],
    ['', 'Defectos', f'{prob_x5_ej2*100:.2f}%'],
    ['Normal 1', f'μ={mu_1},σ={sigma_1}', f'P(X>{x_value})={prob_normal_1:.4f}'],
    ['', 'Botellas', f'{prob_normal_1*100:.2f}%'],
    ['Normal 2', f'μ={mu_2},σ={sigma_2}', f'P({x1}<X<{x2})={prob_between:.4f}'],
    ['', 'Examen', f'{prob_between*100:.2f}%']
]

table = ax9.table(cellText=summary_data, cellLoc='center', loc='center',
                colWidths=[0.3, 0.35, 0.35])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2)

for i in range(3):
    table[(0, i)].set_facecolor('#4CAF50')
    table[(0, i)].set_text_props(weight='bold', color='white', fontsize=10)

for i in range(1, len(summary_data)):
    for j in range(3):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#f0f0f0')

ax9.set_title('RESUMEN DE RESULTADOS', fontsize=11, fontweight='bold', pad=20)

plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig('Taller4_parte1.png', dpi=300, bbox_inches='tight')
plt.savefig('Taller4_parte2.png', dpi=300, bbox_inches='tight')
plt.show()

