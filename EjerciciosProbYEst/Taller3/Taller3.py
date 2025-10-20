import matplotlib.pyplot as plt
import numpy as np
from scipy.special import comb
from mpl_toolkits.mplot3d import Axes3D

print("=" * 80)
print("TALLER #2 - PROBABILIDAD Y ESTADÍSTICA")
print("Alexander Morales - 2024101011")
print("=" * 80)

# ============================================================================
# EJERCICIO 1: DISTRIBUCIÓN DE ESTUDIANTES
# ============================================================================

print("\n" + "=" * 80)
print("EJERCICIO 1: Se seleccionan al azar 3 estudiantes")
print("=" * 80)

print("\nDe un salón que contiene 3 estudiantes de sistemas,")
print("3 de electrónica y 3 de industrial.")
print("\nX = # estudiantes de sistemas")
print("Y = # estudiantes de electrónica")

x_vals = np.array([0, 1, 2])
y_vals = np.array([0, 1, 2])

# Tabla de probabilidades conjuntas
prob_table = np.array([
    [3/28, 9/28, 3/28],   # x=0
    [6/28, 6/28, 0/28],   # x=1
    [1/28, 0/28, 0/28]    # x=2
])

print("\n1.H - TABLA DE PROBABILIDADES")
print("-" * 70)
print("f(x,y) |  y=0      y=1      y=2    | P(X=x)")
print("-" * 70)

marginal_x = np.sum(prob_table, axis=1)
for i, x in enumerate(x_vals):
    print(f"  x={x}  | {prob_table[i,0]:.4f}  {prob_table[i,1]:.4f}  {prob_table[i,2]:.4f}  | {marginal_x[i]:.4f}")

marginal_y = np.sum(prob_table, axis=0)
print("-" * 70)
print(f"P(Y=y) | {marginal_y[0]:.4f}  {marginal_y[1]:.4f}  {marginal_y[2]:.4f}  | 1.0000")
print("-" * 70)

# ============================================================================
# EJERCICIO 2: FÁBRICA DE DULCES
# ============================================================================

print("\n" + "=" * 80)
print("EJERCICIO 2: Fábrica de Dulces")
print("=" * 80)

print("\nUna fábrica de dulces distribuye cajas de chocolate cuya")
print("f.v. de densidad es dado por:")
print("\nf(x,y) = 2/3(2x + y)  para 0 ≤ x ≤ 1, 0 ≤ y ≤ 1")
print("       = 0            en otro caso")

print("\na) Verificar que f(x,y) es una f.v. de prob conjunta")
print("\n2/3 ∫₀¹ ∫₀¹ (2x + y) dx dy = 1 ✓")

# ============================================================================
# EJERCICIO 3: FUNCIÓN DE PROBABILIDAD x + y = 30
# ============================================================================

print("\n" + "=" * 80)
print("EJERCICIO 3: Función de Probabilidad x + y = 30")
print("=" * 80)

print("\nDada la función de probabilidad donde x + y = 30")
print("a las V.A. x a y discretas")

# Datos de la tabla del documento
datos_x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
datos_y = np.array([30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14])

# Verificar que x + y = 30
print(f"\nVerificación: x + y = 30 para todos los pares")
for i in range(min(5, len(datos_x))):
    print(f"  ({datos_x[i]}, {datos_y[i]}) → {datos_x[i]} + {datos_y[i]} = {datos_x[i] + datos_y[i]}")
print("  ...")

# Espacio muestral ampliado
print(f"\nPara x,y ∈ {{0,1,...,30}} - Espacio muestral")
print("Espacio muestral: 31 posibles valores")

# a) P(X+Y<4)
print("\na) P(X+Y<4)")
print("   Si x + y = 30 siempre, entonces x + y nunca es menor que 4")
print("   P(X+Y<4) = 0/31 = 0.0000")
prob_a = 0.0

# b) P(X≥4)
count_b = sum(1 for x in datos_x if x >= 4)
prob_b = count_b / len(datos_x)
print(f"\nb) P(X≥4)")
print(f"   Casos donde x ≥ 4: {count_b} de {len(datos_x)}")
print(f"   P(X≥4) = {count_b}/{len(datos_x)} = {prob_b:.4f}")

# c) P(X≥2, Y≤1)
count_c = sum(1 for x, y in zip(datos_x, datos_y) if x >= 2 and y <= 1)
prob_c = count_c / len(datos_x)
print(f"\nc) P(X≥2, Y≤1)")
print(f"   Si x ≥ 2 y x + y = 30, entonces y ≤ 28")
print(f"   Pero necesitamos y ≤ 1, lo cual implica x ≥ 29")
print(f"   Casos: {count_c} de {len(datos_x)}")
print(f"   P(X≥2, Y≤1) = {count_c}/{len(datos_x)} = {prob_c:.4f}")

# d) P(X≥4, Y≥1)
count_d = sum(1 for x, y in zip(datos_x, datos_y) if x >= 4 and y >= 1)
prob_d = count_d / len(datos_x)
print(f"\nd) P(X≥4, Y≥1)")
print(f"   Si x ≥ 4 y x + y = 30, entonces y = 30 - x ≤ 26")
print(f"   Necesitamos y ≥ 1, es decir x ≤ 29")
print(f"   Casos: {count_d} de {len(datos_x)}")
print(f"   P(X≥4, Y≥1) = {count_d}/{len(datos_x)} = {prob_d:.4f}")

# Análisis del espacio muestral completo (31 valores)
print("\n" + "-" * 70)
print("ANÁLISIS CON ESPACIO MUESTRAL COMPLETO (x ∈ {0,1,...,30}):")
print("-" * 70)

# Todos los posibles pares (x, 30-x) para x de 0 a 30
x_completo = np.arange(0, 31)
y_completo = 30 - x_completo
total_casos = len(x_completo)

# b) P(X≥4) en espacio completo
casos_b_completo = sum(1 for x in x_completo if x >= 4)
prob_b_completo = casos_b_completo / total_casos
print(f"\nb) P(X≥4) = {casos_b_completo}/{total_casos} = {prob_b_completo:.4f}")

# c) P(X≥2, Y≤1) en espacio completo
casos_c_completo = sum(1 for x, y in zip(x_completo, y_completo) if x >= 2 and y <= 1)
prob_c_completo = casos_c_completo / total_casos
print(f"c) P(X≥2, Y≤1) = {casos_c_completo}/{total_casos} = {prob_c_completo:.4f}")
if casos_c_completo > 0:
    print(f"   Casos favorables: x ∈ {{29, 30}}")

# d) P(X≥4, Y≥1)
casos_d_completo = sum(1 for x, y in zip(x_completo, y_completo) if x >= 4 and y >= 1)
prob_d_completo = casos_d_completo / total_casos
print(f"d) P(X≥4, Y≥1) = {casos_d_completo}/{total_casos} = {prob_d_completo:.4f}")
print(f"   Casos favorables: x ∈ {{4, 5, 6, ..., 29}}")

# ============================================================================
# VISUALIZACIÓN COMPLETA
# ============================================================================

fig = plt.figure(figsize=(18, 12))
fig.suptitle('TALLER #2: PROBABILIDAD Y ESTADÍSTICA - EJERCICIOS COMPLETOS', 
            fontsize=16, fontweight='bold', y=0.995)

# ========== EJERCICIO 1: 3 GRÁFICOS ==========

# Gráfico 1: Heatmap de probabilidades conjuntas
ax1 = plt.subplot(3, 3, 1)
im1 = ax1.imshow(prob_table, cmap='YlOrRd', aspect='auto', origin='lower', vmin=0, vmax=0.35)
ax1.set_xticks(range(len(y_vals)))
ax1.set_yticks(range(len(x_vals)))
ax1.set_xticklabels(y_vals)
ax1.set_yticklabels(x_vals)
ax1.set_xlabel('Y (Electrónica)', fontsize=10, fontweight='bold')
ax1.set_ylabel('X (Sistemas)', fontsize=10, fontweight='bold')
ax1.set_title('Ejercicio 1: f(x,y) Conjunta', fontsize=11, fontweight='bold')

for i in range(len(x_vals)):
    for j in range(len(y_vals)):
        value = prob_table[i, j]
        text = ax1.text(j, i, f'{value:.4f}\n({int(value*28)}/28)',
                    ha="center", va="center", color="black", fontsize=8, fontweight='bold')

plt.colorbar(im1, ax=ax1, label='Probabilidad', fraction=0.046)

# Gráfico 2: Distribución 3D
ax2 = plt.subplot(3, 3, 2, projection='3d')
X_grid, Y_grid = np.meshgrid(x_vals, y_vals)
Z_grid = prob_table.T

x_pos = X_grid.ravel()
y_pos = Y_grid.ravel()
z_pos = np.zeros_like(Z_grid).ravel()
dx = dy = 0.7
dz = Z_grid.ravel()

colors = plt.cm.viridis(dz / dz.max())
ax2.bar3d(x_pos, y_pos, z_pos, dx, dy, dz, color=colors, shade=True, alpha=0.8)

ax2.set_xlabel('X', fontsize=9, fontweight='bold')
ax2.set_ylabel('Y', fontsize=9, fontweight='bold')
ax2.set_zlabel('f(x,y)', fontsize=9, fontweight='bold')
ax2.set_title('Ej.1: Distribución 3D', fontsize=11, fontweight='bold')

# Gráfico 3: Marginales
ax3 = plt.subplot(3, 3, 3)
width = 0.35
x_pos_bar = x_vals - width/2
y_pos_bar = y_vals + width/2

bars1 = ax3.bar(x_pos_bar, marginal_x, width, label='P(X=x)', alpha=0.8, 
            color='steelblue', edgecolor='black', linewidth=1.5)
bars2 = ax3.bar(y_pos_bar, marginal_y, width, label='P(Y=y)', alpha=0.8, 
            color='coral', edgecolor='black', linewidth=1.5)

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=8, fontweight='bold')

ax3.set_xlabel('Valor', fontsize=10, fontweight='bold')
ax3.set_ylabel('Probabilidad', fontsize=10, fontweight='bold')
ax3.set_title('Ej.1: Marginales', fontsize=11, fontweight='bold')
ax3.legend(fontsize=9)
ax3.grid(axis='y', alpha=0.3)

# ========== EJERCICIO 2: 3 GRÁFICOS ==========

x_cont = np.linspace(0, 1, 100)
y_cont = np.linspace(0, 1, 100)
X_cont, Y_cont = np.meshgrid(x_cont, y_cont)
Z_cont = (2/3) * (2*X_cont + Y_cont)

# Gráfico 4: Superficie 3D
ax4 = plt.subplot(3, 3, 4, projection='3d')
surf = ax4.plot_surface(X_cont, Y_cont, Z_cont, cmap='viridis', alpha=0.8, 
                        edgecolor='none')
ax4.set_xlabel('X', fontsize=9, fontweight='bold')
ax4.set_ylabel('Y', fontsize=9, fontweight='bold')
ax4.set_zlabel('f(x,y)', fontsize=9, fontweight='bold')
ax4.set_title('Ej.2: f(x,y) = (2/3)(2x+y)', fontsize=11, fontweight='bold')
ax4.view_init(elev=20, azim=45)

# Gráfico 5: Curvas de nivel
ax5 = plt.subplot(3, 3, 5)
contour = ax5.contourf(X_cont, Y_cont, Z_cont, levels=15, cmap='viridis', alpha=0.8)
contour_lines = ax5.contour(X_cont, Y_cont, Z_cont, levels=10, colors='black', 
                            linewidths=0.5, alpha=0.4)
ax5.clabel(contour_lines, inline=True, fontsize=7)
ax5.set_xlabel('X (Densidad)', fontsize=10, fontweight='bold')
ax5.set_ylabel('Y (Hora)', fontsize=10, fontweight='bold')
ax5.set_title('Ej.2: Curvas de Nivel', fontsize=11, fontweight='bold')
plt.colorbar(contour, ax=ax5, label='f(x,y)', fraction=0.046)

# Gráfico 6: Densidades marginales
ax6 = plt.subplot(3, 3, 6)
f_X = (4*x_cont + 1) / 3
f_Y = 2*(1 + y_cont) / 3

line1 = ax6.plot(x_cont, f_X, 'b-', linewidth=2.5, label='f_X(x) = (4x+1)/3')
ax6.fill_between(x_cont, f_X, alpha=0.3, color='blue')

ax6_twin = ax6.twinx()
line2 = ax6_twin.plot(y_cont, f_Y, 'r-', linewidth=2.5, label='f_Y(y) = 2(1+y)/3')
ax6_twin.fill_between(y_cont, f_Y, alpha=0.3, color='red')

ax6.set_xlabel('x, y', fontsize=10, fontweight='bold')
ax6.set_ylabel('f_X(x)', fontsize=9, fontweight='bold', color='blue')
ax6_twin.set_ylabel('f_Y(y)', fontsize=9, fontweight='bold', color='red')
ax6.set_title('Ej.2: Densidades Marginales', fontsize=11, fontweight='bold')

lines = line1 + line2
labels = [l.get_label() for l in lines]
ax6.legend(lines, labels, loc='upper left', fontsize=8)
ax6.grid(True, alpha=0.3)

# ========== EJERCICIO 3: 3 GRÁFICOS ==========

# Gráfico 7: Línea x + y = 30
ax7 = plt.subplot(3, 3, 7)
ax7.plot(x_completo, y_completo, 'b-', linewidth=3, label='x + y = 30')
ax7.scatter(x_completo, y_completo, c='red', s=50, alpha=0.7, edgecolors='black', zorder=5)

# Destacar algunos puntos
puntos_destaque = [(0,30), (15,15), (30,0)]
for x, y in puntos_destaque:
    ax7.scatter(x, y, c='gold', s=200, marker='*', edgecolors='black', linewidth=2, zorder=10)
    ax7.annotate(f'({x},{y})', xy=(x, y), xytext=(x+1, y+1), 
                fontsize=9, fontweight='bold')

ax7.set_xlabel('X', fontsize=10, fontweight='bold')
ax7.set_ylabel('Y', fontsize=10, fontweight='bold')
ax7.set_title('Ej.3: Restricción x + y = 30', fontsize=11, fontweight='bold')
ax7.legend(fontsize=9)
ax7.grid(True, alpha=0.3)
ax7.set_xlim([-2, 32])
ax7.set_ylim([-2, 32])

# Gráfico 8: Probabilidades calculadas
ax8 = plt.subplot(3, 3, 8)
categories = ['P(X+Y<4)', 'P(X≥4)', 'P(X≥2,Y≤1)', 'P(X≥4,Y≥1)']
probabilities = [prob_a, prob_b_completo, prob_c_completo, prob_d_completo]
colors_bars = ['#4CAF50', '#FF9800', '#9C27B0', '#F44336']

bars = ax8.bar(range(len(categories)), probabilities, color=colors_bars, 
            alpha=0.8, edgecolor='black', linewidth=1.5)

for i, (bar, prob) in enumerate(zip(bars, probabilities)):
    height = bar.get_height()
    ax8.text(bar.get_x() + bar.get_width()/2., height + 0.02,
            f'{prob:.4f}',
            ha='center', va='bottom', fontsize=9, fontweight='bold')

ax8.set_xticks(range(len(categories)))
ax8.set_xticklabels(categories, rotation=20, ha='right', fontsize=8)
ax8.set_ylabel('Probabilidad', fontsize=10, fontweight='bold')
ax8.set_title('Ej.3: Probabilidades (n=31)', fontsize=11, fontweight='bold')
ax8.set_ylim([0, 1.0])
ax8.grid(axis='y', alpha=0.3)

# Gráfico 9: Tabla resumen
ax9 = plt.subplot(3, 3, 9)
ax9.axis('tight')
ax9.axis('off')

summary_data = [
    ['Ejercicio', 'Tipo', 'Resultado Clave'],
    ['1. Estudiantes', 'Discreta', 'f(1,1)=6/28'],
    ['', '', 'Σf(x,y)=1.0000'],
    ['2. Fábrica', 'Continua', '∫∫f(x,y)=1.0000'],
    ['', '', 'f_X(x)=(4x+1)/3'],
    ['3. x+y=30', 'Discreta', f'P(X≥4)={prob_b_completo:.4f}'],
    ['', 'Lineal', f'n=31 pares'],
    ['', '', ''],
    ['TALLER #2', 'COMPLETADO', '✓']
]

table = ax9.table(cellText=summary_data, cellLoc='left', loc='center',
                colWidths=[0.35, 0.3, 0.35])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2.2)

for i in range(3):
    table[(0, i)].set_facecolor('#2196F3')
    table[(0, i)].set_text_props(weight='bold', color='white', fontsize=10)

for i in range(3):
    table[(8, i)].set_facecolor('#4CAF50')
    table[(8, i)].set_text_props(weight='bold', color='white', fontsize=10)

for i in range(1, 8):
    for j in range(3):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#f0f0f0')

ax9.set_title('RESUMEN GENERAL', fontsize=11, fontweight='bold', pad=20)

plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig('Taller3_parte1.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# FIGURA ADICIONAL: ANÁLISIS DETALLADO EJERCICIO 3
# ============================================================================

fig2, axes = plt.subplots(2, 2, figsize=(14, 10))
fig2.suptitle('EJERCICIO 3: x + y = 30 - ANÁLISIS DETALLADO', fontsize=14, fontweight='bold')

# Gráfico 1: Distribución completa
ax1 = axes[0, 0]
ax1.scatter(x_completo, y_completo, s=100, c=x_completo, cmap='rainbow', 
        alpha=0.7, edgecolors='black', linewidth=1.5)
ax1.plot(x_completo, y_completo, 'b--', alpha=0.5, linewidth=2)
ax1.set_xlabel('X', fontsize=11, fontweight='bold')
ax1.set_ylabel('Y', fontsize=11, fontweight='bold')
ax1.set_title('Todos los pares (x, 30-x)', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_xlim([-2, 32])
ax1.set_ylim([-2, 32])

# Añadir línea x+y=30
x_line = np.array([0, 30])
y_line = 30 - x_line
ax1.plot(x_line, y_line, 'r-', linewidth=3, alpha=0.3, label='x+y=30')
ax1.legend(fontsize=10)

# Gráfico 2: Regiones de probabilidad
ax2 = axes[0, 1]
ax2.plot(x_completo, y_completo, 'b-', linewidth=3, alpha=0.3)

# Región X≥4
x_region_b = x_completo[x_completo >= 4]
y_region_b = y_completo[x_completo >= 4]
ax2.scatter(x_region_b, y_region_b, s=150, c='orange', alpha=0.7, 
        edgecolors='black', linewidth=2, label='P(X≥4)', marker='s')

# Región X≥2, Y≤1
x_region_c = x_completo[(x_completo >= 2) & (y_completo <= 1)]
y_region_c = y_completo[(x_completo >= 2) & (y_completo <= 1)]
ax2.scatter(x_region_c, y_region_c, s=200, c='purple', alpha=0.8, 
        edgecolors='black', linewidth=2, label='P(X≥2,Y≤1)', marker='^')

ax2.set_xlabel('X', fontsize=11, fontweight='bold')
ax2.set_ylabel('Y', fontsize=11, fontweight='bold')
ax2.set_title('Regiones de Interés', fontsize=12, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_xlim([-2, 32])
ax2.set_ylim([-2, 32])

# Gráfico 3: Distribución de X y Y
ax3 = axes[1, 0]
prob_uniforme = np.ones(len(x_completo)) / len(x_completo)

ax3.bar(x_completo, prob_uniforme, width=0.8, alpha=0.6, color='steelblue', 
    edgecolor='black', label='P(X=x)')
ax3.set_xlabel('Valor', fontsize=11, fontweight='bold')
ax3.set_ylabel('Probabilidad', fontsize=11, fontweight='bold')
ax3.set_title('Distribución Uniforme (cada caso = 1/31)', fontsize=12, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(axis='y', alpha=0.3)
ax3.set_ylim([0, 0.05])

# Gráfico 4: Tabla de resultados
ax4 = axes[1, 1]
ax4.axis('tight')
ax4.axis('off')

results_data = [
    ['Pregunta', 'Casos', 'Probabilidad'],
    ['a) P(X+Y<4)', '0', '0/31 = 0.0000'],
    ['   x+y=30 siempre', '—', 'Imposible'],
    ['b) P(X≥4)', f'{casos_b_completo}', f'{casos_b_completo}/31 = {prob_b_completo:.4f}'],
    ['   x ∈ {4,...,30}', '—', '—'],
    ['c) P(X≥2, Y≤1)', f'{casos_c_completo}', f'{casos_c_completo}/31 = {prob_c_completo:.4f}'],
    ['   x ∈ {29,30}', '—', '—'],
    ['d) P(X≥4, Y≥1)', f'{casos_d_completo}', f'{casos_d_completo}/31 = {prob_d_completo:.4f}'],
    ['   x ∈ {4,...,29}', '—', '—'],
    ['', '', ''],
    ['Total casos', '31', 'Σp = 1.0000']
]

table = ax4.table(cellText=results_data, cellLoc='center', loc='center',
                colWidths=[0.45, 0.2, 0.35])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 1.8)

table[(0, 0)].set_facecolor('#2196F3')
table[(0, 1)].set_facecolor('#2196F3')
table[(0, 2)].set_facecolor('#2196F3')
for i in range(3):
    table[(0, i)].set_text_props(weight='bold', color='white')

ax4.set_title('Resultados Detallados', fontsize=12, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('Taller3_parte2.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "=" * 80)
print("✅ TALLER #2 COMPLETADO CON ÉXITO")
print("=" * 80)
print("\nArchivos generados:")
print("  • Taller3_parte1.png - Vista general 9 gráficos")
print("  • Taller3_parte2.png - Análisis detallado Ejercicio 3")
print("\n✓ Ejercicio 3 CORREGIDO: x + y = 30")
print("=" * 80)
