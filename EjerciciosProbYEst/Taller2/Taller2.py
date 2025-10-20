import matplotlib.pyplot as plt
import numpy as np

# ============================================================================
# EJERCICIO 1: Hallar función de probabilidad y función de distribución
# ============================================================================

print("=" * 80)
print("EJERCICIO 1: Función de Probabilidad y Función de Distribución")
print("=" * 80)
print("\nDado: f(x) = k·x² para 0 ≤ x ≤ 6")
print("      f(x) = 0  para x < 0 o x > 6")

# PARTE 1: Hallar k tal que ∫f(x)dx = 1
print("\n" + "=" * 80)
print("PARTE 1: Hallar la constante k")
print("=" * 80)

k = 1/72
print(f"\n∫₀⁶ k·x² dx = 1")
print(f"k·[x³/3]₀⁶ = 1")
print(f"k·(216/3) = 1")
print(f"k·72 = 1")
print(f"k = {k:.6f} = 1/72")

integral = k * (6**3 / 3)
print(f"\nVerificación: k·72 = {integral:.6f} ✓")

# PARTE 2: Función de distribución acumulativa F(x)
print("\n" + "=" * 80)
print("PARTE 2: Función de Distribución Acumulativa F(x)")
print("=" * 80)

print("\nF(x) = P(X ≤ x) = ∫₀ˣ f(t)dt")
print("F(x) = ∫₀ˣ (1/72)·t² dt")
print("F(x) = (1/72)·[t³/3]₀ˣ")
print("F(x) = x³/216  para 0 ≤ x ≤ 6")

# VISUALIZACIONES EJERCICIO 1
x_vals = np.linspace(0, 6, 500)
f_x = (1/72) * x_vals**2
F_x = x_vals**3 / 216

fig = plt.figure(figsize=(16, 10))

# GRÁFICO 1: Función de Densidad f(x)
ax1 = plt.subplot(2, 3, 1)
ax1.plot(x_vals, f_x, 'b-', linewidth=2.5, label=r'$f(x) = \frac{x^2}{72}$')
ax1.fill_between(x_vals, f_x, alpha=0.3, color='blue')
ax1.axhline(y=0, color='k', linewidth=0.5)
ax1.axvline(x=0, color='k', linewidth=0.5)
ax1.axvline(x=6, color='red', linestyle='--', linewidth=2, alpha=0.7, label='x = 6')
ax1.scatter([0, 6], [0, (1/72)*36], color='red', s=100, zorder=5, edgecolors='black')

ax1.set_xlabel('x', fontsize=12, fontweight='bold')
ax1.set_ylabel('f(x)', fontsize=12, fontweight='bold')
ax1.set_title(r'Función de Densidad: $f(x) = \frac{x^2}{72}$ para $0 \leq x \leq 6$', 
            fontsize=13, fontweight='bold')
ax1.set_xlim([-0.5, 7])
ax1.set_ylim([0, max(f_x)*1.15])
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.legend(fontsize=11)
ax1.text(3, max(f_x)*0.5, r'Área = $\int_0^6 f(x)dx = 1$', 
        fontsize=11, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
        ha='center')

# GRÁFICO 2: Función de Distribución Acumulativa F(x)
ax2 = plt.subplot(2, 3, 2)
x_before = np.linspace(-1, 0, 50)
x_range = np.linspace(0, 6, 500)
x_after = np.linspace(6, 7, 50)

ax2.plot(x_before, np.zeros_like(x_before), 'b-', linewidth=2.5)
ax2.plot(x_range, x_range**3/216, 'b-', linewidth=2.5, label=r'$F(x) = \frac{x^3}{216}$')
ax2.plot(x_after, np.ones_like(x_after), 'b-', linewidth=2.5)
ax2.scatter([0, 6], [0, 1], color='red', s=100, zorder=5, edgecolors='black')

ax2.axhline(y=0, color='k', linewidth=0.5)
ax2.axhline(y=1, color='g', linewidth=1, linestyle='--', alpha=0.5, label='F(x) = 1')
ax2.axvline(x=0, color='k', linewidth=0.5)

ax2.set_xlabel('x', fontsize=12, fontweight='bold')
ax2.set_ylabel('F(x)', fontsize=12, fontweight='bold')
ax2.set_title(r'Función de Distribución: $F(x) = \frac{x^3}{216}$', 
            fontsize=13, fontweight='bold')
ax2.set_xlim([-1, 7])
ax2.set_ylim([-0.1, 1.15])
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.legend(fontsize=11)

ax2.text(-0.5, -0.05, 'F(x)=0', fontsize=10, color='blue', fontweight='bold')
ax2.text(3, 0.5, r'$F(x)=\frac{x^3}{216}$', fontsize=10, color='blue', fontweight='bold')
ax2.text(6.5, 1.05, 'F(x)=1', fontsize=10, color='blue', fontweight='bold')

# GRÁFICO 3: Tabla de Valores
ax3 = plt.subplot(2, 3, 3)
ax3.axis('tight')
ax3.axis('off')

x_table = [0, 1, 2, 3, 4, 5, 6]
f_table = [(1/72)*x**2 for x in x_table]
F_table = [x**3/216 for x in x_table]

table_data = [['x', 'f(x)', 'F(x)']]
for x, fx, Fx in zip(x_table, f_table, F_table):
    table_data.append([f'{x}', f'{fx:.6f}', f'{Fx:.6f}'])

table = ax3.table(cellText=table_data, cellLoc='center', loc='center',
                colWidths=[0.2, 0.35, 0.35])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.5)

for i in range(3):
    table[(0, i)].set_facecolor('#4CAF50')
    table[(0, i)].set_text_props(weight='bold', color='white', fontsize=11)

for i in range(1, len(table_data)):
    for j in range(3):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#f0f0f0')

ax3.set_title('Valores de f(x) y F(x)', fontsize=13, fontweight='bold', pad=20)

# GRÁFICO 4: Área bajo la curva
ax4 = plt.subplot(2, 3, 4)
ax4.plot(x_vals, f_x, 'b-', linewidth=2.5)
ax4.fill_between(x_vals, f_x, alpha=0.4, color='green', label='Área = 1')
ax4.axhline(y=0, color='k', linewidth=0.5)

n_segments = 6
for i in range(n_segments):
    x_start = i
    x_end = i + 1
    x_segment = np.linspace(x_start, x_end, 100)
    f_segment = (1/72) * x_segment**2
    ax4.fill_between(x_segment, f_segment, alpha=0.3, 
                    edgecolor='black', linewidth=1.5)

ax4.set_xlabel('x', fontsize=12, fontweight='bold')
ax4.set_ylabel('f(x)', fontsize=12, fontweight='bold')
ax4.set_title(r'Comprobación: $\int_0^6 \frac{x^2}{72}dx = 1$', 
            fontsize=13, fontweight='bold')
ax4.set_xlim([0, 6.5])
ax4.grid(True, alpha=0.3, linestyle='--')
ax4.legend(fontsize=11)

# GRÁFICO 5: Comparación f(x) vs F(x)
ax5 = plt.subplot(2, 3, 5)
ax5_twin = ax5.twinx()

line1 = ax5.plot(x_vals, f_x, 'b-', linewidth=2.5, label='f(x) - Densidad')
ax5.fill_between(x_vals, f_x, alpha=0.3, color='blue')
line2 = ax5_twin.plot(x_vals, F_x, 'r-', linewidth=2.5, label='F(x) - Acumulada')

ax5.set_xlabel('x', fontsize=12, fontweight='bold')
ax5.set_ylabel('f(x)', fontsize=11, fontweight='bold', color='blue')
ax5_twin.set_ylabel('F(x)', fontsize=11, fontweight='bold', color='red')
ax5.set_title('Comparación: Densidad vs Distribución Acumulada', 
            fontsize=13, fontweight='bold')

ax5.tick_params(axis='y', labelcolor='blue')
ax5_twin.tick_params(axis='y', labelcolor='red')
ax5.grid(True, alpha=0.3, linestyle='--')

lines = line1 + line2
labels = [l.get_label() for l in lines]
ax5.legend(lines, labels, loc='upper left', fontsize=10)

# GRÁFICO 6: Información del ejercicio
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')

info_text = r"""
EJERCICIO 1: RESUMEN

Función de densidad:
• f(x) = x²/72  para 0 ≤ x ≤ 6
• f(x) = 0      en otro caso

Constante de normalización:
• k = 1/72

Función de distribución:
• F(x) = 0      si x < 0
• F(x) = x³/216 si 0 ≤ x ≤ 6  
• F(x) = 1      si x > 6

Comprobación:
∫₀⁶ (x²/72)dx = [x³/216]₀⁶ 
            = 216/216 = 1 ✓
"""

ax6.text(0.1, 0.95, info_text, transform=ax6.transAxes,
        fontsize=11, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.suptitle('EJERCICIO 1: Función de Probabilidad y Función de Distribución', 
            fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig('Taller2_parte1.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nResultados guardados en: Taller2_parte1.png")

# ============================================================================
# EJERCICIO 2: SUMA DE DOS DADOS
# ============================================================================

print("\n" + "=" * 80)
print("EJERCICIO 2: Suma de Dos Dados")
print("=" * 80)
print("\nEspacio muestral: Lanzar dos dados de 6 caras")
print("Variable aleatoria X = suma de los dos dados")
print("Dominio: X ∈ {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}")

# Datos según el documento
x_dados = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
combinaciones = np.array([1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1])
total_combinaciones = 36

# Calcular probabilidades
P_x = combinaciones / total_combinaciones

print("\n1. FUNCIÓN DE PROBABILIDAD P(X=x)")
print("-" * 60)
print(f"{'x':<4} {'Combinaciones':<15} {'P(X=x)':<12} {'Porcentaje'}")
print("-" * 60)
for x, comb, p in zip(x_dados, combinaciones, P_x):
    print(f"{x:<4} {comb}/36 = {comb:<7} {p:.4f}       {p*100:.2f}%")
print("-" * 60)
print(f"TOTAL: {np.sum(combinaciones)}/36          {np.sum(P_x):.4f}       100.00% ✓")

# 2. Función de distribución acumulativa
F_x = np.cumsum(P_x)

print("\n2. FUNCIÓN DE DISTRIBUCIÓN ACUMULATIVA F(X)")
print("-" * 60)
print(f"{'x':<4} {'F(x) = P(X≤x)':<20} {'Decimal'}")
print("-" * 60)
for x, F in zip(x_dados, F_x):
    fraccion = f"{int(F*36)}/36"
    print(f"{x:<4} {fraccion:<20} {F:.4f}")

# 3. Valor Medio (Esperanza)
E_X = np.sum(x_dados * P_x)

print("\n3. VALOR MEDIO E[X]")
print("-" * 60)
print(f"E[X] = Σ x·P(X=x)")
print(f"E[X] = {E_X:.4f} ≈ 7.00")

# 4. Varianza
E_X2 = np.sum(x_dados**2 * P_x)
Var_X = E_X2 - E_X**2
std_X = np.sqrt(Var_X)

print("\n4. VARIANZA Var(X)")
print("-" * 60)
print(f"E[X²] = {E_X2:.4f}")
print(f"E[X]² = {E_X**2:.4f}")
print(f"Var(X) = E[X²] - (E[X])² = {Var_X:.4f}")
print(f"σ = √Var(X) = {std_X:.4f}")

# ============================================================================
# VISUALIZACIONES EJERCICIO 2
# ============================================================================

fig = plt.figure(figsize=(16, 12))
fig.suptitle('EJERCICIO 2: Probabilidad de la Suma de Dos Dados', 
            fontsize=16, fontweight='bold', y=0.995)

# GRÁFICO 1: Función de Probabilidad
ax1 = plt.subplot(3, 3, 1)
bars = ax1.bar(x_dados, P_x, width=0.7, color='coral', alpha=0.8, 
            edgecolor='black', linewidth=1.5)

for bar, p, c in zip(bars, P_x, combinaciones):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
            f'{c}/36\n{p:.3f}',
            ha='center', va='bottom', fontsize=9, fontweight='bold')

ax1.set_xlabel('Suma de dos dados (x)', fontsize=12, fontweight='bold')
ax1.set_ylabel('P(X = x)', fontsize=12, fontweight='bold')
ax1.set_title('Función de Probabilidad P(X=x)', fontsize=13, fontweight='bold')
ax1.set_xticks(x_dados)
ax1.grid(axis='y', alpha=0.3, linestyle='--')
ax1.set_ylim([0, max(P_x)*1.25])

# GRÁFICO 2: Función de Distribución Acumulativa
ax2 = plt.subplot(3, 3, 2)
ax2.step(np.append(x_dados, x_dados[-1]+1), 
        np.append(F_x, 1), 
        where='post', linewidth=2.5, color='purple', label='F(x)')
ax2.scatter(x_dados, F_x, color='purple', s=100, zorder=5, 
            edgecolors='black', linewidth=1.5)

for x, F in zip(x_dados, F_x):
    ax2.text(x + 0.2, F - 0.03, f'{F:.3f}', fontsize=8, 
            color='purple', fontweight='bold')

ax2.set_xlabel('x', fontsize=12, fontweight='bold')
ax2.set_ylabel('F(x) = P(X ≤ x)', fontsize=12, fontweight='bold')
ax2.set_title('Función de Distribución Acumulativa F(x)', fontsize=13, fontweight='bold')
ax2.set_xticks(x_dados)
ax2.set_yticks(np.arange(0, 1.1, 0.1))
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.set_ylim([0, 1.1])
ax2.legend(fontsize=11)

# GRÁFICO 3: Tabla de Probabilidades
ax3 = plt.subplot(3, 3, 3)
ax3.axis('tight')
ax3.axis('off')

table_data = [['x', 'Comb.', 'P(X=x)', 'F(x)']]
for x, c, p, F in zip(x_dados, combinaciones, P_x, F_x):
    table_data.append([f'{x}', f'{c}/36', f'{p:.4f}', f'{F:.4f}'])

table = ax3.table(cellText=table_data, cellLoc='center', loc='center',
                colWidths=[0.15, 0.25, 0.25, 0.25])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2)

for i in range(4):
    table[(0, i)].set_facecolor('#FF6347')
    table[(0, i)].set_text_props(weight='bold', color='white', fontsize=10)

for i in range(1, len(table_data)):
    for j in range(4):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#f0f0f0')

ax3.set_title('Tabla de Valores', fontsize=13, fontweight='bold', pad=20)

# GRÁFICO 4: Valor Medio con distribución
ax4 = plt.subplot(3, 3, 4)
bars = ax4.bar(x_dados, P_x, width=0.7, color='lightblue', 
            alpha=0.7, edgecolor='black', linewidth=1.5)
ax4.axvline(E_X, color='red', linestyle='--', linewidth=3, 
            label=f'E[X] = {E_X:.2f}')

ax4.set_xlabel('x', fontsize=12, fontweight='bold')
ax4.set_ylabel('P(X = x)', fontsize=12, fontweight='bold')
ax4.set_title(f'Valor Medio E[X] = {E_X:.2f}', fontsize=13, fontweight='bold')
ax4.set_xticks(x_dados)
ax4.grid(axis='y', alpha=0.3, linestyle='--')
ax4.legend(fontsize=11)

# GRÁFICO 5: Varianza con ±σ
ax5 = plt.subplot(3, 3, 5)
bars = ax5.bar(x_dados, P_x, width=0.7, color='lightgreen', 
            alpha=0.7, edgecolor='black', linewidth=1.5)

ax5.axvline(E_X, color='red', linestyle='--', linewidth=3, 
            label=f'μ = {E_X:.2f}')
ax5.axvline(E_X - std_X, color='orange', linestyle=':', linewidth=2, 
            label=f'μ - σ = {E_X - std_X:.2f}')
ax5.axvline(E_X + std_X, color='orange', linestyle=':', linewidth=2, 
            label=f'μ + σ = {E_X + std_X:.2f}')
ax5.axvspan(E_X - std_X, E_X + std_X, alpha=0.2, color='orange')

ax5.set_xlabel('x', fontsize=12, fontweight='bold')
ax5.set_ylabel('P(X = x)', fontsize=12, fontweight='bold')
ax5.set_title(f'Varianza = {Var_X:.4f}, σ = {std_X:.4f}', fontsize=13, fontweight='bold')
ax5.set_xticks(x_dados)
ax5.grid(axis='y', alpha=0.3, linestyle='--')
ax5.legend(fontsize=9)

# GRÁFICO 6: Comparación P(x) vs F(x)
ax6 = plt.subplot(3, 3, 6)
ax6_twin = ax6.twinx()

bars = ax6.bar(x_dados, P_x, width=0.4, color='steelblue', alpha=0.6, 
            label='P(X=x)', edgecolor='black')
line = ax6_twin.plot(x_dados, F_x, 'o-', linewidth=2.5, markersize=8, 
                    color='darkgreen', label='F(x)')

ax6.set_xlabel('x', fontsize=12, fontweight='bold')
ax6.set_ylabel('P(X = x)', fontsize=11, fontweight='bold', color='steelblue')
ax6_twin.set_ylabel('F(x)', fontsize=11, fontweight='bold', color='darkgreen')
ax6.set_title('Comparación: P(x) vs F(x)', fontsize=13, fontweight='bold')
ax6.set_xticks(x_dados)
ax6.tick_params(axis='y', labelcolor='steelblue')
ax6_twin.tick_params(axis='y', labelcolor='darkgreen')
ax6.grid(True, alpha=0.3, linestyle='--')

lines1, labels1 = ax6.get_legend_handles_labels()
lines2, labels2 = ax6_twin.get_legend_handles_labels()
ax6.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)

# GRÁFICO 7: Representación visual de los dados
ax7 = plt.subplot(3, 3, 7)
ax7.axis('off')

# Crear matriz de combinaciones
dado1 = [1, 2, 3, 4, 5, 6]
dado2 = [1, 2, 3, 4, 5, 6]
sumas = np.zeros((6, 6))

for i, d1 in enumerate(dado1):
    for j, d2 in enumerate(dado2):
        sumas[i, j] = d1 + d2

im = ax7.imshow(sumas, cmap='RdYlGn_r', aspect='auto')
ax7.set_xticks(range(6))
ax7.set_yticks(range(6))
ax7.set_xticklabels(dado2)
ax7.set_yticklabels(dado1)
ax7.set_xlabel('Dado 2', fontsize=11, fontweight='bold')
ax7.set_ylabel('Dado 1', fontsize=11, fontweight='bold')
ax7.set_title('Matriz de Sumas', fontsize=13, fontweight='bold')

for i in range(6):
    for j in range(6):
        text = ax7.text(j, i, int(sumas[i, j]),
                    ha="center", va="center", color="black", fontweight='bold')

plt.colorbar(im, ax=ax7, label='Suma')

# GRÁFICO 8: Estadísticas resumidas
ax8 = plt.subplot(3, 3, 8)
ax8.axis('tight')
ax8.axis('off')

stats_data = [
    ['Estadístico', 'Valor'],
    ['Mínimo', f'{x_dados[0]}'],
    ['Máximo', f'{x_dados[-1]}'],
    ['Media E[X]', f'{E_X:.4f}'],
    ['E[X²]', f'{E_X2:.4f}'],
    ['Varianza', f'{Var_X:.4f}'],
    ['Desv. Est. σ', f'{std_X:.4f}'],
    ['Moda', '7'],
    ['Mediana', '7']
]

table = ax8.table(cellText=stats_data, cellLoc='center', loc='center',
                colWidths=[0.5, 0.5])
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.5)

for i in range(2):
    table[(0, i)].set_facecolor('#2196F3')
    table[(0, i)].set_text_props(weight='bold', color='white', fontsize=12)

for i in range(1, len(stats_data)):
    for j in range(2):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#f0f0f0')

ax8.set_title('Resumen Estadístico', fontsize=13, fontweight='bold', pad=20)

# GRÁFICO 9: Info del ejercicio
ax9 = plt.subplot(3, 3, 9)
ax9.axis('off')

info_text = """
EJERCICIO 2: RESUMEN

Espacio Muestral:
• Lanzar dos dados de 6 caras
• Total de resultados: 36

Variable Aleatoria:
• X = suma de los dos dados
• Rango: {2, 3, 4, ..., 12}

Probabilidades:
• Máxima: P(X=7) = 6/36 = 0.1667
• Mínima: P(X=2) = P(X=12) = 1/36

Estadísticas:
• Media: E[X] = 7.00
• Varianza: Var(X) ≈ 5.83
• Desv. Est.: σ ≈ 2.42

Propiedades:
• Distribución simétrica
• Moda = Media = Mediana = 7
"""

ax9.text(0.1, 0.95, info_text, transform=ax9.transAxes,
        fontsize=10, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.9))

plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig('Taller2_parte2.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nResultados guardados en: Taller2_parte2.png")