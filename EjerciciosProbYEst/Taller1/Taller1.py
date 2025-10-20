import matplotlib.pyplot as plt
import numpy as np

# =========================
# Gráfica de probabilidades (foto 1)
# =========================
prob_labels = ['(d)', '(e)', '(f)', '(g)', '(h)']
prob_values = [23/57, 18/95, 3/95, 81/500, 27/1000]

plt.figure(figsize=(7,4))
plt.bar(prob_labels, prob_values, color='skyblue', edgecolor='black')
plt.title('Gráfica 1 - Probabilidades (Ejercicios combinatorios)')
plt.xlabel('Ejercicio')
plt.ylabel('Probabilidad')
plt.grid(axis='y', linestyle='--', alpha=0.6)
for i, v in enumerate(prob_values):
    plt.text(i, v + 0.01, f"{v:.3f}", ha='center')
plt.savefig("Taller1-grafica1_probabilidades.png")  # guarda en la carpeta del archivo
plt.show()

# =========================
# Gráfica de combinaciones y permutaciones (foto 2)
# =========================
combin_labels = ['Ingenieros', 'Abogados', 'Ambos']
combin_values = [10, 35, 350]  # (5C2), (7C3), total = 10*35

plt.figure(figsize=(7,4))
plt.bar(combin_labels, combin_values, color='lightgreen', edgecolor='black')
plt.title('Gráfica 2 - Combinaciones y Permutaciones')
plt.xlabel('Tipo de selección')
plt.ylabel('Número de formas')
plt.grid(axis='y', linestyle='--', alpha=0.6)
for i, v in enumerate(combin_values):
    plt.text(i, v + 5, str(v), ha='center')
plt.savefig("Taller1-grafica2_combinaciones.png")
plt.show()

# =========================
# Gráfica de permutaciones con repetición (foto 3)
# =========================
labels_perm = ['Electrónica', 'Sistemas', 'Industrial']
values_perm = [5, 2, 3]
total_perm = 3628800 / (120 * 2 * 6)  # 10! / (5!*2!*3!) = 2520

plt.figure(figsize=(7,4))
plt.bar(labels_perm, values_perm, color='salmon', edgecolor='black')
plt.title('Gráfica 3 - Distribución por tipo de ingeniero')
plt.xlabel('Categoría')
plt.ylabel('Cantidad de ingenieros')
plt.text(1.2, max(values_perm) + 0.5, f'Total permutaciones = {int(total_perm)}', fontsize=10, color='darkred')
plt.savefig("Taller1-grafica3_permutaciones.png")
plt.show()

# =========================
# Gráfica de probabilidades con dados (foto 4)
# =========================
dice_labels = ['P(7 u 11)', 'P(No 7 ni 11)', 'P(No 7 ni 11 en 2 lanzamientos)']
dice_values = [2/9, 7/9, (7/9)**2]

plt.figure(figsize=(7,4))
plt.bar(dice_labels, dice_values, color='orange', edgecolor='black')
plt.title('Gráfica 4 - Probabilidades con dados')
plt.ylabel('Probabilidad')
plt.grid(axis='y', linestyle='--', alpha=0.6)
for i, v in enumerate(dice_values):
    plt.text(i, v + 0.01, f"{v:.3f}", ha='center')
plt.savefig("Taller1-grafica4_dados.png")
plt.show()

# =========================
# Gráfica general comparando fracciones finales (resumen total)
# =========================
final_labels = ['23/57', '18/95', '3/95', '81/500', '27/1000', '49/81']
final_values = [23/57, 18/95, 3/95, 81/500, 27/1000, 49/81]

plt.figure(figsize=(8,4))
plt.plot(final_labels, final_values, marker='o', color='purple', linestyle='-', linewidth=2)
plt.title('Gráfica 5 - Comparación de probabilidades finales')
plt.xlabel('Fracción')
plt.ylabel('Valor decimal')
plt.grid(True, linestyle='--', alpha=0.6)
for i, v in enumerate(final_values):
    plt.text(i, v + 0.01, f"{v:.3f}", ha='center')
plt.savefig("Taller1-grafica5_comparacion_final.png")
plt.show()