import numpy as np
from scipy.stats import ttest_ind

# Notas de dos grupos de estudiantes
grupo_tutoria = [85, 88, 90, 92, 87, 91, 89, 95]
grupo_sin_tutoria = [78, 82, 80, 76, 79, 81, 77, 83]

# Prueba t de Student para comparar medias
t_stat, p_value = ttest_ind(grupo_tutoria, grupo_sin_tutoria)

print("Estadístico t:", t_stat)
print("Valor p:", p_value)

# Interpretación
if p_value < 0.05:
    print("Hay una diferencia significativa entre los grupos.")
else:
    print("No hay diferencia significativa entre los grupos.")
