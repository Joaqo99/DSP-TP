import numpy as np

def get_position(string_values, axis=False):
    """
    Extrae coordenadas desde strings con formato "[x, y, z]" y las devuelve como lista de vectores o coordenadas individuales.

    Parámetros:
    -----------
    pos_list : list of str
        Lista de strings con formato "[x, y, z]".
        
    axis : str or bool, opcional (default=False)
        Si se especifica como "x", "y" o "z", se devuelve solo esa componente de cada vector.
        Si es False, se devuelve el vector completo [x, y, z].

    Retorna:
    --------
    final_values : list of float
        Lista con los vectores completos o con las componentes individuales especificadas.
    """
    final_values = []
    for value in string_values:
        new_value = value.strip()
        new_value = new_value.replace("[", "")
        new_value = new_value.replace("]", "")
        x, y, z = new_value.split(",")
        x = np.round(float(x), 2)
        y = np.round(float(y), 2)
        z = np.round(float(z), 2)
        if not axis:
            final_values.append([x, y, z])
        if axis == "x":
            final_values.append(x)
        elif axis == "y":
            final_values.append(y)
        elif axis == "z":
            final_values.append(z)
        else:
            ValueError("Not possible axis")
    return final_values