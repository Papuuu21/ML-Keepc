from sklearn.preprocessing import LabelEncoder

def eliminacion_cols(df, cols_a_eliminar: list):
    df.drop(columns = cols_a_eliminar, inplace = True)
    return df


def imputar_valores(df, cols_cero, cols_menos_uno, cols_moda):
    # Imputar con 0
    cols_cero = cols_cero
    for col in cols_cero:
        df[col].fillna(value=0, inplace=True)
    
    # Imputar con -1
    cols_menos_uno = cols_menos_uno
    for col in cols_menos_uno:
        df[col].fillna(value=-1, inplace=True)
    
    # Imputar con la moda
    cols_moda = cols_moda
    for col in cols_moda:
        df[col].fillna(df[col].mode()[0], inplace=True)

    return df


def codificar_variables(df, cols_frecuencia, cols_labelencoder):
    """
    Codifica variables usando codificación de frecuencia y LabelEncoder.
    
    Args:
        df: df que contiene los datos.
        cols_frecuencia: Lista de columnas para codificación de frecuencia.
        cols_labelencoder: Lista de columnas para codificación con LabelEncoder.
    
    Returns:
        tuple: df con las columnas codificadas y diccionario con los LabelEncoders.
    """
    # Codificación de frecuencia
    for col in cols_frecuencia:
        frequency_encoding = df[col].value_counts().to_dict()
        df[f'{col} codif'] = df[col].map(frequency_encoding)
    
    # Codificación con LabelEncoder (uno por columna)
    encoders = {}
    for col in cols_labelencoder:
        le = LabelEncoder()
        df[f'{col} codif'] = le.fit_transform(df[col])
        encoders[col] = le  # Guardar el encoder para uso futuro
    
    return df, encoders
