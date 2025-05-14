def saludar(nombre):
    """
    Devuelve un saludo personalizado.

    Args:
        nombre (str): Nombre de la persona a saludar.

    Returns:
        str: Saludo.
    """
    return f"Hola, {nombre}!"


if __name__ == "__main__":
    print(saludar("José"))
