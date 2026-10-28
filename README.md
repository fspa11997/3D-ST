# Imagen a 3D — prototipo Python

Aplicación web con Flask preparada para GitHub y Railway.

## Importante: estado de esta versión
Incluye la interfaz de carga, manejo de archivos y descarga OBJ. Por defecto usa `GENERATOR_MODE=demo` y crea un cubo OBJ de muestra para probar el flujo. **Ese cubo no es una reconstrucción de la imagen.** El motor de IA real aún debe conectarse.

## Probar en Windows
1. Instala Python 3.11.
2. En la carpeta del proyecto ejecuta:
   ```bat
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   python app.py
   ```
3. Abre http://127.0.0.1:5000

## Publicar en Railway
1. Sube el contenido a un repositorio nuevo de GitHub.
2. En Railway crea un proyecto y selecciona Deploy from GitHub Repo.
3. Configura `SECRET_KEY` con un valor aleatorio.
4. El sitio funcionará en modo demo hasta conectar el generador real.

## Conectar el motor de IA
Configura `GENERATOR_MODE=real` y `GENERATOR_COMMAND` con un comando que reciba `{input}` y `{output}`, por ejemplo:
`python generate_model.py "{input}" "{output}"`

El script indicado debe ejecutar un modelo de imagen-a-3D y escribir el OBJ final en la ruta de salida. No se incluyen pesos del modelo ni un generador real. La generación real requiere seleccionar un modelo compatible, revisar su licencia y contar con recursos GPU. Railway sirve la web; no debe asumirse que ofrece GPU.

## Antes de producción
Añadir limpieza de archivos temporales, límites de concurrencia, protección contra abuso y almacenamiento adecuado. Los modelos OBJ con texturas pueden requerir archivos MTL y de textura adicionales.
