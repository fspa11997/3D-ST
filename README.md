# 3D-ST — Imagen a 3D con Google Colab

La aplicación Flask presenta un flujo guiado para generar modelos 3D con TripoSR en un cuaderno interactivo de Google Colab. No se requiere contratar un servidor GPU para hacer pruebas.

## Flujo de uso
1. Abre la web de 3D-ST.
2. Pulsa **Abrir cuaderno en Google Colab**. Si el cuaderno todavía no está publicado en GitHub, usa **Descargar cuaderno (.ipynb)** e impórtalo desde Archivo → Subir notebook en Colab.
3. En Colab, activa una GPU si hay disponibilidad y ejecuta las celdas en orden.
4. Selecciona una imagen cuando el cuaderno lo solicite.
5. Descarga los modelos GLB u OBJ al terminar.

## Cuaderno
`notebooks/3D-ST-TripoSR-Colab.ipynb` contiene la preparación del entorno, instalación de PyTorch/torchmcubes/TripoSR, carga de imagen, inferencia y descarga.

## Publicar en GitHub
Para que el botón **Abrir cuaderno en Google Colab** funcione directamente, sube el archivo del cuaderno a la rama `master` en la ruta `notebooks/3D-ST-TripoSR-Colab.ipynb`. La URL utilizada por la página apunta a esa ruta.

## Limitaciones
- Google Colab gratuito no garantiza disponibilidad de GPU ni sesiones permanentes; el entorno puede terminarse y quizá sea necesario volver a instalar las dependencias.
- La web no envía automáticamente la imagen a Colab. El usuario selecciona la imagen dentro del cuaderno y descarga allí el resultado.
- TripoSR estima la forma a partir de una imagen. La malla puede necesitar reparación, escala y validación antes de imprimirla en 3D.
- La app conserva el backend remoto previo como código, pero la interfaz principal ofrece el flujo de Colab; no se necesita configurar `GPU_API_URL` para utilizar el cuaderno.

## Probar localmente
```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```
