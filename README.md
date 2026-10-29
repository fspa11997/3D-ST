# 3D-ST — Imagen a modelo 3D con Google Colab

La aplicación Flask recibe una fotografía y crea un cuaderno `.ipynb` personalizado que incluye esa imagen y el código para ejecutar TripoSR en Google Colab. La aplicación web no necesita ejecutar una GPU ni generar el modelo en Railway.

## Flujo por defecto
1. El usuario selecciona una imagen en la página de 3D-ST.
2. Flask crea un cuaderno de Colab con la fotografía incrustada.
3. El usuario descarga el cuaderno, abre Google Colab y usa **Archivo → Subir cuaderno**.
4. En Colab activa una GPU si está disponible y selecciona **Entorno de ejecución → Ejecutar todo**.
5. El cuaderno instala las dependencias, ejecuta TripoSR y descarga el resultado en formato OBJ.

## Configuración
Por defecto `GENERATOR_MODE=colab`. Railway solo aloja la web y prepara el cuaderno. No hace falta configurar Modal ni `GPU_API_URL` para este flujo.

Se mantienen los modos opcionales anteriores: `GENERATOR_MODE=remote` para un endpoint GPU externo, o `GENERATOR_MODE=demo` para un cubo OBJ de demostración.

## Ejecutar localmente
```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

## Limitaciones importantes
- El cuaderno contiene la imagen que el usuario subió; el archivo `.ipynb` puede ser grande para fotografías grandes.
- La primera ejecución descarga TripoSR y sus dependencias y puede tardar varios minutos.
- La GPU gratuita de Colab no está garantizada y las sesiones pueden finalizarse.
- Una sola fotografía produce una estimación de la geometría, no dimensiones exactas ni garantía de que la malla esté lista para impresión.
- `outputs/` contiene cuadernos personalizados con imágenes incrustadas. En producción se debe establecer una política de limpieza periódica y limitar el tamaño de las imágenes.
