# Imagen a 3D — 3D-ST + TripoSR

La web Flask corre en Railway y delega la inferencia de IA a un endpoint GPU independiente en Modal.

## Arquitectura
- `app.py`: carga de imagen, petición autenticada al endpoint GPU y descarga OBJ.
- `gpu_worker/modal_app.py`: API Modal que carga TripoSR en una GPU T4, procesa una imagen y devuelve una malla OBJ.
- `requirements.txt`: dependencias ligeras de la web (no incluye PyTorch ni CUDA).
- `Procfile`: comando de inicio para Railway.

## Estado
La aplicación web requiere un endpoint Modal desplegado y estas variables en Railway:
- `GENERATOR_MODE=remote`
- `GPU_API_URL=https://.../generate`
- `GPU_API_TOKEN=...`
- `SECRET_KEY=...`

Consulta `gpu_worker/README.md` para desplegar el motor y configurar los secretos.

## Probar localmente la web
```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:GENERATOR_MODE="demo"
python app.py
```
El modo demo genera un cubo y NO es una reconstrucción de la imagen. Para generación real necesitas configurar el endpoint Modal y usar `GENERATOR_MODE=remote`.

## Nota sobre archivos
Los archivos subidos se borran del servidor web después de la inferencia. Los OBJ se guardan temporalmente en `outputs/`; este prototipo aún necesita una política periódica de limpieza de modelos generados si se va a usar en producción.
