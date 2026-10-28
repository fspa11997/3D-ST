# Motor GPU TripoSR para 3D-ST (Modal)

Este servicio ejecuta TripoSR en una GPU separada. La web principal sigue en Railway.

## 1. Crear cuenta y preparar Modal
1. Crea una cuenta en https://modal.com/ y revisa el plan/créditos y precios vigentes.
2. Instala Python 3.10 o superior en tu computador (para el cliente de Modal) y abre PowerShell en esta carpeta.
3. Crea un entorno virtual e instala Modal:
   ```powershell
   py -m venv .venv
   .venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip modal
   modal setup
   ```

## 2. Crear un token secreto compartido
Genera un token largo aleatorio y guarda exactamente el mismo valor en Modal y Railway.
En PowerShell puedes generar uno:
```powershell
python -c "import secrets; print(secrets.token_urlsafe(40))"
```
Copia el resultado en un lugar seguro. No lo publiques en GitHub.

Crea el secreto en Modal:
```powershell
modal secret create triposr-api GPU_API_TOKEN=PEGA_AQUI_TU_TOKEN
```
Reemplaza `PEGA_AQUI_TU_TOKEN` por el token real, sin comillas.

## 3. Desplegar el endpoint GPU
Desde la carpeta `gpu_worker`, ejecuta:
```powershell
modal deploy modal_app.py
```
La primera compilación puede tardar porque instala las dependencias de TripoSR. Modal devolverá una URL pública para la aplicación/endpoints. Copia la URL del endpoint ASGI y añade `/generate` al final. Por ejemplo, si la URL base fuera `https://usuario--3d-st-triposr-web.modal.run`, la URL para Railway sería `https://usuario--3d-st-triposr-web.modal.run/generate`.

Comprueba el endpoint de salud abriendo la URL base más `/health`. Debe devolver JSON con `"status":"ok"` una vez que el worker se haya iniciado y cargado el modelo.

## 4. Configurar Railway
En el servicio correcto de la web 3D-ST, configura estas variables:
- `GENERATOR_MODE` = `remote`
- `GPU_API_URL` = URL completa del endpoint terminada en `/generate`
- `GPU_API_TOKEN` = el mismo token secreto creado en Modal
- `SECRET_KEY` = una clave aleatoria distinta (no reutilices el token GPU)

Railway debe tener `requirements.txt` y `Procfile` del proyecto actual. Después de actualizar esas variables, redepliega.

## Costos y seguridad
- La GPU usa `T4` y se escala a cero después de 60 segundos de inactividad. El primer pedido tras una pausa puede tardar más por el arranque en frío y carga del modelo.
- Se cobra el uso de Modal según sus condiciones actuales; configura límites de presupuesto en el panel.
- El token de la API es privado: nunca lo guardes en archivos versionados ni lo compartas en capturas.
- Este primer endpoint devuelve OBJ con color por vértice cuando el modelo lo exporta. No genera necesariamente una textura UV; eso es una mejora separada.
- Revisa la licencia MIT de TripoSR y conserva sus avisos cuando distribuyas el modelo/código.
