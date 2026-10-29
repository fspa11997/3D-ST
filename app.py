import os
import uuid
import secrets
import base64
import json
import mimetypes
from pathlib import Path

import requests
from flask import Flask, render_template, request, send_file, redirect, url_for, flash
from werkzeug.utils import secure_filename

BASE = Path(__file__).resolve().parent
UPLOADS, OUTPUTS = BASE / "uploads", BASE / "outputs"
UPLOADS.mkdir(exist_ok=True)
OUTPUTS.mkdir(exist_ok=True)

ALLOWED = {"png", "jpg", "jpeg", "webp"}
MODE = os.getenv("GENERATOR_MODE", "colab").lower()
GPU_API_URL = os.getenv("GPU_API_URL", "").strip().rstrip("/")
GPU_API_TOKEN = os.getenv("GPU_API_TOKEN", "").strip()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "change-this-in-production")
app.config["MAX_CONTENT_LENGTH"] = int(os.getenv("MAX_UPLOAD_MB", "12")) * 1024 * 1024


def demo_obj(path):
    """DEMO ONLY: cube mesh to test download; not reconstructed from the image."""
    verts = [(-1,-1,-1),(1,-1,-1),(1,1,-1),(-1,1,-1),(-1,-1,1),(1,-1,1),(1,1,1),(-1,1,1)]
    faces = [(1,4,3),(1,3,2),(5,6,7),(5,7,8),(1,2,6),(1,6,5),(2,3,7),(2,7,6),(3,4,8),(3,8,7),(4,1,5),(4,5,8)]
    with path.open("w", encoding="utf-8") as f:
        f.write("# DEMO ONLY: cube, not generated from uploaded image\no DemoCube\n")
        for v in verts:
            f.write("v %s %s %s\n" % v)
        for face in faces:
            f.write("f %d %d %d\n" % face)


def remote_generator(src, dst):
    if not GPU_API_URL:
        raise RuntimeError(
            "Falta configurar GPU_API_URL en Railway. Despliega primero el motor TripoSR en Modal "
            "y copia la URL de su endpoint /generate."
        )
    if not GPU_API_TOKEN:
        raise RuntimeError("Falta configurar GPU_API_TOKEN en Railway.")

    headers = {"Authorization": f"Bearer {GPU_API_TOKEN}"}
    timeout = int(os.getenv("GENERATION_TIMEOUT_SECONDS", "900"))
    try:
        with src.open("rb") as image_file:
            response = requests.post(
                GPU_API_URL,
                headers=headers,
                files={"image": (src.name, image_file, "application/octet-stream")},
                data={"remove_background": "true"},
                timeout=(20, timeout),
            )
        if response.status_code != 200:
            detail = response.text[:1200] if response.text else "Respuesta vacía del servidor GPU."
            raise RuntimeError(f"El servicio TripoSR respondió HTTP {response.status_code}: {detail}")
        content_type = response.headers.get("content-type", "").lower()
        if "json" in content_type:
            try:
                detail = response.json().get("detail", response.text[:1200])
            except Exception:
                detail = response.text[:1200]
            raise RuntimeError(f"El servicio GPU no devolvió un archivo OBJ: {detail}")
        body = response.content
        if not body or b"\nv " not in body and not body.lstrip().startswith(b"v "):
            raise RuntimeError("La respuesta del servicio GPU no parece ser un archivo OBJ válido.")
        dst.write_bytes(body)
    except requests.Timeout as exc:
        raise RuntimeError("La generación tardó demasiado. Intenta de nuevo con una imagen más sencilla.") from exc
    except requests.RequestException as exc:
        raise RuntimeError(f"No se pudo conectar con el servidor GPU: {exc}") from exc



def create_colab_notebook(image_path, original_name, notebook_path):
    """Create a Colab notebook that carries the uploaded image inside it."""
    image_bytes = image_path.read_bytes()
    image_b64 = base64.b64encode(image_bytes).decode("ascii")
    mime = mimetypes.guess_type(original_name)[0] or "application/octet-stream"
    safe_name = secure_filename(original_name) or "imagen.png"
    # Image data is embedded so the user does not need to upload the same photo twice.
    cells = [
        {"cell_type": "markdown", "metadata": {}, "source": [
            "# 3D-ST · Generar modelo 3D desde tu fotografía\n",
            "\n",
            "Este cuaderno fue preparado por 3D-ST con la fotografía que seleccionaste. "
            "En Colab elige **Entorno de ejecución → Cambiar tipo de entorno de ejecución → GPU** "
            "si hay una GPU disponible y después usa **Entorno de ejecución → Ejecutar todo**.\n",
            "\n",
            "La fotografía está incluida en este cuaderno; no necesitas volver a subirla. "
            "La instalación puede tardar varios minutos y la disponibilidad gratuita de GPU no está garantizada."
        ]},
        {"cell_type": "markdown", "metadata": {}, "source": ["## 1. Preparar el entorno de TripoSR\n", "Esta celda usa el Python y PyTorch que ya trae Colab; no crea un entorno virtual ni depende de una ruta fija."]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
            "import os, pathlib, subprocess, sys\n",
            "def ejecutar(cmd):\n",
            "    print('$', ' '.join(cmd))\n",
            "    r = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)\n",
            "    if r.returncode != 0:\n",
            "        print(r.stdout[-6000:])  # mostrar el error real de pip\n",
            "        raise RuntimeError('Falló: ' + ' '.join(str(c) for c in cmd[:6]) + ' ... Revisa el registro de arriba.')\n",
            "PY = sys.executable\n",
            "ejecutar([PY, '-m', 'pip', 'install', '--upgrade', 'scikit-build-core', 'pybind11', 'ninja', 'cmake'])\n",
            "ejecutar([PY, '-m', 'pip', 'install', '--no-cache-dir', '--no-build-isolation', 'git+https://github.com/tatsy/torchmcubes.git'])\n",
            "ejecutar([PY, '-m', 'pip', 'install', 'omegaconf==2.3.0', 'Pillow', 'einops', 'transformers>=4.40,<5', 'trimesh', 'rembg', 'onnxruntime', 'huggingface-hub', 'imageio[ffmpeg]', 'gradio', 'xatlas', 'moderngl'])\n",
            "if not pathlib.Path('/content/TripoSR/run.py').exists():\n",
            "    ejecutar(['git', 'clone', 'https://github.com/VAST-AI-Research/TripoSR.git', '/content/TripoSR'])\n",
            "ejecutar([PY, '-c', \"import torch, torchmcubes, rembg, onnxruntime; print('PyTorch:', torch.__version__); print('CUDA:', torch.version.cuda); print('GPU disponible:', torch.cuda.is_available()); print('Dependencias: OK')\"])\n",
            "try:\n",
            "    subprocess.run(['nvidia-smi'], check=True)\n",
            "except Exception:\n",
            "    print('No se detectó una GPU NVIDIA. Si Colab ofrece GPU, actívala en Entorno de ejecución → Cambiar tipo de entorno de ejecución.')\n",
            "print('Entorno preparado.')"
        ]},
        {"cell_type": "markdown", "metadata": {}, "source": ["## 2. Recuperar la fotografía que seleccionaste en 3D-ST"]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
            "import base64, pathlib\n",
            f"nombre_imagen = {json.dumps(safe_name, ensure_ascii=False)}\n",
            f"mime_imagen = {json.dumps(mime)}\n",
            f"imagen_b64 = {json.dumps(image_b64)}\n",
            "imagen = pathlib.Path('/content') / nombre_imagen\n",
            "imagen.write_bytes(base64.b64decode(imagen_b64))\n",
            "print('Fotografía lista:', imagen.name, '-', round(imagen.stat().st_size / 1024, 1), 'KB')"
        ]},
        {"cell_type": "markdown", "metadata": {}, "source": ["## 3. Generar el modelo 3D\n", "TripoSR estima la geometría a partir de la fotografía. El resultado no garantiza dimensiones exactas ni que la malla esté lista para impresión sin revisión."]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
            "import pathlib, subprocess, os, sys\n",
            "PY = sys.executable\n",
            "SALIDA = '/content/3dst-output'\n",
            "os.makedirs(SALIDA, exist_ok=True)\n",
            "import torch\n",
            "hay_gpu = torch.cuda.is_available()\n",
            "print('GPU disponible:', hay_gpu)\n",
            "if not hay_gpu:\n",
            "    print('AVISO: sin GPU el proceso es lento y usa mucha RAM. Activa una GPU en Entorno de ejecución → Cambiar tipo de entorno de ejecución.')\n",
            "# Resolución de la malla: más alta = más detalle, pero más memoria. Si se agota, baja sola.\n",
            "resoluciones = [320, 256, 192] if hay_gpu else [192, 128]\n",
            "print('Generando el modelo 3D...')\n",
            "for res in resoluciones:\n",
            "    print('Intentando con mc-resolution =', res)\n",
            "    comando = [PY, '/content/TripoSR/run.py', str(imagen), '--output-dir', SALIDA, '--model-save-format', 'obj', '--mc-resolution', str(res), '--chunk-size', '4096']\n",
            "    r = subprocess.run(comando)\n",
            "    if r.returncode == 0:\n",
            "        break\n",
            "    print(f'Falló con código {r.returncode} (probable falta de memoria). Reintentando con menor resolución...')\n",
            "else:\n",
            "    raise RuntimeError('TripoSR no pudo generar la malla ni con la resolución más baja. Activa una GPU o reinicia el entorno de ejecución.')\n",
            "archivos = [p for p in pathlib.Path(SALIDA).rglob('*') if p.is_file()]\n",
            "if not archivos:\n",
            "    raise FileNotFoundError('TripoSR terminó, pero no se encontraron archivos de salida.')\n",
            "for archivo in archivos:\n",
            "    print(archivo, f'({archivo.stat().st_size / 1024:.1f} KB)')"
        ]},
        {"cell_type": "markdown", "metadata": {}, "source": ["## 4. Pulir la malla (opcional)\n", "Elimina fragmentos sueltos, repara huecos pequeños y suaviza la superficie. Genera `figura_pulida.obj`. Si el suavizado borra demasiado detalle, baja `iterations` o sáltate esta celda."]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
            "import trimesh, pathlib\n",
            "SALIDA = pathlib.Path('/content/3dst-output')\n",
            "PULIDO = SALIDA / 'pulido'\n",
            "PULIDO.mkdir(exist_ok=True)\n",
            "origen = sorted(SALIDA.rglob('mesh.obj'))\n",
            "if not origen:\n",
            "    raise FileNotFoundError('No se encontró mesh.obj. Ejecuta antes la celda de generación.')\n",
            "malla = trimesh.load(str(origen[0]), force='mesh')\n",
            "print('Original:', len(malla.vertices), 'vértices,', len(malla.faces), 'caras')\n",
            "# 1) Quitar fragmentos sueltos: conservar solo la pieza más grande\n",
            "partes = malla.split(only_watertight=False)\n",
            "if len(partes) > 1:\n",
            "    malla = max(partes, key=lambda m: len(m.faces))\n",
            "# 2) Reparar huecos pequeños y normales\n",
            "trimesh.repair.fill_holes(malla)\n",
            "trimesh.repair.fix_normals(malla)\n",
            "# 3) Suavizado Taubin: alisa la superficie sin encoger la figura\n",
            "trimesh.smoothing.filter_taubin(malla, lamb=0.5, nu=-0.53, iterations=10)\n",
            "malla.export(str(PULIDO / 'figura_pulida.obj'))\n",
            "print('Pulida:', len(malla.vertices), 'vértices,', len(malla.faces), 'caras')\n",
            "print('Guardada en', PULIDO)"
        ]},
        {"cell_type": "markdown", "metadata": {}, "source": ["## 5. Descargar el proyecto 3D\n", "Al ejecutar esta celda, Colab descargará el modelo OBJ (pulido si ejecutaste el paso 4)."]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
            "from google.colab import files\n",
            "import pathlib\n",
            "salida = pathlib.Path('/content/3dst-output')\n",
            "modelos = sorted((salida / 'pulido').glob('*.obj')) or sorted(salida.rglob('*.obj'))\n",
            "if not modelos:\n",
            "    raise FileNotFoundError('No se encontró ningún archivo OBJ.')\n",
            "for modelo in modelos:\n",
            "    print('Descargando:', modelo.name)\n",
            "    files.download(str(modelo))"
        ]},
        {"cell_type": "markdown", "metadata": {}, "source": ["### Nota\n", "Google Colab puede no ofrecer GPU en todas las sesiones y puede terminar sesiones inactivas. Una sola fotografía solo muestra una parte del objeto, por lo que la IA estima las zonas que no se ven."]}
    ]
    notebook = {"cells": cells, "metadata": {"colab": {"name": "3D-ST - " + safe_name, "provenance": []}, "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    notebook_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8")


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("image")
        if not file or not file.filename:
            flash("Selecciona una imagen para continuar.", "error")
            return redirect(url_for("index"))
        ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
        if ext not in ALLOWED:
            flash("Formato no admitido. Usa PNG, JPG, JPEG o WebP.", "error")
            return redirect(url_for("index"))

        job = uuid.uuid4().hex
        src = UPLOADS / (job + "_" + (secure_filename(file.filename) or "imagen"))
        dst = OUTPUTS / (job + ".obj")
        file.save(src)

        try:
            if MODE == "colab":
                notebook_name = job + ".ipynb"
                notebook_path = OUTPUTS / notebook_name
                create_colab_notebook(src, file.filename, notebook_path)
                return render_template(
                    "result.html",
                    notebook=notebook_name,
                    image_name=secure_filename(file.filename) or "imagen",
                    colab=True,
                    status="Tu fotografía quedó incluida en un cuaderno de Colab. Descárgalo, súbelo a Colab y ejecuta las celdas en orden para obtener el modelo 3D.",
                    demo=False,
                )
            elif MODE == "demo":
                demo_obj(dst)
                status = "Modo demostración: se creó un cubo OBJ. No es una reconstrucción de la imagen."
            elif MODE == "remote":
                remote_generator(src, dst)
                status = "Modelo 3D generado por TripoSR correctamente."
            else:
                raise RuntimeError("GENERATOR_MODE no válido. Usa 'colab', 'remote' o 'demo'.")
            return render_template("result.html", filename=dst.name, status=status, demo=(MODE == "demo"), colab=False)
        except Exception as exc:
            app.logger.exception("Fallo al generar modelo")
            flash(str(exc), "error")
            return redirect(url_for("index"))
        finally:
            # The uploaded source is no longer needed after inference.
            try:
                src.unlink(missing_ok=True)
            except OSError:
                pass
    return render_template("index.html", max_mb=os.getenv("MAX_UPLOAD_MB", "12"))


@app.route("/download-notebook/<filename>")
def download_notebook(filename):
    if not filename.endswith(".ipynb") or Path(filename).name != filename:
        flash("Cuaderno no válido.", "error")
        return redirect(url_for("index"))
    path = OUTPUTS / filename
    if not path.is_file():
        flash("El cuaderno ya no está disponible. Vuelve a seleccionar la imagen.", "error")
        return redirect(url_for("index"))
    return send_file(path, as_attachment=True, download_name="3D-ST-Colab.ipynb", mimetype="application/x-ipynb+json")


@app.route("/download/<filename>")
def download(filename):
    if not filename.endswith(".obj") or Path(filename).name != filename:
        flash("Archivo no válido.", "error")
        return redirect(url_for("index"))
    path = OUTPUTS / filename
    if not path.is_file():
        flash("El archivo ya no está disponible.", "error")
        return redirect(url_for("index"))
    return send_file(path, as_attachment=True, download_name="modelo_3d.obj", mimetype="text/plain")


@app.route("/health")
def health():
    return {"status": "ok", "generator_mode": MODE, "gpu_configured": bool(GPU_API_URL and GPU_API_TOKEN)}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")))
