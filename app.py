import os
import uuid
import secrets
from pathlib import Path

import requests
from flask import Flask, render_template, request, send_file, redirect, url_for, flash
from werkzeug.utils import secure_filename

BASE = Path(__file__).resolve().parent
UPLOADS, OUTPUTS = BASE / "uploads", BASE / "outputs"
UPLOADS.mkdir(exist_ok=True)
OUTPUTS.mkdir(exist_ok=True)

ALLOWED = {"png", "jpg", "jpeg", "webp"}
MODE = os.getenv("GENERATOR_MODE", "remote").lower()
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
            if MODE == "demo":
                demo_obj(dst)
                status = "Modo demostración: se creó un cubo OBJ. No es una reconstrucción de la imagen."
            elif MODE == "remote":
                remote_generator(src, dst)
                status = "Modelo 3D generado por TripoSR correctamente."
            else:
                raise RuntimeError("GENERATOR_MODE no válido. Usa 'remote' o 'demo'.")
            return render_template("result.html", filename=dst.name, status=status, demo=(MODE == "demo"))
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


@app.route("/descargar-cuaderno-colab")
def descargar_cuaderno_colab():
    """Descarga el cuaderno interactivo para ejecutar TripoSR en Google Colab."""
    notebook = BASE / "notebooks" / "3D-ST-TripoSR-Colab.ipynb"
    if not notebook.is_file():
        flash("No se encontró el cuaderno de Google Colab.", "error")
        return redirect(url_for("index"))
    return send_file(
        notebook,
        as_attachment=True,
        download_name="3D-ST-TripoSR-Colab.ipynb",
        mimetype="application/x-ipynb+json",
    )


@app.route("/health")
def health():
    return {"status": "ok", "generator_mode": MODE, "gpu_configured": bool(GPU_API_URL and GPU_API_TOKEN)}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")))
