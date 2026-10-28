import os, uuid, subprocess
from pathlib import Path
from flask import Flask, render_template, request, send_file, redirect, url_for, flash
from werkzeug.utils import secure_filename

BASE = Path(__file__).resolve().parent
UPLOADS, OUTPUTS = BASE / "uploads", BASE / "outputs"
UPLOADS.mkdir(exist_ok=True); OUTPUTS.mkdir(exist_ok=True)
ALLOWED = {"png", "jpg", "jpeg", "webp"}
MODE = os.getenv("GENERATOR_MODE", "demo").lower()
COMMAND = os.getenv("GENERATOR_COMMAND", "").strip()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "change-this-in-production")
app.config["MAX_CONTENT_LENGTH"] = int(os.getenv("MAX_UPLOAD_MB", "12")) * 1024 * 1024

def demo_obj(path):
    # DEMO ONLY: cube mesh to test download. Not reconstructed from the image.
    verts = [(-1,-1,-1),(1,-1,-1),(1,1,-1),(-1,1,-1),(-1,-1,1),(1,-1,1),(1,1,1),(-1,1,1)]
    faces = [(1,4,3),(1,3,2),(5,6,7),(5,7,8),(1,2,6),(1,6,5),(2,3,7),(2,7,6),(3,4,8),(3,8,7),(4,1,5),(4,5,8)]
    with path.open("w", encoding="utf-8") as f:
        f.write("# DEMO ONLY: cube, not generated from uploaded image\no DemoCube\n")
        for v in verts: f.write("v %s %s %s\n" % v)
        for face in faces: f.write("f %d %d %d\n" % face)

def real_generator(src, dst):
    if not COMMAND:
        raise RuntimeError("Falta configurar GENERATOR_COMMAND para conectar el motor de IA 3D.")
    cmd = COMMAND.format(input=str(src), output=str(dst))
    result = subprocess.run(cmd, shell=True, cwd=str(BASE), capture_output=True, text=True,
                            timeout=int(os.getenv("GENERATION_TIMEOUT_SECONDS", "900")))
    if result.returncode or not dst.exists() or dst.stat().st_size == 0:
        raise RuntimeError("El generador no pudo crear el OBJ. " + (result.stderr or result.stdout or "")[-1800:])

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("image")
        if not file or not file.filename:
            flash("Selecciona una imagen para continuar.", "error"); return redirect(url_for("index"))
        ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
        if ext not in ALLOWED:
            flash("Formato no admitido. Usa PNG, JPG, JPEG o WebP.", "error"); return redirect(url_for("index"))
        job = uuid.uuid4().hex
        src = UPLOADS / (job + "_" + (secure_filename(file.filename) or "imagen"))
        dst = OUTPUTS / (job + ".obj")
        file.save(src)
        try:
            if MODE == "demo":
                demo_obj(dst)
                status = "Modo demostración: se creó un cubo OBJ para comprobar la descarga. No es una reconstrucción de la imagen."
            else:
                real_generator(src, dst)
                status = "Modelo OBJ generado correctamente."
            return render_template("result.html", filename=dst.name, status=status, demo=(MODE == "demo"))
        except Exception as e:
            app.logger.exception("Fallo al generar modelo")
            flash(str(e), "error"); return redirect(url_for("index"))
    return render_template("index.html", max_mb=os.getenv("MAX_UPLOAD_MB", "12"))

@app.route("/download/<filename>")
def download(filename):
    if not filename.endswith(".obj") or Path(filename).name != filename:
        flash("Archivo no válido.", "error"); return redirect(url_for("index"))
    path = OUTPUTS / filename
    if not path.is_file():
        flash("El archivo ya no está disponible.", "error"); return redirect(url_for("index"))
    return send_file(path, as_attachment=True, download_name="modelo_3d.obj", mimetype="text/plain")

@app.route("/health")
def health():
    return {"status": "ok", "generator_mode": MODE}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")))
