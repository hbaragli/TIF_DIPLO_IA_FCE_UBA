import os
import shutil
from dotenv import load_dotenv

def collect_pdfs(src_dir: str, dest_dir: str) -> None:
    os.makedirs(dest_dir, exist_ok=True)
    for root, _, files in os.walk(src_dir):
        for f in files:
            if f.lower().endswith('.pdf'):
                src = os.path.join(root, f)
                dst = os.path.join(dest_dir, f)
                # Evitar sobreescritura
                if os.path.exists(dst):
                    name, ext = os.path.splitext(f)
                    i = 1
                    while True:
                        candidate = os.path.join(dest_dir, f"{name}_{i}{ext}")
                        if not os.path.exists(candidate):
                            dst = candidate
                            break
                        i += 1
                shutil.copy2(src, dst)
                print(f"Copiado: {src} → {dst}")

if __name__ == "__main__":
    # Carga de env
    load_dotenv()
    ORIGEN  = os.getenv("ORIGEN_PDF")
    DESTINO = os.getenv("DESTINO_PDF")


    if not ORIGEN or not DESTINO:
        raise RuntimeError(
            "DEFINIR LAS VARIABLES ORIGEN_PDF y DESTINO_PDF EN EL ARCHIVO DE CONFIGURACIÓN"
        )

    collect_pdfs(ORIGEN, DESTINO)


    