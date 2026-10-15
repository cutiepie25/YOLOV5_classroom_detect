# YOLOV5_classroom_detect

Proyecto de **Minería Multimedia**: reconoce en tiempo real a los estudiantes de un salón de clase con **YOLOv5** y transmite la cámara a una interfaz web hecha con **Flask** (streaming MJPEG).

Cuando el modelo reconoce a un estudiante, la interfaz muestra una puerta animada que se abre, un mensaje de bienvenida (`¡Bienvenido, <nombre>!`) y un registro de los eventos de detección.

## Características

- Detección personalizada de 9 clases (estudiantes): Juanes, Andrés, Mateo, Maria, Roberto, Santiago Arias, Jhon, Santiago Mendoza y Miguel.
- Servidor web Flask con video en vivo, FPS, número de detecciones y estado de la puerta.
- Funciona en CPU; si hay GPU disponible usa CUDA.
- Si no encuentra `best.pt`, usa los pesos preentrenados `yolov5s.pt`.

## Estructura

```
├── main.py          # Servidor Flask + hilo de captura e inferencia YOLOv5
├── best.pt          # Pesos entrenados con el dataset de estudiantes
├── yolov5s.pt       # Pesos preentrenados de YOLOv5s (respaldo)
├── estudiantes/     # Dataset en formato YOLO
│   ├── images/{train,val,test}
│   └── labels/{train,val,test}
└── yolov5/          # Copia del repositorio de Ultralytics YOLOv5
```

El dataset tiene 27 imágenes de entrenamiento, 5 de validación y 7 de prueba.

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate        # En Windows: .venv\Scripts\activate
pip install -r yolov5/requirements.txt
pip install flask
```

## Uso

```bash
python main.py                         # Usa best.pt (o yolov5s.pt de respaldo)
python main.py --weights yolov5s.pt    # Pesos personalizados
python main.py --source 1              # Índice de la cámara (por defecto 0)
python main.py --conf 0.3              # Umbral de confianza (por defecto 0.1)
python main.py --img-size 640          # Tamaño de imagen para la inferencia
python main.py --port 8080             # Puerto del servidor (por defecto 5000)
```

Abre <http://localhost:5000> en el navegador.

### Endpoints

| Ruta          | Descripción                                                      |
|---------------|------------------------------------------------------------------|
| `/`           | Interfaz web principal                                           |
| `/video_feed` | Stream MJPEG con las detecciones dibujadas                       |
| `/status`     | JSON con FPS, número de detecciones, estado de la puerta y persona detectada |

## Entrenamiento

El modelo `best.pt` se entrenó con el script de YOLOv5 sobre el dataset `estudiantes/`. Ejemplo:

```bash
cd yolov5
python train.py --img 640 --batch 16 --epochs 100 --data <ruta/al/dataset.yaml> --weights ../yolov5s.pt
```

> El archivo `dataset.yaml` no está en el repositorio. Debe apuntar a `estudiantes/images/{train,val,test}` y definir las 9 clases en el mismo orden que `CLASES_ESTUDIANTES` en `main.py`.

## Créditos

- [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5) (licencia AGPL-3.0, ver `yolov5/LICENSE`).
