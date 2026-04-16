"""
=============================================================================
  Minería Multimedia — Detección de Estudiantes en Tiempo Real con YOLOv5
  Versión Web (Flask + MJPEG Streaming)
=============================================================================

  Descripción:
    Servidor web que transmite la webcam en tiempo real con detección
    YOLOv5, puerta animada y mensajes de bienvenida. Accesible desde
    el navegador en http://localhost:5000

  Uso:
    python main.py                         # Usa yolov5s.pt por defecto
    python main.py --weights best.pt       # Usa pesos personalizados
    python main.py --source 0              # Índice de cámara
    python main.py --conf 0.3              # Umbral de confianza
    python main.py --port 8080             # Puerto personalizado

  Requisitos:
    pip install torch torchvision opencv-python numpy flask

  Notas:
    - YOLOv5 debe estar clonado en ./yolov5/
    - Compatible con CPU (no requiere GPU)
    - Abre http://localhost:5000 en el navegador para ver la cámara
=============================================================================
"""

import sys
import os
import time
import argparse
import math
import threading
import numpy as np
import cv2

# ── Agregar yolov5 al path ───────────────────────────────────────────────────
YOLOV5_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yolov5")
if YOLOV5_DIR not in sys.path:
    sys.path.insert(0, YOLOV5_DIR)

import torch
from flask import Flask, Response, render_template_string, jsonify


# =============================================================================
#  CONFIGURACIÓN GLOBAL
# =============================================================================

CLASES_ESTUDIANTES = [
    "Juanes", "Andrés", "Mateo", "Maria", "Roberto",
    "Santiago Arias", "Jhon", "Santiago Mendoza", "Miguel",
]

COLORES_CLASES = [
    (255, 100, 60),    # Juanes
    (50, 205, 50),     # Andrés
    (0, 165, 255),     # Mateo
    (255, 0, 255),     # Maria
    (0, 255, 255),     # Roberto
    (255, 255, 0),     # Santiago Arias
    (80, 127, 255),    # Jhon
    (147, 20, 255),    # Santiago Mendoza
    (0, 215, 255),     # Miguel
]

MENSAJE_COOLDOWN = 5.0
PUERTA_ANIM_DURACION = 0.8
PUERTA_ABIERTA_TIEMPO = 4.0


# =============================================================================
#  CLASE: Estado de la Puerta Animada
# =============================================================================

class PuertaAnimada:
    """Simula una puerta con animación de apertura/cierre."""

    def __init__(self):
        self.estado = "cerrada"
        self.progreso = 0.0
        self.tiempo_inicio_anim = 0.0
        self.ultima_deteccion = 0.0
        self.nombre_actual = ""
        self.color_actual = (200, 200, 200)

    def detectar_persona(self, nombre: str, color: tuple):
        """Notifica que se detectó una persona conocida."""
        self.ultima_deteccion = time.time()
        self.nombre_actual = nombre
        self.color_actual = color
        if self.estado in ("cerrada", "cerrando"):
            self.estado = "abriendo"
            self.tiempo_inicio_anim = time.time()

    def resetear(self):
        """Resetea la puerta al estado cerrado."""
        self.estado = "cerrando"
        self.tiempo_inicio_anim = time.time()
        self.nombre_actual = ""

    def actualizar(self):
        """Actualiza el estado y la animación de la puerta cada frame."""
        ahora = time.time()
        if self.estado == "abriendo":
            elapsed = ahora - self.tiempo_inicio_anim
            self.progreso = min(1.0, elapsed / PUERTA_ANIM_DURACION)
            self.progreso = 1.0 - (1.0 - self.progreso) ** 3
            if self.progreso >= 0.99:
                self.progreso = 1.0
                self.estado = "abierta"
        elif self.estado == "abierta":
            if ahora - self.ultima_deteccion > PUERTA_ABIERTA_TIEMPO:
                self.estado = "cerrando"
                self.tiempo_inicio_anim = ahora
        elif self.estado == "cerrando":
            elapsed = ahora - self.tiempo_inicio_anim
            t = min(1.0, elapsed / PUERTA_ANIM_DURACION)
            t = t ** 3
            self.progreso = 1.0 - t
            if self.progreso <= 0.01:
                self.progreso = 0.0
                self.estado = "cerrada"
                self.nombre_actual = ""

    def dibujar(self, frame: np.ndarray):
        """Dibuja la puerta animada en el frame."""
        h, w = frame.shape[:2]

        puerta_w = int(w * 0.20)
        puerta_h = int(h * 0.50)
        puerta_x = w - puerta_w - 25
        puerta_y = (h - puerta_h) // 2

        # Marco
        grosor_marco = 4
        cv2.rectangle(frame,
            (puerta_x - grosor_marco, puerta_y - grosor_marco),
            (puerta_x + puerta_w + grosor_marco, puerta_y + puerta_h + grosor_marco),
            (80, 80, 80), grosor_marco)

        # Interior
        brillo = int(60 * self.progreso)
        interior = (40 + brillo, 50 + brillo, 30 + brillo)
        cv2.rectangle(frame, (puerta_x, puerta_y),
            (puerta_x + puerta_w, puerta_y + puerta_h), interior, -1)

        # Hojas de la puerta
        mitad = puerta_w // 2
        apertura = int(mitad * self.progreso)

        if self.estado in ("abriendo", "abierta"):
            r = int(140 + (self.color_actual[0] - 140) * self.progreso * 0.4)
            g = int(110 + (self.color_actual[1] - 110) * self.progreso * 0.4)
            b = int(80 + (self.color_actual[2] - 80) * self.progreso * 0.4)
            color_puerta = (max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b)))
        else:
            color_puerta = (140, 110, 80)

        # Hoja izquierda
        hoja_izq_x2 = puerta_x + mitad - apertura
        if hoja_izq_x2 > puerta_x:
            cv2.rectangle(frame, (puerta_x, puerta_y),
                (hoja_izq_x2, puerta_y + puerta_h), color_puerta, -1)
            cv2.rectangle(frame, (puerta_x, puerta_y),
                (hoja_izq_x2, puerta_y + puerta_h), (100, 80, 55), 2)
            manija_x = hoja_izq_x2 - 8
            if manija_x > puerta_x + 5:
                cv2.circle(frame, (manija_x, puerta_y + puerta_h // 2), 5, (200, 200, 200), -1)

        # Hoja derecha
        hoja_der_x1 = puerta_x + mitad + apertura
        if hoja_der_x1 < puerta_x + puerta_w:
            cv2.rectangle(frame, (hoja_der_x1, puerta_y),
                (puerta_x + puerta_w, puerta_y + puerta_h), color_puerta, -1)
            cv2.rectangle(frame, (hoja_der_x1, puerta_y),
                (puerta_x + puerta_w, puerta_y + puerta_h), (100, 80, 55), 2)
            manija_x = hoja_der_x1 + 8
            if manija_x < puerta_x + puerta_w - 5:
                cv2.circle(frame, (manija_x, puerta_y + puerta_h // 2), 5, (200, 200, 200), -1)

        # Letrero
        font = cv2.FONT_HERSHEY_SIMPLEX
        letrero = "MINERIA MULTIMEDIA"
        (tw, th), _ = cv2.getTextSize(letrero, font, 0.45, 1)
        lx = puerta_x + (puerta_w - tw) // 2
        ly = puerta_y - 12
        cv2.rectangle(frame, (lx - 6, ly - th - 6), (lx + tw + 6, ly + 6), (50, 50, 50), -1)
        cv2.rectangle(frame, (lx - 6, ly - th - 6), (lx + tw + 6, ly + 6), (100, 100, 100), 1)
        cv2.putText(frame, letrero, (lx, ly), font, 0.45, (220, 220, 220), 1, cv2.LINE_AA)

        # Estado
        estados = {"cerrada": ("CERRADA", (0, 0, 200)), "abriendo": ("ABRIENDO...", (0, 200, 255)),
                   "abierta": ("ABIERTA", (0, 220, 0)), "cerrando": ("CERRANDO...", (0, 150, 255))}
        txt, col = estados.get(self.estado, ("", (200, 200, 200)))
        (tw2, _), _ = cv2.getTextSize(txt, font, 0.45, 1)
        cv2.putText(frame, txt, (puerta_x + (puerta_w - tw2) // 2, puerta_y + puerta_h + 22),
                    font, 0.45, col, 1, cv2.LINE_AA)

        # LED pulsante
        led_x = puerta_x + puerta_w // 2
        led_y = puerta_y + puerta_h + 38
        if self.estado in ("abriendo", "cerrando"):
            r = int(5 + 3 * abs(math.sin(time.time() * 5)))
        else:
            r = 6
        cv2.circle(frame, (led_x, led_y), r + 2, (60, 60, 60), -1)
        cv2.circle(frame, (led_x, led_y), r, col, -1)


# =============================================================================
#  FUNCIONES AUXILIARES
# =============================================================================

def cargar_modelo(weights_path: str):
    """Carga el modelo YOLOv5 desde el repo local."""
    print(f"[INFO] Cargando modelo desde: {weights_path}")
    modelo = torch.hub.load(YOLOV5_DIR, "custom", path=weights_path,
                            source="local", force_reload=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    modelo.to(device)
    print(f"[INFO] Dispositivo: {device}")
    print(f"[INFO] Clases del modelo: {modelo.names}")
    return modelo


def obtener_color_clase(clase_id: int) -> tuple:
    """Retorna el color BGR asignado a una clase."""
    if 0 <= clase_id < len(COLORES_CLASES):
        return COLORES_CLASES[clase_id]
    return (200, 200, 200)


def obtener_nombre_clase(modelo, clase_id: int) -> str:
    """Retorna el nombre de la clase."""
    if hasattr(modelo, "names") and clase_id in modelo.names:
        return modelo.names[clase_id]
    if 0 <= clase_id < len(CLASES_ESTUDIANTES):
        return CLASES_ESTUDIANTES[clase_id]
    return f"Clase_{clase_id}"


def es_clase_conocida(nombre_clase: str) -> bool:
    """Verifica si la clase detectada es un estudiante conocido o 'person'."""
    if nombre_clase in CLASES_ESTUDIANTES:
        return True
    if nombre_clase.lower() == "person":
        return True
    return False


def dibujar_detecciones(frame, resultados, modelo):
    """Dibuja bounding boxes y retorna la mejor detección conocida."""
    detecciones = resultados.xyxy[0].cpu().numpy()
    mejor_nombre, mejor_confianza, mejor_color = None, 0.0, None

    for det in detecciones:
        x1, y1, x2, y2, conf, cls_id = det
        x1, y1, x2, y2, cls_id = int(x1), int(y1), int(x2), int(y2), int(cls_id)
        nombre = obtener_nombre_clase(modelo, cls_id)
        color = obtener_color_clase(cls_id)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        etiqueta = f"{nombre} {conf:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(etiqueta, font, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
        cv2.putText(frame, etiqueta, (x1 + 3, y1 - 5), font, 0.6,
                    (255, 255, 255), 2, cv2.LINE_AA)

        if es_clase_conocida(nombre) and conf > mejor_confianza:
            mejor_confianza = conf
            mejor_nombre = nombre
            mejor_color = color

    return mejor_nombre, mejor_confianza, mejor_color


def dibujar_mensaje_bienvenida(frame, nombre, alpha=1.0):
    """Dibuja un mensaje de bienvenida con fade en la parte inferior."""
    h, w = frame.shape[:2]
    mensaje = f"Bienvenido a Mineria Multimedia, {nombre}!"
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(mensaje, font, 0.8, 2)
    msg_x = (w - tw) // 2
    msg_y = h - 50
    pad = 15

    overlay = frame.copy()
    cv2.rectangle(overlay, (msg_x - pad, msg_y - th - pad),
                  (msg_x + tw + pad, msg_y + pad), (30, 30, 30), -1)
    cv2.rectangle(overlay, (msg_x - pad, msg_y - th - pad),
                  (msg_x + tw + pad, msg_y + pad), (0, 220, 120), 2)
    cv2.addWeighted(overlay, alpha * 0.85, frame, 1 - alpha * 0.85, 0, frame)

    color_texto = (int(120 + 135 * alpha), int(255 * alpha), int(120 + 100 * alpha))
    cv2.putText(frame, mensaje, (msg_x, msg_y), font, 0.8,
                color_texto, 2, cv2.LINE_AA)


def dibujar_info_hud(frame, fps, modelo_nombre, num_det, puerta_estado):
    """Dibuja información HUD en la esquina superior izquierda."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    y_off = 22
    lineas = [
        f"FPS: {fps:.1f}",
        f"Modelo: {modelo_nombre}",
        f"Detecciones: {num_det}",
        f"Puerta: {puerta_estado.upper()}",
    ]
    overlay = frame.copy()
    cv2.rectangle(overlay, (5, 5), (230, len(lineas) * (y_off + 2) + 10), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    for i, linea in enumerate(lineas):
        cv2.putText(frame, linea, (12, 22 + i * y_off), font, 0.45,
                    (180, 180, 180), 1, cv2.LINE_AA)


def resolver_pesos(weights_arg: str) -> str:
    """Resuelve ruta de pesos con fallback de best.pt a yolov5s.pt."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.isabs(weights_arg) and os.path.isfile(weights_arg):
        return weights_arg
    for ruta in [os.path.join(script_dir, weights_arg),
                 os.path.join(YOLOV5_DIR, weights_arg), weights_arg]:
        if os.path.isfile(ruta):
            return os.path.abspath(ruta)
    if "best" in weights_arg:
        print(f"[AVISO] No se encontró '{weights_arg}'. Buscando yolov5s.pt...")
        fallback = os.path.join(script_dir, "yolov5s.pt")
        if os.path.isfile(fallback):
            print(f"[INFO] Usando respaldo: {fallback}")
            return fallback
    print(f"[AVISO] No se encontró '{weights_arg}'. Se intentará descargar.")
    return weights_arg


# =============================================================================
#  PLANTILLA HTML — Interfaz Web Premium
# =============================================================================

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Minería Multimedia — Detección de Estudiantes</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        *, *::before, *::after {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        :root {
            --bg-primary: #0a0e1a;
            --bg-secondary: #111827;
            --bg-card: #1a1f35;
            --bg-glass: rgba(26, 31, 53, 0.7);
            --accent-primary: #6366f1;
            --accent-secondary: #8b5cf6;
            --accent-glow: rgba(99, 102, 241, 0.3);
            --accent-green: #10b981;
            --accent-green-glow: rgba(16, 185, 129, 0.3);
            --accent-red: #ef4444;
            --accent-yellow: #f59e0b;
            --text-primary: #f1f5f9;
            --text-secondary: #94a3b8;
            --text-muted: #64748b;
            --border-color: rgba(99, 102, 241, 0.15);
            --border-glow: rgba(99, 102, 241, 0.4);
        }

        body {
            font-family: 'Inter', -apple-system, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
            overflow-x: hidden;
        }

        /* ── Fondo animado con gradientes ── */
        body::before {
            content: '';
            position: fixed;
            top: 0; left: 0;
            width: 100%; height: 100%;
            background: 
                radial-gradient(ellipse at 20% 50%, rgba(99, 102, 241, 0.08) 0%, transparent 50%),
                radial-gradient(ellipse at 80% 20%, rgba(139, 92, 246, 0.06) 0%, transparent 50%),
                radial-gradient(ellipse at 50% 80%, rgba(16, 185, 129, 0.05) 0%, transparent 50%);
            z-index: 0;
            pointer-events: none;
        }

        /* ── Header ── */
        .header {
            position: relative;
            z-index: 10;
            padding: 1.2rem 2rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: var(--bg-glass);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border-bottom: 1px solid var(--border-color);
        }

        .header-left {
            display: flex;
            align-items: center;
            gap: 1rem;
        }

        .logo-icon {
            width: 42px;
            height: 42px;
            background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary));
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.3rem;
            box-shadow: 0 4px 15px var(--accent-glow);
        }

        .header-title h1 {
            font-size: 1.25rem;
            font-weight: 700;
            background: linear-gradient(135deg, var(--text-primary), var(--accent-secondary));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .header-title p {
            font-size: 0.75rem;
            color: var(--text-muted);
            font-weight: 400;
        }

        .status-badge {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.5rem 1rem;
            background: rgba(16, 185, 129, 0.1);
            border: 1px solid rgba(16, 185, 129, 0.3);
            border-radius: 50px;
            font-size: 0.8rem;
            font-weight: 500;
            color: var(--accent-green);
        }

        .status-dot {
            width: 8px;
            height: 8px;
            background: var(--accent-green);
            border-radius: 50%;
            animation: pulse-dot 2s ease-in-out infinite;
        }

        @keyframes pulse-dot {
            0%, 100% { opacity: 1; box-shadow: 0 0 0 0 var(--accent-green-glow); }
            50% { opacity: 0.7; box-shadow: 0 0 0 6px transparent; }
        }

        /* ── Contenido principal ── */
        .main-content {
            position: relative;
            z-index: 5;
            max-width: 1400px;
            margin: 0 auto;
            padding: 1.5rem 2rem;
            display: grid;
            grid-template-columns: 1fr 320px;
            gap: 1.5rem;
        }

        /* ── Video Feed ── */
        .video-container {
            background: var(--bg-card);
            border-radius: 16px;
            border: 1px solid var(--border-color);
            overflow: hidden;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            position: relative;
        }

        .video-container::before {
            content: '';
            position: absolute;
            top: 0; left: 0;
            width: 100%; height: 100%;
            border-radius: 16px;
            background: linear-gradient(135deg, var(--accent-glow), transparent);
            opacity: 0;
            transition: opacity 0.3s ease;
            pointer-events: none;
            z-index: 2;
        }

        .video-container:hover::before {
            opacity: 0.1;
        }

        .video-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0.8rem 1.2rem;
            background: rgba(0, 0, 0, 0.3);
            border-bottom: 1px solid var(--border-color);
        }

        .video-header-left {
            display: flex;
            align-items: center;
            gap: 0.6rem;
        }

        .rec-indicator {
            display: flex;
            align-items: center;
            gap: 0.4rem;
            font-size: 0.7rem;
            font-weight: 600;
            color: var(--accent-red);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .rec-dot {
            width: 8px;
            height: 8px;
            background: var(--accent-red);
            border-radius: 50%;
            animation: blink 1s steps(1) infinite;
        }

        @keyframes blink {
            0% { opacity: 1; }
            50% { opacity: 0; }
        }

        .video-label {
            font-size: 0.8rem;
            color: var(--text-secondary);
            font-weight: 500;
        }

        .video-frame {
            width: 100%;
            display: block;
            background: #000;
            aspect-ratio: 16/9;
            object-fit: contain;
        }

        /* ── Panel lateral ── */
        .side-panel {
            display: flex;
            flex-direction: column;
            gap: 1.2rem;
        }

        .card {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 14px;
            padding: 1.2rem;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.15);
            transition: border-color 0.3s ease, box-shadow 0.3s ease;
        }

        .card:hover {
            border-color: var(--border-glow);
            box-shadow: 0 4px 20px rgba(99, 102, 241, 0.1);
        }

        .card-title {
            font-size: 0.7rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: var(--text-muted);
            margin-bottom: 0.8rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .card-title .icon {
            font-size: 0.9rem;
        }

        /* ── Info del modelo ── */
        .model-info .info-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.45rem 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.04);
        }

        .model-info .info-row:last-child {
            border-bottom: none;
        }

        .info-label {
            font-size: 0.78rem;
            color: var(--text-secondary);
        }

        .info-value {
            font-size: 0.78rem;
            font-weight: 600;
            color: var(--text-primary);
        }

        .info-value.highlight {
            color: var(--accent-primary);
        }

        /* ── Deteccion actual ── */
        .detection-display {
            text-align: center;
            padding: 0.5rem 0;
        }

        .detection-name {
            font-size: 1.5rem;
            font-weight: 800;
            background: linear-gradient(135deg, var(--accent-green), #34d399);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 0.3rem;
            opacity: 0;
            transform: translateY(10px);
            transition: all 0.4s ease;
        }

        .detection-name.active {
            opacity: 1;
            transform: translateY(0);
        }

        .detection-conf {
            font-size: 0.75rem;
            color: var(--text-muted);
        }

        .no-detection {
            color: var(--text-muted);
            font-size: 0.85rem;
            font-style: italic;
        }

        /* ── Puerta estado ── */
        .door-status {
            display: flex;
            align-items: center;
            gap: 0.8rem;
        }

        .door-visual {
            width: 50px;
            height: 70px;
            border: 2px solid var(--text-muted);
            border-radius: 4px;
            position: relative;
            overflow: hidden;
            background: var(--bg-secondary);
            flex-shrink: 0;
        }

        .door-leaf-left, .door-leaf-right {
            position: absolute;
            top: 0;
            width: 50%;
            height: 100%;
            background: linear-gradient(135deg, #8b7355, #a0896b);
            transition: transform 0.8s cubic-bezier(0.22, 1, 0.36, 1);
        }

        .door-leaf-left { left: 0; transform-origin: left; }
        .door-leaf-right { right: 0; transform-origin: right; }

        .door-leaf-left.open { transform: scaleX(0.1); }
        .door-leaf-right.open { transform: scaleX(0.1); }

        .door-info {
            flex: 1;
        }

        .door-state {
            font-size: 0.85rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.2rem;
        }

        .door-state.cerrada { color: var(--accent-red); }
        .door-state.abriendo { color: var(--accent-yellow); }
        .door-state.abierta { color: var(--accent-green); }
        .door-state.cerrando { color: var(--accent-yellow); }

        .door-person {
            font-size: 0.75rem;
            color: var(--text-secondary);
        }

        /* ── Log de eventos ── */
        .event-log {
            max-height: 200px;
            overflow-y: auto;
            scrollbar-width: thin;
            scrollbar-color: var(--accent-primary) transparent;
        }

        .event-log::-webkit-scrollbar {
            width: 4px;
        }

        .event-log::-webkit-scrollbar-thumb {
            background: var(--accent-primary);
            border-radius: 4px;
        }

        .event-item {
            display: flex;
            gap: 0.6rem;
            padding: 0.5rem 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.03);
            animation: slideIn 0.3s ease;
        }

        @keyframes slideIn {
            from { opacity: 0; transform: translateX(-10px); }
            to { opacity: 1; transform: translateX(0); }
        }

        .event-time {
            font-size: 0.65rem;
            color: var(--text-muted);
            font-family: 'JetBrains Mono', monospace;
            white-space: nowrap;
            padding-top: 2px;
        }

        .event-text {
            font-size: 0.75rem;
            color: var(--text-secondary);
            line-height: 1.4;
        }

        .event-text strong {
            color: var(--accent-green);
            font-weight: 600;
        }

        /* ── Clases conocidas ── */
        .class-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0.4rem;
        }

        .class-chip {
            padding: 0.35rem 0.6rem;
            border-radius: 8px;
            font-size: 0.68rem;
            font-weight: 500;
            background: rgba(99, 102, 241, 0.08);
            border: 1px solid rgba(99, 102, 241, 0.15);
            color: var(--text-secondary);
            text-align: center;
            transition: all 0.3s ease;
        }

        .class-chip.detected {
            background: rgba(16, 185, 129, 0.15);
            border-color: rgba(16, 185, 129, 0.4);
            color: var(--accent-green);
        }

        /* ── Responsive ── */
        @media (max-width: 900px) {
            .main-content {
                grid-template-columns: 1fr;
                padding: 1rem;
            }
            .header {
                padding: 1rem;
                flex-wrap: wrap;
                gap: 0.5rem;
            }
        }

        /* ── Welcome banner ── */
        .welcome-banner {
            position: fixed;
            bottom: -100px;
            left: 50%;
            transform: translateX(-50%);
            padding: 1rem 2.5rem;
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.9), rgba(5, 150, 105, 0.9));
            backdrop-filter: blur(20px);
            border-radius: 16px;
            border: 1px solid rgba(255, 255, 255, 0.2);
            box-shadow: 0 8px 32px rgba(16, 185, 129, 0.3);
            z-index: 100;
            transition: bottom 0.6s cubic-bezier(0.22, 1, 0.36, 1);
            text-align: center;
        }

        .welcome-banner.show {
            bottom: 2rem;
        }

        .welcome-banner h2 {
            font-size: 1.1rem;
            font-weight: 700;
            color: #fff;
            margin-bottom: 0.2rem;
        }

        .welcome-banner p {
            font-size: 0.8rem;
            color: rgba(255, 255, 255, 0.8);
        }
    </style>
</head>
<body>

    <!-- ── Header ── -->
    <header class="header">
        <div class="header-left">
            <div class="logo-icon">🎓</div>
            <div class="header-title">
                <h1>Minería Multimedia</h1>
                <p>Sistema de Detección de Estudiantes — YOLOv5</p>
            </div>
        </div>
        <div class="status-badge">
            <span class="status-dot"></span>
            <span id="statusText">EN VIVO</span>
        </div>
    </header>

    <!-- ── Contenido Principal ── -->
    <main class="main-content">

        <!-- Video Feed -->
        <div class="video-container">
            <div class="video-header">
                <div class="video-header-left">
                    <div class="rec-indicator">
                        <span class="rec-dot"></span>
                        REC
                    </div>
                    <span class="video-label">Cámara en Vivo — Detección YOLOv5</span>
                </div>
                <span class="video-label" id="fpsDisplay">-- FPS</span>
            </div>
            <img src="/video_feed" class="video-frame" alt="Video Feed" id="videoFeed">
        </div>

        <!-- Panel Lateral -->
        <aside class="side-panel">

            <!-- Info del modelo -->
            <div class="card model-info">
                <div class="card-title"><span class="icon">⚙️</span> Modelo</div>
                <div class="info-row">
                    <span class="info-label">Pesos</span>
                    <span class="info-value highlight" id="modelName">{{ model_name }}</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Dispositivo</span>
                    <span class="info-value" id="deviceInfo">{{ device }}</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Confianza</span>
                    <span class="info-value" id="confThreshold">{{ conf }}</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Detecciones</span>
                    <span class="info-value" id="detCount">0</span>
                </div>
            </div>

            <!-- Detección actual -->
            <div class="card">
                <div class="card-title"><span class="icon">👤</span> Detección Actual</div>
                <div class="detection-display">
                    <div class="detection-name" id="detName">—</div>
                    <div class="detection-conf" id="detConf"></div>
                    <div class="no-detection" id="noDetection">Esperando detección...</div>
                </div>
            </div>

            <!-- Estado de la puerta -->
            <div class="card">
                <div class="card-title"><span class="icon">🚪</span> Estado de la Puerta</div>
                <div class="door-status">
                    <div class="door-visual">
                        <div class="door-leaf-left" id="doorLeft"></div>
                        <div class="door-leaf-right" id="doorRight"></div>
                    </div>
                    <div class="door-info">
                        <div class="door-state cerrada" id="doorState">CERRADA</div>
                        <div class="door-person" id="doorPerson">Sin acceso</div>
                    </div>
                </div>
            </div>

            <!-- Clases conocidas -->
            <div class="card">
                <div class="card-title"><span class="icon">📋</span> Estudiantes Registrados</div>
                <div class="class-grid" id="classGrid">
                    {% for name in class_names %}
                    <div class="class-chip" id="chip-{{ loop.index0 }}">{{ name }}</div>
                    {% endfor %}
                </div>
            </div>

            <!-- Log de eventos -->
            <div class="card">
                <div class="card-title"><span class="icon">📜</span> Registro de Accesos</div>
                <div class="event-log" id="eventLog">
                    <div class="event-item">
                        <span class="event-time">--:--:--</span>
                        <span class="event-text">Sistema iniciado. Esperando detecciones...</span>
                    </div>
                </div>
            </div>

        </aside>
    </main>

    <!-- Banner de bienvenida -->
    <div class="welcome-banner" id="welcomeBanner">
        <h2 id="welcomeText">¡Bienvenido!</h2>
        <p>Acceso concedido — Minería Multimedia</p>
    </div>

    <script>
        // Polling para obtener estado actualizado
        let lastDetection = '';
        let welcomeTimeout = null;

        async function fetchStatus() {
            try {
                const res = await fetch('/status');
                const data = await res.json();

                // FPS
                document.getElementById('fpsDisplay').textContent = data.fps.toFixed(1) + ' FPS';

                // Detecciones
                document.getElementById('detCount').textContent = data.num_detections;

                // Puerta
                const doorState = document.getElementById('doorState');
                doorState.textContent = data.door_state.toUpperCase();
                doorState.className = 'door-state ' + data.door_state;

                const isOpen = data.door_state === 'abierta' || data.door_state === 'abriendo';
                document.getElementById('doorLeft').className = 'door-leaf-left' + (isOpen ? ' open' : '');
                document.getElementById('doorRight').className = 'door-leaf-right' + (isOpen ? ' open' : '');

                // Detección actual
                const detName = document.getElementById('detName');
                const detConf = document.getElementById('detConf');
                const noDetection = document.getElementById('noDetection');

                if (data.current_name) {
                    detName.textContent = data.current_name;
                    detName.classList.add('active');
                    detConf.textContent = 'Confianza: ' + (data.current_conf * 100).toFixed(1) + '%';
                    noDetection.style.display = 'none';
                    document.getElementById('doorPerson').textContent = data.current_name;

                    // Resaltar chip de clase
                    document.querySelectorAll('.class-chip').forEach(c => c.classList.remove('detected'));
                    const classNames = {{ class_names | tojson }};
                    const idx = classNames.indexOf(data.current_name);
                    if (idx >= 0) {
                        const chip = document.getElementById('chip-' + idx);
                        if (chip) chip.classList.add('detected');
                    }

                    // Banner de bienvenida
                    if (data.current_name !== lastDetection) {
                        lastDetection = data.current_name;
                        showWelcome(data.current_name);
                        addEvent(data.current_name, data.current_conf);
                    }
                } else {
                    detName.classList.remove('active');
                    noDetection.style.display = 'block';
                    detConf.textContent = '';
                    if (data.door_state === 'cerrada') {
                        document.getElementById('doorPerson').textContent = 'Sin acceso';
                        lastDetection = '';
                        document.querySelectorAll('.class-chip').forEach(c => c.classList.remove('detected'));
                    }
                }

            } catch (e) {
                console.error('Error fetching status:', e);
            }
        }

        function showWelcome(name) {
            const banner = document.getElementById('welcomeBanner');
            document.getElementById('welcomeText').textContent = '¡Bienvenido, ' + name + '!';
            banner.classList.add('show');

            if (welcomeTimeout) clearTimeout(welcomeTimeout);
            welcomeTimeout = setTimeout(() => {
                banner.classList.remove('show');
            }, 4000);
        }

        function addEvent(name, conf) {
            const log = document.getElementById('eventLog');
            const now = new Date();
            const timeStr = now.toLocaleTimeString('es-CO');
            const item = document.createElement('div');
            item.className = 'event-item';
            item.innerHTML = `
                <span class="event-time">${timeStr}</span>
                <span class="event-text"><strong>${name}</strong> detectado (${(conf*100).toFixed(1)}%)</span>
            `;
            log.insertBefore(item, log.firstChild);
            // Limitar a 20 eventos
            while (log.children.length > 20) {
                log.removeChild(log.lastChild);
            }
        }

        // Polling cada 500ms
        setInterval(fetchStatus, 500);
        fetchStatus();
    </script>
</body>
</html>
"""


# =============================================================================
#  ESTADO COMPARTIDO (thread-safe)
# =============================================================================

class AppState:
    """Estado compartido entre el hilo de captura y el servidor Flask."""
    def __init__(self):
        self.lock = threading.Lock()
        self.frame = None
        self.fps = 0.0
        self.num_detections = 0
        self.door_state = "cerrada"
        self.current_name = None
        self.current_conf = 0.0

    def update(self, frame, fps, num_det, door_state, name, conf):
        with self.lock:
            self.frame = frame.copy() if frame is not None else None
            self.fps = fps
            self.num_detections = num_det
            self.door_state = door_state
            self.current_name = name
            self.current_conf = conf

    def get_frame(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def get_status(self):
        with self.lock:
            return {
                "fps": self.fps,
                "num_detections": self.num_detections,
                "door_state": self.door_state,
                "current_name": self.current_name,
                "current_conf": self.current_conf,
            }


# Estado global
app_state = AppState()


# =============================================================================
#  HILO DE CAPTURA Y PROCESAMIENTO
# =============================================================================

def hilo_captura(source, modelo, img_size):
    """Hilo que captura frames de la webcam y ejecuta inferencia YOLOv5."""
    global app_state

    print(f"[INFO] Abriendo cámara (source={source})...")
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print("[ERROR] No se pudo abrir la cámara.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    ancho = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Resolución de cámara: {ancho}x{alto}")
    modelo_nombre = os.path.basename(resolver_pesos(args.weights))

    puerta = PuertaAnimada()
    ultimo_mensaje_tiempo = 0.0
    mostrar_bienvenida = False
    nombre_bienvenida = ""
    bienvenida_alpha = 0.0

    fps = 0.0
    frame_count = 0
    fps_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[AVISO] No se pudo leer frame.")
            time.sleep(0.1)
            continue

        # Inferencia
        resultados = modelo(frame, size=img_size)
        nombre, confianza, color = dibujar_detecciones(frame, resultados, modelo)
        num_det = len(resultados.xyxy[0])

        ahora = time.time()

        # Lógica de la puerta
        current_name = None
        current_conf = 0.0

        if nombre is not None:
            puerta.detectar_persona(nombre, color if color else (0, 220, 0))
            current_name = nombre
            current_conf = confianza

            if ahora - ultimo_mensaje_tiempo > MENSAJE_COOLDOWN:
                mostrar_bienvenida = True
                nombre_bienvenida = nombre
                ultimo_mensaje_tiempo = ahora
                bienvenida_alpha = 0.0
                print(f"[DETECCIÓN] {nombre} (conf: {confianza:.2f})")

        puerta.actualizar()
        puerta.dibujar(frame)

        # Mensaje de bienvenida
        if mostrar_bienvenida:
            dt = ahora - ultimo_mensaje_tiempo
            if dt < 0.5:
                bienvenida_alpha = min(1.0, dt / 0.5)
            elif dt < (MENSAJE_COOLDOWN - 1.0):
                bienvenida_alpha = 1.0
            elif dt < MENSAJE_COOLDOWN:
                bienvenida_alpha = max(0.0, 1.0 - (dt - (MENSAJE_COOLDOWN - 1.0)))
            else:
                mostrar_bienvenida = False
                bienvenida_alpha = 0.0
            if bienvenida_alpha > 0.01:
                dibujar_mensaje_bienvenida(frame, nombre_bienvenida, bienvenida_alpha)

        # HUD
        frame_count += 1
        if frame_count % 10 == 0:
            fps = 10.0 / (time.time() - fps_time + 1e-9)
            fps_time = time.time()
        dibujar_info_hud(frame, fps, modelo_nombre, num_det, puerta.estado)

        # Actualizar estado compartido
        app_state.update(frame, fps, num_det, puerta.estado, current_name, current_conf)

    cap.release()
    print("[INFO] Cámara liberada.")


# =============================================================================
#  APLICACIÓN FLASK
# =============================================================================

app = Flask(__name__)


def generar_frames():
    """Generador de frames MJPEG para el streaming."""
    while True:
        frame = app_state.get_frame()
        if frame is None:
            time.sleep(0.05)
            continue

        # Codificar como JPEG
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ret:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

        # ~30 FPS máximo para el stream
        time.sleep(0.033)


@app.route('/')
def index():
    """Página principal con la interfaz web."""
    device = "CUDA" if torch.cuda.is_available() else "CPU"
    return render_template_string(
        HTML_TEMPLATE,
        model_name=os.path.basename(resolver_pesos(args.weights)),
        device=device,
        conf=f"{args.conf:.0%}",
        class_names=CLASES_ESTUDIANTES,
    )


@app.route('/video_feed')
def video_feed():
    """Endpoint de streaming MJPEG."""
    return Response(
        generar_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/status')
def status():
    """Endpoint JSON con el estado actual."""
    return jsonify(app_state.get_status())


# =============================================================================
#  MAIN
# =============================================================================

def main():
    """Punto de entrada principal."""
    global args

    parser = argparse.ArgumentParser(
        description="Detección de estudiantes — Minería Multimedia (Web)"
    )
    parser.add_argument("--weights", type=str, default="best.pt",
                        help="Ruta a pesos (default: best.pt, fallback: yolov5s.pt)")
    parser.add_argument("--source", type=int, default=0,
                        help="Índice de la cámara (default: 0)")
    parser.add_argument("--conf", type=float, default=0.1,
                        help="Umbral de confianza (default: 0.25)")
    parser.add_argument("--img-size", type=int, default=640,
                        help="Tamaño de imagen para inferencia (default: 640)")
    parser.add_argument("--port", type=int, default=5000,
                        help="Puerto del servidor web (default: 5000)")
    args = parser.parse_args()

    # Resolver pesos
    weights_path = resolver_pesos(args.weights)

    # Cargar modelo
    try:
        modelo = cargar_modelo(weights_path)
    except Exception as e:
        print(f"[ERROR] No se pudo cargar el modelo: {e}")
        sys.exit(1)

    modelo.conf = args.conf
    modelo.iou = 0.45
    modelo.max_det = 20

    # Iniciar hilo de captura
    hilo = threading.Thread(
        target=hilo_captura,
        args=(args.source, modelo, args.img_size),
        daemon=True,
    )
    hilo.start()

    # Iniciar servidor Flask
    print("=" * 60)
    print(f"  🌐 Servidor web iniciado en: http://localhost:{args.port}")
    print(f"  📷 Cámara: source={args.source}")
    print(f"  🤖 Modelo: {os.path.basename(weights_path)}")
    print(f"  Presiona Ctrl+C para detener")
    print("=" * 60)

    app.run(host='0.0.0.0', port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
