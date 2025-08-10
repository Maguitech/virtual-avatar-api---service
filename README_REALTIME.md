# 🤖 Sistema de Avatar en Tiempo Real

Un sistema completo para conversación en tiempo real con avatares generados por IA que combina transcripción de voz, generación de avatares y streaming en vivo.

## 🌟 Características

- **🎤 Captura de audio en tiempo real** desde el navegador
- **📝 Transcripción automática** de voz a texto 
- **🎭 Generación de avatares** usando el sistema LiteAvatar
- **🌐 WebSockets** para comunicación en tiempo real
- **🔄 Streaming continuo** de videos de avatar
- **💬 Interfaz web intuitiva** con visualización de audio
- **🎯 API REST** para procesamiento de audio

## 📋 Requisitos

- Python 3.8+
- FFmpeg (para conversión de audio)
- Navegador web moderno con soporte WebSocket
- Micrófono (para captura de audio)

## 🚀 Instalación Rápida

### 1. Instalar dependencias
```bash
python install_realtime_deps.py
```

### 2. Iniciar el sistema completo
```bash
python start_realtime_system.py
```

Esto iniciará automáticamente:
- 🔌 API Server (puerto 8000)
- 🌐 WebSocket Server (puerto 8765) 
- 🖥️ Cliente web en el navegador

## 📖 Instalación Manual

### 1. Instalar dependencias Python
```bash
pip install websockets>=11.0.0
pip install SpeechRecognition>=3.10.0
pip install pydub>=0.25.1
pip install requests>=2.31.0
pip install fastapi>=0.104.0
pip install uvicorn>=0.24.0
pip install python-multipart>=0.0.6
```

### 2. Iniciar servidores por separado

**Terminal 1 - API Server:**
```bash
python avatar_api.py
```

**Terminal 2 - WebSocket Server:**
```bash
python realtime_websocket_avatar.py
```

**Terminal 3 - Cliente Web:**
Abrir `realtime_avatar_client.html` en el navegador

## 🎯 Arquitectura del Sistema

```
┌─────────────────┐    WebSocket    ┌─────────────────┐
│   Cliente Web   │◄──────────────►│ WebSocket Server│
│   (Navegador)   │                 │                 │
└─────────────────┘                 └─────────────────┘
        │                                    │
        │ Audio Chunks                       │ HTTP Requests
        │                                    │
        ▼                                    ▼
┌─────────────────┐                 ┌─────────────────┐
│ Captura Audio   │                 │   Avatar API    │
│ Transcripción   │                 │                 │
└─────────────────┘                 └─────────────────┘
                                             │
                                             │ Processing
                                             ▼
                                    ┌─────────────────┐
                                    │  LiteAvatar     │
                                    │  Generation     │
                                    └─────────────────┘
```

## 🔧 Componentes del Sistema

### 1. **WebSocket Server** (`realtime_websocket_avatar.py`)
- Maneja conexiones WebSocket
- Transcribe audio a texto usando SpeechRecognition
- Envía audio a la API de avatar
- Gestiona el flujo de datos en tiempo real

### 2. **Avatar API** (`avatar_api.py`)
- API REST para generación de avatares
- Soporte para múltiples formatos de audio (MP3, WAV, etc.)
- Conversión automática de audio (16kHz, mono, 16-bit)
- Sistema de trabajos asíncronos

### 3. **Cliente Web** (`realtime_avatar_client.html`)
- Interfaz de usuario interactiva
- Captura de audio desde micrófono
- Visualización en tiempo real
- Reproducción de videos de avatar

## 📡 API Endpoints

### WebSocket (ws://localhost:8765)
- `audio_chunk` - Enviar chunk de audio
- `text_input` - Enviar texto directamente
- `transcription` - Recibir transcripción
- `avatar_video` - Recibir URL del video

### HTTP API (http://localhost:8000)
- `POST /generate` - Generar avatar desde audio
- `GET /status/{job_id}` - Estado del trabajo
- `GET /download/{job_id}` - Descargar video
- `GET /jobs` - Listar trabajos
- `GET /docs` - Documentación Swagger

## 🎮 Uso del Sistema

### 1. **Modo Conversación por Voz**
1. Hacer clic en "Conectar"
2. Hacer clic en "Iniciar Grabación"
3. Hablar al micrófono
4. Ver transcripción en tiempo real
5. El avatar aparecerá hablando tu mensaje

### 2. **Modo Texto Directo**
1. Escribir texto en el campo de entrada
2. Hacer clic en "Enviar Texto"
3. El sistema generará el avatar (requiere TTS)

## 🔧 Configuración

### Variables de entorno
```bash
# WebSocket Server
WEBSOCKET_HOST=localhost
WEBSOCKET_PORT=8765

# Avatar API
API_HOST=localhost  
API_PORT=8000
API_URL=http://localhost:8000
```

### Configuración de audio
```python
# En realtime_websocket_avatar.py
SAMPLE_RATE = 16000  # Hz
CHANNELS = 1         # Mono
SAMPLE_WIDTH = 2     # 16-bit
```

## 🚨 Solución de Problemas

### **Error: No se puede acceder al micrófono**
- Asegurar permisos de micrófono en el navegador
- Usar HTTPS o localhost para acceso a micrófono
- Verificar que el micrófono funcione en otras apps

### **Error: WebSocket no conecta**
- Verificar que el servidor WebSocket esté ejecutándose
- Comprobar el puerto 8765 no esté en uso
- Revisar configuración de firewall

### **Error: Avatar API no responde**
- Verificar que la API esté ejecutándose en puerto 8000
- Comprobar logs del servidor para errores
- Verificar que FFmpeg esté instalado

### **Error: Transcripción no funciona**
- Verificar conexión a internet (Google Speech API)
- Comprobar que el audio tenga suficiente volumen
- Instalar PocketSphinx para transcripción offline

## 🔮 Características Futuras

- [ ] **TTS Integration** - Generar audio desde texto
- [ ] **Multiple Languages** - Soporte multiidioma
- [ ] **Voice Cloning** - Clonar voces específicas
- [ ] **Real-time Mixing** - Mezcla de video en tiempo real
- [ ] **WebRTC Support** - Streaming P2P
- [ ] **Avatar Customization** - Personalización de avatares
- [ ] **Emotion Detection** - Detección de emociones en voz

## 📊 Rendimiento

- **Latencia de audio**: ~100-200ms
- **Generación de avatar**: ~3-10s dependiendo del hardware
- **Transcripción**: ~500ms-2s
- **Streaming**: Tiempo real

## 🤝 Contribución

1. Fork el repositorio
2. Crear rama para feature (`git checkout -b feature/nueva-caracteristica`)
3. Commit cambios (`git commit -am 'Agregar nueva característica'`)
4. Push a la rama (`git push origin feature/nueva-caracteristica`)
5. Crear Pull Request

## 📄 Licencia

MIT License - ver archivo LICENSE para detalles

## 🆘 Soporte

Para soporte y preguntas:
- Crear issue en el repositorio
- Revisar logs de los servidores
- Verificar configuración según este README

---

**¡Disfruta conversando con tu avatar en tiempo real! 🤖✨**