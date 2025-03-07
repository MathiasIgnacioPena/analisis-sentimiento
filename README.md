# API-Youtube-Project

## Descripción

Este proyecto tiene como objetivo analizar datos de varios canales de YouTube utilizando la API de YouTube. Se recopilan estadísticas de los canales, detalles de los videos y comentarios, y se realizan análisis de sentimientos sobre los comentarios. Además, se generan visualizaciones para entender mejor los datos recopilados.

## Funcionalidades

1. **Recopilación de datos de canales**: Obtiene estadísticas de los canales como el nombre del canal, número de suscriptores, vistas totales, número de videos y la lista de reproducción de subida.

2. **Recopilación de IDs de videos**: Obtiene una lista de IDs de videos de una lista de reproducción específica.

3. **Recopilación de detalles de videos**: Obtiene estadísticas detalladas de los videos como el título, descripción, etiquetas, fecha de publicación, número de vistas, likes, comentarios, duración, definición y subtítulos.

4. **Recopilación de comentarios**: Obtiene los comentarios principales de los videos.

5. **Análisis de sentimientos**: Realiza un análisis de sentimientos sobre los comentarios utilizando la librería VADER de NLTK.

6. **Visualización de datos**: Genera gráficos y nubes de palabras para visualizar las estadísticas de los canales y los comentarios.

## Requisitos

- Python 3.x
- Librerías: pandas, numpy, dateutil, isodate, matplotlib, seaborn, googleapiclient, nltk, wordcloud

## Instalación

1. Clona el repositorio:
    ```bash
    git clone https://github.com/tu-usuario/API-Youtube-Project.git
    ```

2. Instala las dependencias:
    ```bash
    pip install -r requirements.txt
    ```

## Uso

1. Obtén una clave de API de YouTube y reemplaza `api_key` en el archivo `main.py` con tu clave de API.

2. Ejecuta el script `main.py` para recopilar datos y realizar análisis:
    ```bash
    python src/main.py
    ```

3. Los datos recopilados se guardarán en archivos CSV para referencia futura.

## Estructura del Proyecto

```
API-Youtube-Project/
│
├── data/
│   ├── raw/
│   │   ├── video_data_top10_channels.csv
│   │   └── comments_data_top10_channels.csv
│
├── src/
│   ├── main.py
│
├── .venv/
│
├── README.md
│
└── requirements.txt
```

## Contribuciones

Las contribuciones son bienvenidas. Por favor, abre un issue o un pull request para discutir cualquier cambio que te gustaría realizar.

## Licencia

Este proyecto está licenciado bajo la Licencia MIT. Consulta el archivo LICENSE para más detalles.
