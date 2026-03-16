# ARQUITECTURA RAG PARA QA DOCUMENTAL 

Este repositorio contiene una solución propuesta de una arquitectura RAG para QA Documental.

## Contenido

- **/data**: carpeta data con los pdfs usados en la solución. En este caso se han usado documntos públicos de la regulación 2026 de la FIA para la Formula 1 (https://www.fia.com/regulation/category/110).

- **/src**: carpeta que contiene el código python usado en la solución. Está compuesta por los archivos:
    - **ingest.py**: script que lee los pdfs de la carpeta /data y crea la base de datos vecrtorial con los chunks creados.
    - **answer.py**: script que recibe una pregunta y responde usando un modelo LLM y la información contenida en los archivos pdf.


## Cómo Ejecutar

- Para ejecutar, en primer lugar debemos instalar las dependecias necesárias a través del archivo requirements.txt y el comando ```pip install -r requirements.txt```.

- Para la creación de la base de datos vectorial, ejecutaremos el comando ```python src/ingest.py``` que leerá todos los archivos pdf de la carpeta /data, obtendrá el texto de estos y creará los chunkings a insertar en la base de datos.

- La ejecución se realiza mediante el comando ```python src/answer.py --query "<query>"``` donde query será la pregunta que queremos realizar a nuestro modelo. Para el uso del modelo de Huggingface es necesário exportar la variable de entorno **HF_TOKEN** con el token propio de autenticación. Un ejemplo de ejecución:
    ```bash
    python src/answer.py --query "What are the F1 Teams shutdown periods on the year 2013?"
    ```


## Decisiones Técnicas

Debido al caracter de challenge técnico, para esta arquitectura se han usado solamente herramientas y modelos open source de forma a mantener la simplicidad. Aunque permiten obtener una solución de forma testear una arquitectura no son herramientas que usaria en un entorno Productivo y más adelante se explican alternativas. 

Como herramientas usadas tenemos:
 - **Langchain** como framework de trabajo para la orquestación de todas las herramientas;
 - **ChromaDb** como base de datos vectorial debido a su simplicidad y facil integración con langchain y un entorno local.
 - **multilingual-e5-base** como modelo de embedding. Modelo conocido dentro de los rankings de huggingface que permite buenos resultados aun siendo un modelo sencillo.
 - **bge-reranker-v2-m3** como reranker debido a su soporte multi lenguaje y buena valoración aun siendo un reranker ligero;
 - **Qwen3.5-9B** como modelo LLM, usado desde un punto de Inferencia de Hugginface debido a sus buenas capacidades y perfecto para pruebas en un entorno local.

No obstante a continuación se propone una arquitectura más enfocada a un entorno productivo. 

En primer lugar, como framework podriamos seguir usando langchain ya que contiene herramientas bastante versatiles y poderosas que ayudan a la hora de mantener una buena orquestación de varios modelos, templates de prompts, conexiones a bases de datos vectoriales, etc pero sugeriria tambien el uso de langgraph de forma a poder crear un agente más completo. Langgraph nos ayudaria a crear flujos de ejecución más complejos en el caso de querer ejecutar secuencias de prompts (analisis de texto, extracción, normalizacion de valores, etc) o ejecución de pasos condicionales. 

En cuanto a la base de datos vectorial, aunque Chromadb podría servir en un sistema de RAG pequeño o mediano, sugeriria PostgreSQL, con la extensión PGVector debido a su facilidad de uso, escalibilidad y versatilidad al ser una base datos clásica que añade ahora el soporte a vectores y queries de similitud semantica, o una base de datos vectorial especializada como Milvus que contiene herramientas potentes para la busqueda semantica cuando nos encontramos ante un alto némero de vectores y de grande dimensionalidad.

Como modelo embedding podría ser viable el uso de multilingual-e5-base pero si nos encontramos ante un caso de uso de un grande número de documentos, queremos usar un modelo que permita crear embeddings de altas dimensiones. 
En el caso de tambien querer almacenar no sólo documentos pero tambien imagenes, videos o o algun otro tipo de archivo, podriamos contar con el nuevo embedding multimodal Gemini Embedding 2 que podria permitir nuevos casos de uso

En cuanto al reranker, realizando una investigación ha sido posible ver que tanto para un entorno local como para un entorno productivo, bge-reranker-v2-m3 consiste en una solución muy buena a tener en cuenta por su soporte de multi idioma y ligereza.

En cuanto a modelo LLM a usar, mi elección serían los modelos Gemini de Google ya que consisten en unos modelos muy buenos para el análisis de documentos y presentan una grande ventana de contexto que permitiria el analisis o inclusión de un mayor número de textos. 

-----

Comentar que esta arquitectura recomendada podría variar según el entorno de despliegue (proveedor de cloud, despliegue en cluster kubernetes, uso de un entorno local) o el presupuesto que se pretende usar. 

En el caso de contar ya de una solución con un cluster Kubernetes, podriamos realizar un despliegue de una api con flask donde ejecutariamos nuestro agente.

## Métricas

Para poder realizar una validación sencilla del sistema, se incluye un archivo **eval.jsonl** con un pequeño conjunto de preguntas, respuestas esperadas y pasajes fuente esperados. A partir de este archivo, el script **test.py** ejecuta una evaluación automática recorriendo cada pregunta del dataset.

Para cada entrada de **eval.jsonl**, el flujo es el siguiente:

- se lee la pregunta;
- se genera una respuesta usando el LLM y únicamente el contexto recuperado a través de los top_k resultados más similares;
- se comparan tanto la respuesta generada como las fuentes recuperadas frente a los valores esperados del dataset;
- se guarda el resultado en el archivo **eval_results.jsonl**.

El archivo **eval_results.jsonl** contiene una línea JSON por cada caso evaluado. En cada línea se almacena, entre otros campos:

- el identificador de la pregunta;
- la pregunta evaluada;
- la respuesta generada por el sistema;
- la respuesta esperada;
- las fuentes recuperadas por el RAG;
- las fuentes esperadas;
- las métricas calculadas para ese caso.

Las métricas implementadas son:

- **exact_match**: comprueba si la respuesta generada coincide exactamente con la respuesta esperada tras una normalización básica del texto. Esta normalización convierte el texto a minúsculas, elimina signos de puntuación y espacios duplicados. Es una métrica estricta y útil cuando esperamos respuestas muy concretas.

- **token_f1**: calcula el solapamiento entre los tokens de la respuesta generada y los de la respuesta esperada. Esta métrica permite detectar respuestas parcialmente correctas aunque no estén redactadas exactamente igual.

- **source_hit**: comprueba si al menos una de las fuentes recuperadas por el sistema coincide con alguna de las fuentes esperadas definidas en el dataset, comparando **pdf_name** y **page**. Esta métrica permite validar la parte de recuperación del RAG, independientemente de la redacción final del modelo.
