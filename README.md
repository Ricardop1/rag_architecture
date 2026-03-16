# ARQUITECTURA RAG PARA QA DOCUMENTAL 

Este repositorio contiene una solución propuesta de una arquitectura RAG para QA Documental.

## Contenido

- **/data**: carpeta data con los pdfs usados en la solución. En este caso se han usado documntos públicos de la regulación 2026 de la FIA para Formula 1 (https://www.fia.com/regulation/category/110).

- **/src**: carpeta que contiene el código python usado en la solución. Está compuesta por los archivos:
    - **ingest.py**: script que lee los pdfs de la carpeta /data y crea la base de datos vecrtorial con los chunks creados.
    - **answer.py**: script que recibe una pregunta y responde usando un modelo LLM y la información contenida en los archivos pdf.


## Cómo Ejecutar

- Para ejecutar, en primer lugar debemos instalar las dependecias necesárias a través del archivo requirements.txt y el comando ```pip install -r requirements.txt```.

- Para la creación de la base de datos vectorial, ejecutaremos el comando ```python src/ingest.py``` que leerá todos los archivos pdf de la carpeta /data, obtendrá el texto de estos y creará los chunkings a insertar en la base de datos.

- La ejecución se realiza mediante el comando ```python src/answer.py --query "<query>"``` donde query será la pregunta que queremos realizar a nuestro modelo. Por ejemplo:
    ```bash
    python src/answer.py --query "What are the F1 Teams shutdown periods on the year 2013?"
    ```


## Decisiones Técnicas


## Métricas


## Manejo de Documentos
