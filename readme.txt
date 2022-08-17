# Readme para utilización de la aplicación ARdetector ⭐

La aplicación de ARdetecor, ha sido creada para el análisis de de arritmias utilizando como datos de entrada la base de datos de arritmias del MIT-BIH. Esta aplicación se basa en el estudio y extracción de las características de la señal
ECG para posteriormente analizar sus anomalias con una red neuronal convolucional, previamente entrenada.

## Preparación entorno 🔥

Para la ejecución y desarrollo de la aplicación, en el caso personal se ha utilizado VS-code, pero se puede ejecutar en otro entorno de desarrollo. Los puntos necesarios para la ejecución del proyecto son:

* **Instalar Python:** Lo primero es intalar el interprete de Python. <br>

    Instalación de Python: <br>
    > sudo apt update
    > sudo apt install python3-dev python3-pip python3-venv


*  **Inlcusión de paquetes:** Los paquetes necesarios para la ejecución del proyecto. <br>

    Tensoflow <br>
    Scypi <br>
    numpy <br>
    wfdb <br>
    matplotlib <br>
    neurokit2 <br>
    sklearn <br>



## Ejecución de ARdetector

Para ejecutar el proyecto, primero se ejecuta desde la terminal el archivo de código "/modeldetector.py" el cual contiene la red neuronal y se encaragará de compilarla, entrenarla y generar los pesos y modelo, que se alojaran en la carpeta "/modelo". 
Una vez han sido generados los pesos y modelo, pasamos a ejecutar el archivo "/app.py" dónde se encuentra la aplicación.