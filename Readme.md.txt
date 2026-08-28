
========================================================================
PROYECTO: Implamentación de la estrategia Causal Training a la PDE Allen-Cahn. 🚀

### DESCRIPCIÓN

Este proyecto implementa la estrategia "Causal Training" propuesta por Sifan Wang (2022), el cual tiene por objetivo mitigar la violación de la causalidad en las PINNs
al resolver PDEs evolutivas, se propone la ecuación de Allen-Cahn como problema de referencia (Allen_Cahn.py). 
La violación de la causalidad es una patología que presentan la PINNs estándar, bajo el marco propuesto por Raissi (2019). 

### 📋REQUISITOS DEL SISTEMA

* Python 3.8 o superior
* Git

Librerías de Python requeridas: 

* torch
* numpy 
* matplotlib

### ⚙️INSTALACIÓN Y CONFIGURACIÓN

1. Clonar el repositorio:
git clone https://github.com/cruzgamamiguelangel-debug/Causal_Training_Allen_Cahn.git
2. Ir al directorio del proyecto:
cd Causal_Training_Allen-Cahn
3. (Opcional pero recomendado) Crear un entorno virtual:
python -m venv venv 

Activar entorno en Windows:
venv\Scripts\activate 

Activar entorno en Mac/Linux:
source venv/bin/activate
4. Instalar las dependencias necesarias:
pip install -r requirements.txt
(Nota: Si no hay archivo requirements.txt, instalar manualmente con pip install [librerías])

### 💻CÓMO EJECUTAR EL CÓDIGO

Para iniciar el proceso de entrenamiento, ejecuta el siguiente comando en la terminal: 

python Allen_Cahn.py 

### 📋ESTRUCTURA DEL ARCHIVO

* Allen_Cahn.py : Script principal que contiene la lógica del modelo y el bucle de entrenamiento.
* Readme.txt           : Este archivo con las instrucciones de uso.

### CONTACTO /AUTOR

Desarrollado por: Miguel Ángel Cruz Gama 
Contacto: miguel.cruz@inaoe.mx
GitHub: https://github.com/cruzgamamiguelangel-debug/Causal_Training_Allen_Cahn.git