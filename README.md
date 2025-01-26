# Public Infrastructure Service Access

## Local Development

### Installation
1. Clone the repository
2. This project uses [Poetry](https://python-poetry.org/) for dependency and packaging management. You can use these [installation instructions](https://python-poetry.org/docs/#installation) to add it to your system.
  - If you choose to use poetry you can execute the following commands in a terminal while you are located in the repository folder:
  ```poetry install```
  This will create a virtual environment and install all the necessary library dependencies.
  - If you **do not** choose to use poetry we first recommend creating a virtual environment and
  then install the package and necessary dependencies by installing the **.whl** file in the **dist** folder. So in a terminal, while you are located in the repository folder, execute the following:
    - ```python -m venv venv``` to create the virtual environment
    - Activate the environment with ```venv\Scripts\Activate.ps1``` (on Windows) or ```source venv/bin/activate``` (on MacOS)
    - ```cd dist``` to enter the dist folder
    - ```pip install gpbp-0.2.0-py3-none-any.whl``` to install the package

### Usage
1. Some example usage can be seen in the ```examples/gpbp_showcase.ipynb``` notebook
2. When in the ```gpbp_app``` folder you can run a [Streamlit](https://streamlit.io/) app to use
the package using an interface. Specifically, while you are located in the repository folder, execute the following:
  - ```cd gpbp_app``` to enter the application folder
  - ```streamlit run main_page.py``` to run the app, which will automatically open a browser window

## Deploying/running the web interface (Docker)

Docker is a tool that allows you to package an application and its dependencies in a virtual container that can run on most operating systems. The web interface for the PISA project can be run using Docker. To do this, you need to have Docker installed on your system. You can download Docker from [here](https://www.docker.com/products/docker-desktop).

After you have installed Docker, you can run the following command in the repository to build an image of the PISA project. All the necessary dependencies will be installed inside the image, isolated from your system. The image will be tagged as `pisa`.

```sh
docker build -t pisa .
```

Now that you have built the image, you can run a container using it. The following command will run a container from the `pisa` image and expose the web interface on port 8501 to localhost only. The command will show the URL on which it's accessible. The container will be named `pisa` as well.

```sh
docker run -p 127.0.0.1:8501:8501 --name pisa pisa
```

You can use the following commands after running the container for the first time:

```sh
docker stop pisa
docker start pisa
docker logs pisa
```

The image is also ready to be deployed on a server or a cloud platform that supports Docker.

## References
- [Travel Distance Calculations in Python](https://pythoncharmers.com/blog/travel-distance-python-with-geopandas-folium-alphashape-osmnx-buffer.html)
- [Geocoding Services in Python](https://towardsdatascience.com/comparison-of-geocoding-services-applied-to-stroke-care-facilities-in-vietnam-with-python-ff0ba753a590)
- [Visualising Global Population Datasets in Python](https://towardsdatascience.com/visualising-global-population-datasets-with-python-c87bcfc8c6a6)
- [GPBP Publications - Temporary GitHub Repository](https://github.com/Analytics-for-a-Better-World/GPBP_Analytics_Tools)
- [Installation and Usage of Openroute Service](https://giscience.github.io/openrouteservice/installation/Installation-and-Usage.html)

