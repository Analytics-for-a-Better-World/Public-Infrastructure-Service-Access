# Public Infrastructure Service Access (PISA)

PISA assists policymakers in identifying and planning key resources such as roads, bridges, and healthcare centres. It helps to optimise the placement of new infrastructure investments. Its goal is to increase access to public services, reduce expenses, and improve the quality of life in a region. 

## Documentation

For detailed documentation on the code, read the [docs](https://analytics-for-a-better-world.github.io/Public-Infrastructure-Service-Access/index.html).

> ⚠️ **Important:**  
> The latest version of PISA contains significant refactoring and improvements.  
> If you need the original codebase, use version `0.1.0` available from the [GitHub releases](https://github.com/Analytics-for-a-Better-World/Public-Infrastructure-Service-Access/releases/tag/v0.1.0) or refer to the [list of releases](https://github.com/Analytics-for-a-Better-World/Public-Infrastructure-Service-Access/releases) for more information.  
> For new projects and ongoing development, it is **recommended to use the latest version** to benefit from updated features and optimizations.


## Local Development

### Installation

This project can be installed in two ways:

#### Option 1: Using Poetry (Recommended)

This project uses [Poetry](https://python-poetry.org/) for dependency and packaging management.

1. Install Poetry by following the [official installation instructions](https://python-poetry.org/docs/#installation)
2. Clone the repository: `git clone https://github.com/yourusername/Public-Infrastructure-Service-Access.git`
3. Navigate to the repository folder: `cd Public-Infrastructure-Service-Access`
4. Install the project and its dependencies: `poetry install`

#### Option 2: Using pip and venv

If you prefer traditional Python package management:

1. Clone the repository: `git clone https://github.com/yourusername/Public-Infrastructure-Service-Access.git`
2. Navigate to the repository folder: `cd Public-Infrastructure-Service-Access`
3. Create a virtual environment: `python -m venv venv`
4. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - macOS/Linux: `source venv/bin/activate`
5. Install the package: `pip install dist/pisa-2.0.0-py3-none-any.whl`

> If you need the original codebase, use the wheel file for version `0.1.0` (`pip install dist/gpbp-0.1.0-py3-none-any.whl`)


#### External Dependencies

PISA requires additional tools for certain functionality:

1. **Optimization Solver**: Used to optimize the location of new facilities.
   - For beginners, we recommend the [COIN-OR Branch-and-Cut solver (CBC)](https://github.com/coin-or/Cbc#download)
   - Note the path to the executable of `cbc` for configuration
   - Advanced users can substitute CBC with other solvers compatible with the package

2. **Mapbox API** (Optional): Used to generate facility isopolygons.
   - [Sign up for a free Mapbox account](https://www.mapbox.com/signup/)
   - Get your API key from the Mapbox account dashboard

After installing `cbc` and/or creating a Mapbox API key, create a `.env` file in the repository root with the following settings:

```sh
# Path to your cbc solver executable (for optimization)
CBC_SOLVER_PATH=<path_to_solver>

# Mapbox API key (optional, for isopolygon generation)
MAPBOX_API_TOKEN=<your_api_key_here>
```

This file is ignored by git by default for security.

### Usage

We provide two examples to understand and interact with the package: 

1. Some example usage can be seen in the `examples/pisa_showcase_*.ipynb` notebooks. These notebooks show the basic package data flow and allow you to explore the outputs of each function. The notebooks differ in the tool used to generate isopolygons: OSM or Mapbox. Note that Mapbox requires an API key (see External Dependencies section above).
2. A more visual example can be seen in the [Streamlit](https://streamlit.io/) app to interact with the package using a graphical interface. You can start the app by running `streamlit run pisa_app/main_page.py` from the main repository directory, which will automatically open a browser window.

## Deploying/running the web interface (Docker)

Docker is a tool that allows you to package an application and its dependencies in a virtual container that can run on most operating systems. The web interface for the PISA project can be run using Docker. To do this, you need to have Docker installed on your system. You can download Docker from [here](https://www.docker.com/products/docker-desktop).

After you have installed Docker, you can run the following command in the repository to build an image of the PISA project. The Dockerfile uses Poetry to install the application and its dependencies inside the image, consistent with the recommended installation method, and includes all necessary geospatial libraries.

```sh
docker build -t pisa .
```

Now that you have built the image, you can run a container using it. The following command will run a container from the `pisa` image and expose the web interface on port 8501 to localhost only. The command will show the URL on which it's accessible. The container will be named `pisa` as well.

```sh
docker run -p 127.0.0.1:8501:8501 --name pisa pisa
```

You can use the following commands to manage the container after the initial run:

```sh
# Stop the container
docker stop pisa

# Start a stopped container
docker start pisa

# View logs from the container
docker logs pisa

# Remove the container (if you need to start fresh)
docker rm pisa
```

The image is also ready to be deployed on a server or a cloud platform that supports Docker.

## References
- [Travel Distance Calculations in Python](https://pythoncharmers.com/blog/travel-distance-python-with-geopandas-folium-alphashape-osmnx-buffer.html)
- [Geocoding Services in Python](https://towardsdatascience.com/comparison-of-geocoding-services-applied-to-stroke-care-facilities-in-vietnam-with-python-ff0ba753a590)
- [Visualising Global Population Datasets in Python](https://towardsdatascience.com/visualising-global-population-datasets-with-python-c87bcfc8c6a6)
- [GPBP Publications - Temporary GitHub Repository](https://github.com/Analytics-for-a-Better-World/GPBP_Analytics_Tools)
- [Installation and Usage of Openroute Service](https://giscience.github.io/openrouteservice/installation/Installation-and-Usage.html)

