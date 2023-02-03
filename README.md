# Public Infrastructure Service Access

## Installation
1. Clone the repository
2. This project uses [Poetry](https://python-poetry.org/) for dependency and packaging management. You can use these [installation instructions](https://python-poetry.org/docs/#installation) to add it to your system.
  - If you choose to use poetry you can execute the following commands in a terminal while you are located in the repository folder:
  ```bash poetry install```
  This will create a virtual environment and install all the necessary library dependencies.
  - If you **do not** choose to use poetry we first recommend creating a virtual environment and
  then install the package and necessary dependencies by installing the **.whl** file in the **dist** folder. So in a terminal, while you are located in the repository folder, execute the following:
    - ```bash python -m venv venv``` to create the virtual environment
    - ```bash venv\Scripts\Activate.ps1``` to activate it
    - ```bash cd dist``` to enter the dist folder
    - ```bash pip install gpbp-0.1.0-py3-none-any.whl``` to install the package

## Usage
1. Some example usage can be seen in the ```examples/gpbp_showcase.ipynb``` notebook
2. When in the ```gpbp_app``` folder you can run a [Streamlit](https://streamlit.io/) app to use
the package using an interface. Specifically, while you are located in the repository folder, execute the following:
  - ```bash cd gpbp_app``` to enter the application folder
  - ```bash streamlit run main_page.py``` to run the app, which will automatically open a browser window

## References
- [Travel Distance Calculations in Python](https://pythoncharmers.com/blog/travel-distance-python-with-geopandas-folium-alphashape-osmnx-buffer.html)
- [Geocoding Services in Python](https://towardsdatascience.com/comparison-of-geocoding-services-applied-to-stroke-care-facilities-in-vietnam-with-python-ff0ba753a590)
- [Visualising Global Population Datasets in Python](https://towardsdatascience.com/visualising-global-population-datasets-with-python-c87bcfc8c6a6)
- [GPBP Publications - Temporary GitHub Repository](https://github.com/Analytics-for-a-Better-World/GPBP_Analytics_Tools)

