# titanic-ml

This repo uses the Kaggle [titanic](https://www.kaggle.com/c/titanic) dataset to explore various machine learning techniques to find an optimal solution.

The project is divided into 2 main folders - [`src`](./src) and [`notebooks`](./notebooks). The `notebooks` folder contains the Jupyter notebooks that were used to explore the data and models, while `src` folder contains definition for supporting Python classes.

## Notebooks

1. [titanic_explore.ipynb](./notebooks/titanic_explore.ipynb) - This notebook explores the dataset and comes up with different encoding strategies for various fields in the data.
2. [titanic_models.ipynb](./notebooks/titanic_models.ipynb) - This notebook explores various models to find the one that gives the best result. It also uses `GridSearchCV` and `RandomizedGridSearch` to find the optimal parameters.

## Python classes

1. [transformers.py](./src/modules/transformers.py) - Since we want to use pandas `DataFrame` object to view and explore data, we had to implement custom transformers. These transformers are a basic passthrough for underlying `sklearn` transformers. This was done because `sklearn` transformers return numpy arrays which lose a lot of the functionality that was desired for data exploration.

# Final Performance
Using basic sklearn models I was able to attain 81.0055% accuracy. Which resulted in 76.076% accuracy of the Kaggle submission.
