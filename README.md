# Machine Learning Project To Predict and Understand the Factors Affecting or Associated with Country Level Cereal Yields

## Introduction
This project aims to understand what macro-level factors may affect or be associated with cereal yields. These yields specifically refer to the kg of production per hectare and as such reflect to what extent a country is able to extract the highest amount of cereal from a given area. This is an important issue given that we are already exploiting the large majority of the global arable land with major expansions still possible in highly sensitive areas such as the amazon and other rainforests and biodiversity hotspots. By understanding and modelling the yield we might be able to understand what factors could be drivers of improved yields and better predict the response in yield to certain macro level changes such as temperature change, economic development and areas of investment. 

## Summary of Project
This project used a typical time series model with a 5 year horizon but with a number of economic and geographic variables included alongisde the lag of the yield to try improve the model and capture which variables might have important effects on the global changes in yield. Lags were created for each country but the model trained on all countries to try learn patterns at a global level. The model used was a RandomForestRegressor as part of a sklearn pipeline. The most important variable by some distance was found to be the 5 year lag of the yield but model performance on test data was found to improve with the inclusion of the other variables. Given the short nature of this project, there are a number of routes for improvement such as:

- trying ARIMA and ARIMAX models
- finding other potentially important variables.
- carrying out more feature engineering
- analysing countries separately or in groups based on similar economic status or climate.
- 

## Data Overviews

- The target is the 'Cereal yield (kg per hectare)' taken from the World Bank [https://data.worldbank.org/indicator]. This contains annual yield values by country.
- Variables explored largely came from the World Bank as well and included:
    - Year
    - Investment in transport with private participation (current US$)
    - Employment in agriculture, male (% of male employment) (modeled ILO estimate)
    - Employment in agriculture, female (% of female employment) (modeled ILO estimate)
    - Arable land (% of land area)
    - Agriculture, forestry, and fishing, value added (% of GDP)
    - Renewable internal freshwater resources, total (billion cubic meters)
    - Arable land (hectares per person)
    - Permanent cropland (% of land area)
    - Energy intensity level of primary energy (MJ/$2017 PPP GDP)
    - Annual freshwater withdrawals, total (billion cubic meters)
    - Fertilizer consumption (kilograms per hectare of arable land)
    - Government expenditure on education, total (% of GDP)
    - Agricultural machinery, tractors per 100 sq. km of arable land
    - Total natural resources rents (% of GDP)
    - Agricultural irrigated land (% of total agricultural land)
- Variables from other sources used were:
    - Country latitude and logitude
    - HistoricalGDP per capita [https://ourworldindata.org/grapher/world-gdp-over-the-last-two-millennia]
    - Historical anual average temperature
