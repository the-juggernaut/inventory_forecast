# Inventory Demand Forecasting System

This project addresses a common challenge in supply chain management: optimizing inventory levels in the face of fluctuating demand, variable supplier lead times, and compliance constraints. The goal is to build a robust sales forecasting pipeline that supports smarter inventory decisions, reduces holding costs, and mitigates the risk of stockouts or overstocking.

Inconsistent stock availability and missed sales opportunities 
have underscored the need for a more data-driven, proactive approach to demand forecasting. This project explores predictive modeling techniques to support inventory planning for a Walmart store.

## Takeaways

This was my first time working with time series data, and it gave me hands-on experience with the challenges of forecasting under uncertainty. I chose to look at this task as an ML task and not a deployment task. Hence, I explored a mix of models to understand their strengths and trade-offs

Due to limited time and compute (Colab wasn’t enough for LSTM training), I couldn’t fully explore deep learning. Still, the project highlighted the value of good feature engineering and gave me a solid start in demand forecasting and inventory planning.

Please feel free to share feedback to [jagannathabhay@gmail.com](mailto:jagannathabhay@gmail.com)

## Dataset

**Source**: [M5 Forecasting - Accuracy (Kaggle)](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data)

The raw dataset includes:

* Daily unit sales per item, per store
* Calendar features (holidays, events)
* Sell prices and temporal pricing info

### Preprocessing Steps

To prepare the dataset for modeling, I picked store (`CA_3`) because it ahd the highest sales volume. The forecasting is done on this store's data. (See `store_filter.ipynb`.)

I wrote a feature engineering module (`feature_extractor.py`) generate a rich set of predictive features, including:

* **Lag features** (7, 14, 28 days)
* **Rolling means and standard deviations**
* **Sell price volatility**
* **Time-based features** (day-of-week, month, year)
* **One-hot encoded holidays and events**

---
## Models Implemented

Three models were implemented and compared:

1. **Prophet**

   * Prophet was selected for its ease of use and solid performance on seasonal retail data.
   * It handles holidays and changepoints with minimal configuration, making it a useful first benchmark.

2. **LightGBM**

   * A powerful gradient boosting algorithm well-suited for structured time series data.
   * I chose LightGBM due to its ability to capture non-linear patterns from engineered features and external regressors like price, holidays, and time lags.

3. **LSTM**
   * The LSTM model was more complex and slower to train, but I wanted to test its ability to learn hidden patterns from the sequential sales data without heavy manual feature crafting.
   * However, I found it challenging to preprocess and shape the data correctly to feed into the model. As a result, the LSTM underperformed compared to the other models.
   * The implementation would likely benefit from better feature tuning, architectural adjustments, and possibly longer training cycles.

---

## Results Summary

I evaluated the models using Root Mean Square Error (RMSE) and Mean Absolute Percentage Error (MAPE):

| Model    | RMSE                      | MAPE   |
| -------- | ------------------------- | ------ |
| LightGBM | 3.71                      | 55.21% |
| Prophet  | 469.64                    | 5.78% |
| LSTM     | Incomplete |  NA      |


While Prophet and LightGBM achieved comparable results, I found LightGBM to offer more flexibility in incorporating engineered features. 

---

## Improvements


Future versions of this system could benefit from the following enhancements:

* Scale from single-store to multi-store, multi-region forecasting
* A full inventory optimization with multi-head forecasting over fields other than demand.
* Integrate automated inventory decisions based on forecasts
* Add new  deep learning models such  as N-BEATS or Temporal Convolutional Networks
* Establishing an API for 
   * Retraining based on real-time sales feedback
   * Building in regulatory compliance logic and optimization objectives
   * Improve observability (dashboards, logging, and monitoring tools)
---
