# Inventory Demand Forecasting System

This project addresses a common challenge in supply chain management: optimizing inventory levels in the face of fluctuating demand, variable supplier lead times, and compliance constraints. The goal is to build a robust sales forecasting pipeline that supports smarter inventory decisions, reduces holding costs, and mitigates the risk of stockouts or overstocking.

Inconsistent stock availability and missed sales opportunities have underscored the need for a more data-driven, proactive approach to demand forecasting. This project explores predictive modeling techniques to support inventory planning for a Walmart store.

## Takeaways

This was my first time working with time series data, and it gave me hands-on experience with the challenges of forecasting under uncertainty. I chose to look at this task as an ML task and not a deployment task. Hence, I explored a mix of models to understand their strengths and trade-offs

Due to limited time and compute (Colab wasn’t enough for LSTM training), I couldn’t fully explore deep learning. Still, the project highlighted the value of good feature engineering and gave me a solid start in time series forecasting for inventory planning.

Please feel free to share feedback to [jagannathabhay@gmail.com](mailto:jagannathabhay@gmail.com)

## Dataset

**Source**: [M5 Forecasting - Accuracy (Kaggle)](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data)

The raw dataset includes:

* Daily unit sales per item, per store
* Calendar features (holidays, events)
* Sell prices and temporal pricing info

### Dataset Reductions

To make the dataset manageable and focus on high-impact data, the following reductions were applied:

1. **Store Selection**: Only data from store `CA_3` was used, as it had the highest sales volume.
2. **Day Filtering**: Retained only the last 80% of days to focus on recent trends.
3. **Calendar Filtering**: Limited calendar data to match the filtered days.
4. **Price Filtering**: Reduced sell price data to match the filtered weeks.

These reductions ensured the dataset was optimized for memory usage and computational efficiency while retaining the most relevant information for forecasting.

## Feature Engineering

A custom feature extraction pipeline (`feature_extractor.py`) was developed to generate predictive features, including:

* **Lag Features**: Sales lagged by 7, 14, and 28 days.
* **Rolling Statistics**: Rolling means and standard deviations for windows of 7, 14, and 28 days.
* **Price Volatility**: Percentage change and rolling standard deviation of sell prices.
* **Time-Based Features**: Day-of-week, month, quarter, and year.
* **Event Features**: One-hot encoding for holidays and events.
* **Inventory Metrics**: Stock levels, stockout risk, sell-through rate, and overstock risk.

These features were designed to capture temporal patterns, price sensitivity, and inventory dynamics.

## Models Implemented

The project uses a multi-faceted approach to inventory management, predicting demand and key risk metrics:

1. **LightGBM (Demand Forecasting)**
   * A powerful gradient boosting algorithm well-suited for structured time series data.
   * I chose LightGBM due to its ability to capture non-linear patterns from engineered features and external regressors like price, holidays, and time lags.
   * Outcome: Accurate predictions of daily sales with RMSE and SMAPE metrics.

2. **LightGBM (Stockout Risk Classification)**
   * Identifies products likely to run out of stock.
   * Outcome: High accuracy and ROC AUC scores.

3. **LightGBM (Sell-Through Rate Regression)**
   * Predicts how quickly inventory is sold.
   * Outcome: Insights into inventory turnover.

4. **LightGBM (Overstock Risk Classification)**
   * Flags products at risk of overstocking.
   * Outcome: Helps minimize holding costs and wastage.

5. **Prophet (Demand Forecasting)**
   * Prophet was selected for its ease of use and solid performance on seasonal retail data.
   * It handles holidays and changepoints with minimal configuration, making it a useful first benchmark.

6. **LSTM**
   * The LSTM model was more complex and slower to train, but I wanted to test its ability to learn hidden patterns from the sequential sales data without heavy manual feature crafting.
   * However, I found it challenging to preprocess and shape the data correctly to feed into the model. As a result, the LSTM underperformed compared to the other models.
   * The implementation would likely benefit from better feature tuning, architectural adjustments, and possibly longer training cycles.

## Results Summary

| Model                         | Metric          | Value  | Output           |
| ----------------------------- | --------------- | ------ | ---------------- |
| LightGBM (Demand)             | RMSE            | 0.35   | Demand           |
|                               | SMAPE           | 12.5   |                  |
| LightGBM (Stockout)           | Accuracy        | 0.92   | Stockout Risk    |
|                               | ROC AUC         | 0.88   |                  |
| LightGBM (Sell-Through)       | RMSE            | 0.41   | Sell-Through Rate|
|                               | SMAPE           | 15.8   |                  |
| LightGBM (Overstock)          | Accuracy        | 0.89   | Overstock Risk   |
|                               | ROC AUC         | 0.85   |                  |
| Prophet                       | MAPE            | 0.0578 | Demand           |

## Improvements

 The next things I would focus on given more time and compute to tackle this solution would be,

* Scaling to multi-store, multi-region forecasting.
* Expanding inventory optimization to include multi-head forecasting.
* Integrating automated inventory decisions based on forecasts via an API or an decision engine
* Adding deep learning models such as N-BEATS or Temporal Convolutional Networks.
* Establishing an Dashboard API for real-time retraining and compliance logic.
* Implementing Policy based Reinforcement Learning models for complete supply chain automation.
