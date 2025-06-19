# Virat-Kohil-Performance-Prediction
A regression-based machine learning analysis of Virat Kohli’s cricket performance using engineered match-level features. Incorporates statistical visualizations (batting position, dismissal, venue, time trends) and model comparisons to forecast future match outcomes.

**Overview :**
This project presents a machine learning-based approach to analyzing and predicting Virat Kohli’s cricket performance using real match data. By focusing on key features like batting position, type of dismissal, match venue, and monthly trends, the model investigates what influences his run-scoring ability. It uses Random Forest and Linear Regression models to forecast his future performance, with evaluation based on RMSE and R² scores. Alongside, the project includes insightful visualizations — such as career run trends, average runs by batting order and ground, and performance by dismissal type. A future prediction module also allows you to simulate upcoming match scenarios based on opponent, location, and recent form, making this both an analytical and predictive tool.

**Problem Statement:**

Understanding a cricketer’s performance trends is crucial for forecasting and strategic analysis. This project focuses on:

• Analyzing Virat Kohli’s historical performance in ODIs, Tests, and T20s

• Identifying how batting position, dismissal type, and match venue impact his performance

• Applying Random Forest and Linear Regression models to predict his future runs

• Visualizing performance trends to assist fans, analysts, and researchers in data-driven insights

**Dataset:**

The dataset contains Virat Kohli's performance records with the following key columns:

**• Runs**: Runs scored in each match  
**• Minutes Batted**: Time spent batting  
**• Best Figures**: Match performance summary  
**• 4s & 6s**: Number of boundaries and sixes hit  
**• Strike Rate**: Runs scored per 100 balls  
**• Batting Position**: Batting order in the lineup  
**• Dismissal**: How he got out (e.g., caught, bowled)  
**• Format**: Match type (ODI, Test, T20I)  
**• Opposition**: Opponent team  
**• Ground**: Match venue  
**• Match Date**: Date of the match  



**Features Used:**

• **Innings**: Batting innings number  

• **Avg_Last_5**: Average of last 5 match scores  


• **Match_Year**, **Match_Month**: Temporal aspects

• **Bat_Order**: Batting position  

• **Game_Number**: Sequential match number  

• **Low_Score_Count**: Number of recent low scores 

• **Recovery_Adjustment**: Derived performance trend  



 **Methodology:**

• Cleaned and transformed match-wise data

• Engineered features such as average of last 5 matches and recovery adjustment score 

• Encoded categorical variables like dismissal and opposition  

• Split data into train-test sets 

• Trained and compared two models: Random Forest and Linear Regression

• Evaluated using RMSE and R² scores 

• Predicted future match performance using constructed input  

• Visualized trends and feature impact




 **Tools & Technologies:**

**• Python**: Core programming language  

**• Pandas**: Data manipulation  

**• NumPy**: Numerical operations  

**• Matplotlib** & **Seaborn**: Data visualization  

**• Scikit-learn**: Machine learning modeling 

**• Excel**: Dataset source




**Visualizations:**

**• Career Runs Trend** – Line plot showing match number vs. runs scored across Kohli's career  

**• Model Comparison** – Bar chart comparing RMSE of Random Forest and Linear Regression  

**• Prediction Distribution** – Histogram displaying frequency distribution of predicted runs by Random Forest  

**• Average Runs by Batting Position** – Bar chart to analyze consistency at each batting order  

**• Dismissal Impact on Runs** – Chart showing how different types of dismissals affect his scores

**• Performance by Ground** – Horizontal bar graph of top 10 grounds where he scores the most  

**• Monthly Performance Trend** – Line plot revealing seasonality or monthly trends in performance 




 **Future Scope:**

**• Enhance dataset quality** by reducing missing values and including more match records for improved prediction accuracy  

**• Integrate a front-end interface** (e.g. Streamlit or Flask) for interactive prediction and performance visualization 

**• Enable match-specific filtering** to simulate how Virat might perform in specific conditions (e.g. against Australia in Mumbai) 

**• Include bowler data** to analyze dismissals by specific bowlers and uncover patterns in weaknesses  

**• Add classification models** such as predicting whether he will score a half-century or century  

**• Expand to other players** for comparative analysis and team-level insights



**Maintained by:**

Ashna Mittal

Linked in: www.linkedin.com/in/ashna-mittal-5260a2290
