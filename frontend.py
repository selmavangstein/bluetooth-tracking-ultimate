'''
Example code from streamlit documentation -->

import streamlit as st
import pandas as pd
import numpy as np

st.write("Streamlit supports a wide range of data visualizations, including [Plotly, Altair, and Bokeh charts](https://docs.streamlit.io/develop/api-reference/charts). 📊 And with over 20 input widgets, you can easily make your data interactive!")

all_users = ["Alice", "Bob", "Charly"]
with st.container(border=True):
    users = st.multiselect("Users", all_users, default=all_users)
    rolling_average = st.toggle("Rolling average")

np.random.seed(42)
data = pd.DataFrame(np.random.randn(20, len(users)), columns=users)
if rolling_average:
    data = data.rolling(7).mean().dropna()

tab1, tab2 = st.tabs(["Chart", "Dataframe"])
tab1.line_chart(data, height=250)
tab2.dataframe(data, height=250, use_container_width=True)

'''
# Frontend currently only for outlier removal charts 
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import FinalOutlier2 # File name needs to be changed 

df = pd.read_csv("2-9-test3-uwb.csv")  
df_read = FinalOutlier2.removeOutliers_ts(df) 

st.title("Charts of Original Measurement Lines Compared to Outlier Adjusted Lines 📊")
st.write("This frontend visualizes plots of original measurement lines compared to outlier adjusted lines for each beacon.")

# Extract columns 
b_columns = []
for column in df.columns:
    if not column.startswith('b'):
        continue
    else:
        b_columns.append(column)

# Tabs for Chart and Dataframe
tab1, tab2 = st.tabs(["Chart", "Dataframe"])

# Plotting
with tab1:
    st.write("Measurement Line vs Outlier Adjusted Line")

    for column in b_columns:
        fig, ax = plt.subplots()
        ax.plot(df_read['timestamp'], df_read[column], label=f"Original {column}")
        ax.plot(df_read['timestamp'], df_read[f'{column}_adjusted'], label=f"Adjusted {column}")
        ax.set_xlabel("Timestamp")
        ax.set_ylabel("Distance")
        ax.set_title(f"Comparison for {column}")
        ax.legend()
        st.pyplot(fig) 

# Original df
with tab2:
    st.write("Original Data for Each Beacon ")
    st.dataframe(df, height=400, use_container_width=True)
