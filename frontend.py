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

from final_postprocessing_pipeline import *
from cleanrawoutput import clean

df = ""
logFilePath = "streamlit.log"
newFileNamePath = "streamlitcleaneddata.csv"

st.title("BTU Comps")
st.write("Please upload the raw output from the beacon")

uploaded_file = st.file_uploader("Choose a file", type=["log", "txt"])
if uploaded_file is not None:
    # Save the uploaded file to disk
    with open(f"{logFilePath}", "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.write("File saved successfully!")
else:
    st.write("Please upload a file to proceed.")


# clean the raw output
st.write("Cleaning the raw output...")
clean(logFilePath, newFileNamePath)

# test selection
tests = [("Distance Correction", distanceCorrection), ("Velocity Clamping", velocityClamping), ("Outlier Removal", removeOutliers), ("Kalman Filter", pipelineKalman), ("EMA", smoothData), ("Velocity Clamping", velocityClamping)]
st.write("Select the tests you want to run:")
selected_tests = st.multiselect(
    "Cleaning",
    [test[0] for test in tests],
    default=[test[0] for test in tests]
)

# Ask the user to input beacon positions or use defaults
default_beacon_positions = np.array([[0, 0], [28.7, 0], [28.7, 25.7], [0, 25.7]])  

st.write("Please input the beacon positions or use the default values:")
beacon_positions_input = st.text_area(
    "Enter beacon positions as a list of lists (e.g., [[0, 0], [28.7, 0], [28.7, 25.7], [0, 25.7]])",
    value=str(default_beacon_positions.tolist())
)

# Add a button to start the pipeline
if st.button("Start Pipeline"):
    # run pipeline on the cleaned data
    # start report
    doc = Document()
    gen_title(doc, author="RF Positioning Report")
    
    # Filter the tests based on user selection
    tests_to_run = [test for test in tests if test[0] in selected_tests]

    # Load the cleaned data
    df = pd.read_csv(newFileNamePath)

    # Run the selected tests
    for test_name, test_func in tests_to_run:
        with st.spinner(f"Running {test_name}..."):
            df = test_func(df)
        st.success(f"{test_name} completed.")

    try:
        beacon_positions = np.array(eval(beacon_positions_input))
    except:
        st.write("Invalid input. Using default beacon positions.")
        beacon_positions = default_beacon_positions

    # st.write("Beacon positions:", beacon_positions)

    logFilePath = logFilePath.split(".")[0]

    # Add the cleaned data to the report
    data = (logFilePath, df)

    # Show loading spinner while generating plots
    with st.spinner("Generating plots... (this could take a while)"):
        # 1d plots
        plot1d([data], plot=False, doc=doc)

        # Plot the final DFs
        plotted = plotPlayers(data, beacon_positions, plot=False)
        imgPath = plotted[0]
        animated = plotted[1]

    add_section(doc, sectionName=data[0], sectionText="", imgPath=imgPath, caption="Final Player Movement Path")

    
    # Add the animated mp4 to the site
    st.video(animated, format="video/mp4", start_time=0, loop=True, autoplay=True)
    # add images to the site
    st.image(imgPath, caption="Player Movement Path")
    
    gen_pdf(doc, logFilePath+"_report")

    # Provide a download link for the generated PDF
    with open(logFilePath+"_report.pdf", "rb") as pdf_file:
        st.download_button(
            label="Download Report",
            data=pdf_file,
            file_name=logFilePath.split("/")[-1]+"_report.pdf",
            mime="application/pdf"
        )

    # ask the user if they want to restart the pipeline
    if st.button("Restart Pipeline"):
        st.caching.clear_cache()
        st.experimental_rerun()
























# Extract columns 
# b_columns = []
# for column in df.columns:
#     if not column.startswith('b'):
#         continue
#     else:
#         b_columns.append(column)

# # Tabs for Chart and Dataframe
# tab1, tab2 = st.tabs(["Chart", "Dataframe"])

# # Plotting
# with tab1:
#     st.write("Measurement Line vs Outlier Adjusted Line")

#     for column in b_columns:
#         fig, ax = plt.subplots()
#         ax.plot(df_read['timestamp'], df_read[column], label=f"Original {column}")
#         ax.plot(df_read['timestamp'], df_read[f'{column}_adjusted'], label=f"Adjusted {column}")
#         ax.set_xlabel("Timestamp")
#         ax.set_ylabel("Distance")
#         ax.set_title(f"Comparison for {column}")
#         ax.legend()
#         st.pyplot(fig) 

# # Original df
# with tab2:
#     st.write("Original Data for Each Beacon ")
#     st.dataframe(df, height=400, use_container_width=True)
