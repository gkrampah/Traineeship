import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load("final_xgb_model.pkl")

Test_X = np.load("Test_X_transformed.npy")
Test_y = np.load("Test_y_transformed.npy")

Test_df = pd.DataFrame(
    Test_X, columns=[f"PCA_{str(t)}" for t in range(1, Test_X.shape[1] + 1)]
)

Test_df["target"] = Test_y


def highlight_columns(col):
    color = "skyblue" if col.name in ["target", "Predicted target"] else "green"
    return ["background-color: {}".format(color) for _ in col]


def run():

    
    add_selectbox = st.sidebar.selectbox(
        "How would you like to see the app work?", ("Example", "Batch")
    )

    st.sidebar.info("This app is created to predict patient hospital charges")
    st.sidebar.success("https://www.pycaret.org")

    # st.sidebar.image(image_hospital)

    st.title("XGBoost Classification of Subscribers")

    if add_selectbox == "Example":
        if st.button("Random Example"):
            st.write("Selecting random examples from the test data")
            samples = Test_df.sample(20)
            st.session_state.samples = samples
            st.dataframe(samples.style.apply(highlight_columns, axis=0))

        if "samples" in st.session_state and st.button("Predict"):
            input = st.session_state.samples.drop("target", axis=1)
            output = model.predict(input)
            st.session_state.samples["Predicted target"] = output

            st.dataframe(
                st.session_state.samples.style.apply(highlight_columns, axis=0)
            )

            # st.dataframe(
            #     st.session_state.samples.drop("target", axis=1).style.apply(
            #         highlight_columns, axis=0
            #     )
            # )

    if add_selectbox == "Batch":

        file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])

        if file_upload is not None:
            sample_data = pd.read_csv(file_upload, index_col=0)
            st.session_state.sample_data = sample_data
            st.dataframe(sample_data.style.apply(highlight_columns, axis=0))

        if "sample_data" in st.session_state and st.button("Predict"):
            input = st.session_state.sample_data.drop("target", axis=1)

            output = model.predict(input)
            st.session_state.sample_data["Predicted target"] = output

            st.dataframe(
                st.session_state.sample_data.style.apply(highlight_columns, axis=0)
            )


if __name__ == "__main__":
    run()
