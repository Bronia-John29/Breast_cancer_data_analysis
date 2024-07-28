import streamlit as st
import altair as alt
import pandas as pd

st.title('Test Streamlit App')

# Create a simple Altair chart
data = pd.DataFrame({
    'a': ['A', 'B', 'C', 'D', 'E'],
    'b': [5, 3, 6, 7, 2]
})

chart = alt.Chart(data).mark_bar().encode(
    x='a',
    y='b'
)

st.altair_chart(chart, use_container_width=True)
