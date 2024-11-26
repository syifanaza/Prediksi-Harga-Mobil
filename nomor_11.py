import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Memuat model prediksi harga mobil dari file .sav
model = pickle.load(open('linear_regression_model.sav', 'rb'))

# Judul aplikasi
st.title('Prediksi Harga Mobil')
st.header("Dataset")

# Membaca file CSV untuk menampilkan data
df1 = pd.read_csv('car_price.csv')
st.dataframe(df1)

# Menampilkan statistik deskriptif dari dataset
st.write("Statistik Deskriptif")
st.dataframe(df1.describe())

# Menampilkan grafik untuk masing-masing kolom
st.write("Grafik Highway-mpg")
chart_highwaympg = df1['highwaympg']
st.line_chart(chart_highwaympg)

st.write("Grafik Curbweight")
chart_curbweight = df1['curbweight']
st.line_chart(chart_curbweight)

st.write("Grafik Horsepower")
chart_horsepower = df1['horsepower']
st.line_chart(chart_horsepower)

# Menambahkan pilihan model lain
model_choice = st.selectbox('Pilih Model', ['Linear Regression', 'Ridge Regression', 'Random Forest'])

if model_choice == 'Linear Regression':
    model = pickle.load(open('linear_regression_model.sav', 'rb'))
elif model_choice == 'Ridge Regression':
    model = pickle.load(open('ridge_regression_model.sav', 'rb'))
else:
    model = pickle.load(open('random_forest_model.sav', 'rb'))

# Memastikan bahwa model yang dimuat adalah tipe yang benar
st.write(f"Model yang dimuat adalah: {type(model)}")

# Input nilai untuk variabel independen (fitur)
highwaympg = st.slider('Pilih Highway MPG', min_value=0, max_value=100, value=30)
curbweight = st.slider('Pilih Curbweight', min_value=0, max_value=10000, value=2500)
horsepower = st.slider('Pilih Horsepower', min_value=0, max_value=1000, value=150)

# Tombol untuk memulai prediksi
if st.button('Prediksi'):
    # Membuat prediksi berdasarkan input yang diberikan
    car_prediction = model.predict([[highwaympg, curbweight, horsepower]])

    # Mengubah hasil prediksi ke tipe data yang dapat ditampilkan
    harga_mobil_str = np.array(car_prediction)
    harga_mobil_float = float(harga_mobil_str[0][0])

    # Menampilkan hasil prediksi harga mobil
    harga_mobil_formatted = f"Rp {harga_mobil_float:,.2f}"
    st.write(f'Harga Mobil yang diprediksi adalah: {harga_mobil_formatted}')

    # Menambahkan fitur untuk unduh hasil prediksi
    prediksi_data = pd.DataFrame({
        'Highway MPG': [highwaympg],
        'Curbweight': [curbweight],
        'Horsepower': [horsepower],
        'Harga Prediksi': [harga_mobil_float]
    })

    st.download_button(
        label="Unduh Hasil Prediksi",
        data=prediksi_data.to_csv(index=False),
        file_name='prediksi_harga_mobil.csv',
        mime='text/csv'
    )

# Menampilkan penjelasan tentang model
st.write("""
Model ini menggunakan fitur-fitur seperti Highway MPG, Curbweight, dan Horsepower untuk memprediksi harga mobil. 
Highway MPG menunjukkan efisiensi bahan bakar, Curbweight mencerminkan berat mobil, 
dan Horsepower mengindikasikan daya mesin mobil. Kombinasi faktor-faktor ini digunakan untuk menghasilkan harga mobil.
""")
