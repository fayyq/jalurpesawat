import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from io import StringIO


# ============================================================
# KONFIGURASI HALAMAN
# ============================================================

st.set_page_config(
    page_title="Dashboard Hotspot Kebakaran Indonesia",
    page_icon="🔥",
    layout="wide"
)


# ============================================================
# KONFIGURASI DATA
# ============================================================

SENSOR = "VIIRS_SNPP_NRT"
AREA = "95,-11,141,6"


# ============================================================
# MAP KEY DARI STREAMLIT SECRETS
# ============================================================

try:
    MAP_KEY = st.secrets["FIRMS_MAP_KEY"]
except Exception:
    MAP_KEY = None


if not MAP_KEY:
    st.error(
        "❌ MAP_KEY NASA FIRMS belum dikonfigurasi."
    )

    st.info(
        "Tambahkan FIRMS_MAP_KEY pada "
        "Streamlit Secrets."
    )

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚙️ Pengaturan Analisis")

st.sidebar.markdown(
    """
    Atur parameter analisis menggunakan
    pilihan di bawah ini.
    """
)


HARI_MUNDUR = st.sidebar.slider(
    "📅 Periode Data (hari)",
    min_value=7,
    max_value=365,
    value=60,
    step=1
)


BASELINE_DAYS = st.sidebar.slider(
    "📊 Baseline c' Laney (hari)",
    min_value=7,
    max_value=60,
    value=30,
    step=1
)


SIGMA_K = st.sidebar.slider(
    "📐 Nilai Sigma",
    min_value=1.0,
    max_value=4.0,
    value=3.0,
    step=0.5
)


WINDOW_SIZE = st.sidebar.slider(
    "📈 Window Moving Average",
    min_value=3,
    max_value=30,
    value=7,
    step=1
)


Z_THRESH = st.sidebar.slider(
    "⚠️ Threshold Z-Score",
    min_value=1.0,
    max_value=4.0,
    value=2.0,
    step=0.1
)


MIN_CONSECUTIVE = st.sidebar.slider(
    "⏱️ Minimum Durasi Anomali",
    min_value=2,
    max_value=10,
    value=3,
    step=1
)


TREND_LEN = st.sidebar.slider(
    "📉 Panjang Trend",
    min_value=2,
    max_value=10,
    value=3,
    step=1
)


# ============================================================
# JUDUL DASHBOARD
# ============================================================

st.title(
    "🔥 Dashboard Analisis Hotspot Kebakaran Indonesia"
)

st.markdown(
    """
    Dashboard ini menganalisis data hotspot kebakaran
    Indonesia dari **NASA FIRMS** menggunakan
    **Bagan Kendali c' Laney**, **Z-Score**, dan
    deteksi pola.
    """
)

st.divider()


# ============================================================
# FUNGSI MENGAMBIL DATA NASA FIRMS
# ============================================================

@st.cache_data(ttl=600)
def tarik_firms(
    map_key,
    sensor,
    area,
    hari_mundur
):

    akhir = pd.Timestamp.today().normalize()

    awal = (
        akhir
        - pd.Timedelta(
            days=hari_mundur
        )
    )

    semua_data = []

    # Pengambilan data per 10 hari
    chunk = 10

    tanggal_list = pd.date_range(
        awal,
        akhir,
        freq=f"{chunk}D"
    )

    for tgl in tanggal_list:

        url = (
            "https://firms.modaps.eosdis.nasa.gov/"
            f"api/area/csv/{map_key}/"
            f"{sensor}/{area}/{chunk}/"
            f"{tgl:%Y-%m-%d}"
        )

        try:

            response = requests.get(
                url,
                timeout=60
            )

            if response.status_code != 200:
                continue

            if not response.text.strip():
                continue

            data = pd.read_csv(
                StringIO(
                    response.text
                )
            )

            if not data.empty:
                semua_data.append(data)

        except Exception:
            continue


    # Tidak ada data
    if not semua_data:
        return pd.DataFrame()


    # Gabungkan seluruh data
    df = pd.concat(
        semua_data,
        ignore_index=True
    )


    # ========================================================
    # FILTER CONFIDENCE
    # ========================================================

    if "confidence" in df.columns:

        confidence = (
            df["confidence"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        df = df[
            confidence.isin(
                [
                    "n",
                    "h",
                    "nominal",
                    "high"
                ]
            )
        ]


    # ========================================================
    # FORMAT TANGGAL
    # ========================================================

    if "acq_date" in df.columns:

        df["tanggal"] = pd.to_datetime(
            df["acq_date"],
            errors="coerce"
        )

    else:

        return pd.DataFrame()


    df = df.dropna(
        subset=["tanggal"]
    )


    df = df.sort_values(
        "tanggal"
    )


    # Hilangkan duplikasi
    df = df.drop_duplicates()


    return df


# ============================================================
# AGREGASI DATA HARIAN
# ============================================================

def agregasi_harian(df):

    if df.empty:

        return pd.DataFrame(
            columns=[
                "tanggal",
                "hotspot"
            ]
        )


    harian = (
        df.groupby("tanggal")
        .size()
        .reset_index(
            name="hotspot"
        )
    )


    harian = harian.sort_values(
        "tanggal"
    )


    return harian


# ============================================================
# FUNGSI MENGHITUNG DURASI TRUE BERTURUT-TURUT
# ============================================================

def consecutive_true_count(series):

    hasil = []

    count = 0

    for value in series:

        if bool(value):

            count += 1

        else:

            count = 0

        hasil.append(count)

    return hasil


# ============================================================
# FUNGSI MENGHITUNG RUN TREND
# ============================================================

def consecutive_same_sign(series):

    hasil = []

    count = 0

    previous = None

    for value in series:

        if value == 0:

            count = 0

        elif value == previous:

            count += 1

        else:

            count = 1

        hasil.append(count)

        previous = value

    return hasil


# ============================================================
# BAGAN KENDALI c' LANEY
# ============================================================

def batas_kendali_laney(
    harian,
    baseline_days=30,
    k=3
):

    df = harian.copy()


    # Data baseline
    baseline = (
        df["hotspot"]
        .tail(baseline_days)
    )


    cbar = baseline.mean()


    if pd.isna(cbar) or cbar <= 0:
        cbar = 1


    # Standardisasi
    z = (
        df["hotspot"] - cbar
    ) / np.sqrt(cbar)


    # Estimasi sigma
    sigma_z = (
        z.diff()
        .abs()
        .mean()
        / 1.128
    )


    if (
        pd.isna(sigma_z)
        or sigma_z <= 0
    ):

        sigma_z = 1


    # Lebar batas kendali
    lebar = (
        k
        * np.sqrt(cbar)
        * sigma_z
    )


    CL = cbar

    UCL = (
        cbar
        + lebar
    )

    LCL = max(
        0,
        cbar - lebar
    )


    # Simpan ke dataframe

    df["CL"] = CL

    df["UCL"] = UCL

    df["LCL"] = LCL


    # Anomali Laney

    df["anomali_laney"] = (
        (df["hotspot"] > UCL)
        |
        (df["hotspot"] < LCL)
    )


    return df


# ============================================================
# ANALISIS STATISTIK
# ============================================================

def analisis(
    harian,
    baseline_days,
    sigma_k,
    window_size,
    z_thresh,
    min_consecutive,
    trend_len
):

    df = batas_kendali_laney(
        harian,
        baseline_days,
        sigma_k
    )


    # ========================================================
    # MOVING AVERAGE
    # ========================================================

    df["moving_average"] = (
        df["hotspot"]
        .rolling(
            window=window_size,
            min_periods=1
        )
        .mean()
    )


    # ========================================================
    # Z-SCORE
    # ========================================================

    mean_hotspot = (
        df["hotspot"].mean()
    )


    std_hotspot = (
        df["hotspot"].std()
    )


    if (
        pd.isna(std_hotspot)
        or std_hotspot == 0
    ):

        std_hotspot = 1


    df["z_score"] = (
        df["hotspot"]
        - mean_hotspot
    ) / std_hotspot


    df["anomali_z"] = (
        df["z_score"].abs()
        > z_thresh
    )


    # ========================================================
    # DURASI ANOMALI
    # ========================================================

    df["durasi_anomali"] = (
        consecutive_true_count(
            df["anomali_z"]
        )
    )


    df["anomali_durasi"] = (
        df["durasi_anomali"]
        >= min_consecutive
    )


    # ========================================================
    # TREND
    # ========================================================

    df["selisih"] = (
        df["hotspot"].diff()
    )


    df["arah"] = np.where(
        df["selisih"] > 0,
        1,
        np.where(
            df["selisih"] < 0,
            -1,
            0
        )
    )


    df["run_trend"] = (
        consecutive_same_sign(
            df["arah"]
        )
    )


    df["trend_naik"] = (
        (df["arah"] == 1)
        &
        (
            df["run_trend"]
            >= trend_len
        )
    )


    df["trend_turun"] = (
        (df["arah"] == -1)
        &
        (
            df["run_trend"]
            >= trend_len
        )
    )


    # ========================================================
    # SEQUENCE
    # ========================================================

    df["sequence"] = (
        df["anomali_z"]
        .rolling(
            window=3,
            min_periods=1
        )
        .sum()
    )


    # ========================================================
    # KORELASI
    # ========================================================

    df["correlation"] = (
        df["hotspot"]
        .rolling(
            window=window_size
        )
        .corr(
            df["moving_average"]
        )
    )


    return df


# ============================================================
# MENGAMBIL DATA
# ============================================================

with st.spinner(
    "⏳ Mengambil data hotspot dari NASA FIRMS..."
):

    raw_data = tarik_firms(
        MAP_KEY,
        SENSOR,
        AREA,
        HARI_MUNDUR
    )


# ============================================================
# CEK DATA
# ============================================================

if raw_data.empty:

    st.error(
        "❌ Data hotspot tidak berhasil diperoleh."
    )

    st.info(
        """
        Kemungkinan penyebab:

        1. MAP_KEY tidak valid
        2. NASA FIRMS sedang mengalami gangguan
        3. Data pada periode tersebut belum tersedia
        4. Koneksi internet bermasalah
        """
    )

    st.stop()


# ============================================================
# AGREGASI HARIAN
# ============================================================

harian = agregasi_harian(
    raw_data
)


if harian.empty:

    st.error(
        "❌ Tidak terdapat data harian."
    )

    st.stop()


# ============================================================
# ANALISIS
# ============================================================

hasil = analisis(
    harian,
    BASELINE_DAYS,
    SIGMA_K,
    WINDOW_SIZE,
    Z_THRESH,
    MIN_CONSECUTIVE,
    TREND_LEN
)


# ============================================================
# KPI
# ============================================================

total_hotspot = len(
    raw_data
)


rata_rata = (
    hasil["hotspot"]
    .mean()
)


hotspot_terakhir = (
    hasil["hotspot"]
    .iloc[-1]
)


total_anomali = (
    hasil["anomali_z"]
    .sum()
)


# ============================================================
# TAMPILKAN KPI
# ============================================================

col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "🔥 Total Hotspot",
        f"{total_hotspot:,}"
    )


with col2:

    st.metric(
        "📊 Rata-rata Harian",
        f"{rata_rata:,.1f}"
    )


with col3:

    st.metric(
        "📍 Hotspot Terakhir",
        f"{hotspot_terakhir:,}"
    )


with col4:

    st.metric(
        "⚠️ Total Anomali",
        f"{total_anomali:,}"
    )


st.divider()


# ============================================================
# GRAFIK c' LANEY
# ============================================================

st.subheader(
    "📈 Pergerakan Hotspot dan Bagan Kendali c' Laney"
)


fig = go.Figure()


# Hotspot harian

fig.add_trace(
    go.Scatter(
        x=hasil["tanggal"],
        y=hasil["hotspot"],
        mode="lines+markers",
        name="Hotspot Harian"
    )
)


# Center Line

fig.add_trace(
    go.Scatter(
        x=hasil["tanggal"],
        y=hasil["CL"],
        mode="lines",
        name="CL"
    )
)


# Upper Control Limit

fig.add_trace(
    go.Scatter(
        x=hasil["tanggal"],
        y=hasil["UCL"],
        mode="lines",
        name="UCL"
    )
)


# Lower Control Limit

fig.add_trace(
    go.Scatter(
        x=hasil["tanggal"],
        y=hasil["LCL"],
        mode="lines",
        name="LCL"
    )
)


# Titik anomali

anomali = hasil[
    hasil["anomali_z"]
]


if not anomali.empty:

    fig.add_trace(
        go.Scatter(
            x=anomali["tanggal"],
            y=anomali["hotspot"],
            mode="markers",
            name="Anomali"
        )
    )


fig.update_layout(
    xaxis_title="Tanggal",
    yaxis_title="Jumlah Hotspot",
    hovermode="x unified",
    height=550,
    legend_title="Keterangan"
)


st.plotly_chart(
    fig,
    use_container_width=True
)


# ============================================================
# GRAFIK Z-SCORE
# ============================================================

st.subheader(
    "📊 Deteksi Anomali Berdasarkan Z-Score"
)


fig_z = go.Figure()


fig_z.add_trace(
    go.Scatter(
        x=hasil["tanggal"],
        y=hasil["z_score"],
        mode="lines+markers",
        name="Z-Score"
    )
)


fig_z.add_hline(
    y=Z_THRESH,
    line_dash="dash",
    annotation_text=(
        f"Batas Atas +{Z_THRESH}"
    )
)


fig_z.add_hline(
    y=-Z_THRESH,
    line_dash="dash",
    annotation_text=(
        f"Batas Bawah -{Z_THRESH}"
    )
)


fig_z.update_layout(
    xaxis_title="Tanggal",
    yaxis_title="Z-Score",
    height=400
)


st.plotly_chart(
    fig_z,
    use_container_width=True
)


# ============================================================
# HASIL POLA
# ============================================================

st.subheader(
    "🔎 Hasil Deteksi Pola"
)


col1, col2, col3 = st.columns(3)


with col1:

    jumlah_laney = (
        hasil["anomali_laney"]
        .sum()
    )

    st.metric(
        "Anomali c' Laney",
        f"{jumlah_laney:,}"
    )


with col2:

    trend_naik = (
        hasil["trend_naik"]
        .sum()
    )

    st.metric(
        "📈 Trend Naik",
        f"{trend_naik:,}"
    )


with col3:

    trend_turun = (
        hasil["trend_turun"]
        .sum()
    )

    st.metric(
        "📉 Trend Turun",
        f"{trend_turun:,}"
    )


# ============================================================
# STATUS TERAKHIR
# ============================================================

st.subheader(
    "📌 Status Hotspot Terakhir"
)


terakhir = hasil.iloc[-1]


if bool(terakhir["anomali_z"]):

    st.warning(
        """
        ⚠️ **Hari terakhir terdeteksi sebagai anomali
        berdasarkan Z-Score.**
        """
    )


elif bool(terakhir["anomali_laney"]):

    st.warning(
        """
        ⚠️ **Hari terakhir berada di luar batas
        kendali c' Laney.**
        """
    )


else:

    st.success(
        """
        ✅ **Jumlah hotspot hari terakhir masih
        berada dalam pola normal.**
        """
    )


# ============================================================
# TABEL HASIL ANALISIS
# ============================================================

st.subheader(
    "📋 Tabel Hasil Analisis"
)


tabel = hasil[
    [
        "tanggal",
        "hotspot",
        "moving_average",
        "z_score",
        "anomali_z",
        "anomali_laney",
        "trend_naik",
        "trend_turun"
    ]
].copy()


tabel["tanggal"] = (
    tabel["tanggal"]
    .dt.strftime("%Y-%m-%d")
)


tabel = tabel.rename(
    columns={
        "tanggal": "Tanggal",
        "hotspot": "Hotspot",
        "moving_average": "Moving Average",
        "z_score": "Z-Score",
        "anomali_z": "Anomali Z-Score",
        "anomali_laney": "Anomali c' Laney",
        "trend_naik": "Trend Naik",
        "trend_turun": "Trend Turun"
    }
)


st.dataframe(
    tabel,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# DATA MENTAH
# ============================================================

with st.expander(
    "📂 Lihat Data Mentah NASA FIRMS"
):

    st.dataframe(
        raw_data,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# INFORMASI METODE
# ============================================================

with st.expander(
    "ℹ️ Tentang Dashboard"
):

    st.markdown(
        """
        ### 🛰️ Sumber Data

        **NASA FIRMS (Fire Information for Resource Management System)**

        Sensor yang digunakan:

        **VIIRS SNPP Near Real-Time (VIIRS_SNPP_NRT)**

        Wilayah:

        **Indonesia**

        ### 📊 Metode Statistik

        Dashboard menggunakan beberapa metode:

        **1. Bagan Kendali c' Laney**

        Digunakan untuk melihat apakah jumlah hotspot
        harian berada dalam batas kendali statistik.

        **2. Z-Score**

        Digunakan untuk mengidentifikasi hari dengan
        jumlah hotspot yang menyimpang dari pola umum.

        **3. Moving Average**

        Digunakan untuk melihat pola rata-rata hotspot
        dalam beberapa hari.

        **4. Trend Detection**

        Digunakan untuk mengidentifikasi kecenderungan
        hotspot meningkat atau menurun.

        **5. Sequence Detection**

        Digunakan untuk melihat pola anomali yang terjadi
        secara berturut-turut.

        ### ⚠️ Catatan

        Satu hotspot merupakan satu titik deteksi satelit,
        sehingga **1 hotspot tidak selalu berarti 1 kejadian
        kebakaran**.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🔥 Dashboard Analisis Hotspot Kebakaran Indonesia | "
    "NASA FIRMS | Statistika Big Data"
)