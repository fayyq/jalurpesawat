import time
from io import StringIO

import numpy as np
import pandas as pd
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


# ============================================================
# 1. KONFIGURASI STREAMLIT
# ============================================================

st.set_page_config(
    page_title="Dashboard Hotspot Kebakaran",
    page_icon="🔥",
    layout="wide"
)


# ============================================================
# 2. JUDUL
# ============================================================

st.title("🔥 Dashboard Analisis Hotspot Kebakaran Indonesia")

st.write(
    "Dashboard ini menganalisis data hotspot kebakaran "
    "dari NASA FIRMS menggunakan Bagan Kendali c' Laney "
    "dan deteksi pola/anomali."
)


# ============================================================
# 3. SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Parameter Dashboard")

MAP_KEY = st.sidebar.text_input(
    "NASA FIRMS MAP_KEY",
    type="password"
)

SENSOR = st.sidebar.selectbox(
    "Sensor",
    [
        "VIIRS_SNPP_NRT",
        "VIIRS_NOAA20_NRT",
        "VIIRS_NOAA21_NRT"
    ]
)

HARI_MUNDUR = st.sidebar.slider(
    "Jumlah hari data",
    min_value=5,
    max_value=60,
    value=30
)

BASELINE_DAYS = st.sidebar.slider(
    "Baseline Bagan Kendali",
    min_value=7,
    max_value=30,
    value=14
)

SIGMA_K = st.sidebar.slider(
    "Batas Sigma",
    min_value=1.0,
    max_value=4.0,
    value=3.0,
    step=0.5
)

WINDOW_SIZE = st.sidebar.slider(
    "Window Z-Score",
    min_value=3,
    max_value=14,
    value=7
)

Z_THRESH = st.sidebar.slider(
    "Threshold Z-Score",
    min_value=1.0,
    max_value=4.0,
    value=2.0,
    step=0.5
)

MIN_CONSECUTIVE = st.sidebar.slider(
    "Minimum hari beruntun",
    min_value=2,
    max_value=7,
    value=3
)

TREND_LEN = st.sidebar.slider(
    "Minimum panjang trend",
    min_value=2,
    max_value=7,
    value=3
)

# Wilayah Indonesia
AREA = "95,-11,141,6"


# ============================================================
# 4. FUNGSI MENGAMBIL DATA NASA FIRMS
# ============================================================

@st.cache_data(ttl=600)
def tarik_firms(
    map_key,
    sensor,
    area,
    hari_mundur,
    chunk=5
):

    akhir = (
        pd.Timestamp.utcnow()
        .tz_localize(None)
        .normalize()
    )

    awal = (
        akhir
        - pd.Timedelta(days=hari_mundur)
    )

    potongan = []

    tanggal_list = pd.date_range(
        awal,
        akhir,
        freq=f"{chunk}D"
    )

    for tgl in tanggal_list:

        url = (
            "https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
            f"{map_key}/{sensor}/{area}/{chunk}/"
            f"{tgl:%Y-%m-%d}"
        )

        try:

            response = requests.get(
                url,
                timeout=60
            )

            response.raise_for_status()

            data = pd.read_csv(
                StringIO(response.text)
            )

            if (
                "latitude" in data.columns
                and len(data) > 0
            ):
                potongan.append(data)

        except Exception as e:

            st.warning(
                f"Gagal mengambil data "
                f"{tgl:%Y-%m-%d}: {e}"
            )

        time.sleep(0.3)

    # --------------------------------------------------------
    # Tidak ada data
    # --------------------------------------------------------

    if not potongan:

        return pd.DataFrame()

    # --------------------------------------------------------
    # Gabungkan data
    # --------------------------------------------------------

    df = pd.concat(
        potongan,
        ignore_index=True
    ).drop_duplicates()

    # --------------------------------------------------------
    # Pastikan acq_time benar
    # --------------------------------------------------------

    df["acq_time"] = (
        pd.to_numeric(
            df["acq_time"],
            errors="coerce"
        )
        .fillna(0)
        .astype(int)
        .astype(str)
        .str.zfill(4)
    )

    # --------------------------------------------------------
    # Buat kolom waktu
    # --------------------------------------------------------

    df["waktu"] = pd.to_datetime(
        df["acq_date"].astype(str)
        + " "
        + df["acq_time"],
        format="%Y-%m-%d %H%M",
        errors="coerce"
    )

    # --------------------------------------------------------
    # Filter confidence
    # --------------------------------------------------------

    if "confidence" in df.columns:

        confidence = (
            df["confidence"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        # VIIRS:
        # n = nominal
        # h = high
        #
        # Hanya menggunakan confidence
        # nominal dan high.

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

    # --------------------------------------------------------
    # Pastikan FRP numerik
    # --------------------------------------------------------

    if "frp" in df.columns:

        df["frp"] = pd.to_numeric(
            df["frp"],
            errors="coerce"
        )

    # --------------------------------------------------------
    # Hapus waktu tidak valid
    # --------------------------------------------------------

    df = df.dropna(
        subset=["waktu"]
    )

    return df.reset_index(
        drop=True
    )


# ============================================================
# 5. AGREGASI DATA HARIAN
# ============================================================

def agregasi_harian(df):

    if df.empty:

        return pd.DataFrame()

    harian = (
        df.assign(
            tanggal=df["waktu"].dt.normalize()
        )
        .groupby("tanggal")
        .agg(
            jumlah=("frp", "size"),
            frp_mean=("frp", "mean")
        )
        .asfreq("D")
    )

    harian["jumlah"] = (
        harian["jumlah"]
        .fillna(0)
    )

    harian["frp_mean"] = (
        harian["frp_mean"]
        .interpolate(
            limit_direction="both"
        )
    )

    return harian.reset_index()


# ============================================================
# 6. FUNGSI MENGHITUNG STREAK
# ============================================================

def consecutive_true_count(mask):

    hasil = []

    run = 0

    for nilai in mask.to_numpy():

        if nilai:

            run += 1

        else:

            run = 0

        hasil.append(run)

    return pd.Series(
        hasil,
        index=mask.index
    )


def consecutive_same_sign(diff):

    sign = np.sign(
        diff.fillna(0)
    ).to_numpy()

    hasil = []

    run = 0
    previous = 0

    for nilai in sign:

        if (
            nilai != 0
            and nilai == previous
        ):

            run += 1

        elif nilai != 0:

            run = 1

        else:

            run = 0

        hasil.append(run)

        previous = nilai

    return pd.Series(
        hasil,
        index=diff.index
    )


# ============================================================
# 7. BAGAN KENDALI c' LANEY
# ============================================================

def batas_kendali_laney(
    data,
    baseline_n,
    k=3
):

    baseline_n = min(
        baseline_n,
        len(data)
    )

    base = data.iloc[
        :baseline_n
    ]

    cbar = base.mean()

    if cbar <= 0:

        return (
            cbar,
            cbar,
            cbar
        )

    z = (
        base - cbar
    ) / np.sqrt(cbar)

    sigma_z = (
        z.diff()
        .abs()
        .mean()
        / 1.128
    )

    if pd.isna(sigma_z):

        sigma_z = 1

    lebar = (
        k
        * np.sqrt(cbar)
        * sigma_z
    )

    ucl = cbar + lebar

    lcl = max(
        cbar - lebar,
        0
    )

    return (
        cbar,
        ucl,
        lcl
    )


# ============================================================
# 8. ANALISIS STATISTIK
# ============================================================

def analisis(data_harian):

    df = data_harian.copy()

    if len(df) < 3:

        return df

    # --------------------------------------------------------
    # c' LANEY
    # --------------------------------------------------------

    cl, ucl, lcl = batas_kendali_laney(
        df["jumlah"],
        BASELINE_DAYS,
        SIGMA_K
    )

    df["cl"] = cl
    df["ucl"] = ucl
    df["lcl"] = lcl

    # --------------------------------------------------------
    # Z-SCORE JUMLAH HOTSPOT
    # --------------------------------------------------------

    mean_jumlah = (
        df["jumlah"]
        .shift(1)
        .rolling(WINDOW_SIZE)
        .mean()
    )

    std_jumlah = (
        df["jumlah"]
        .shift(1)
        .rolling(WINDOW_SIZE)
        .std()
    )

    std_jumlah = std_jumlah.replace(
        0,
        np.nan
    )

    df["z_jumlah"] = (
        (df["jumlah"] - mean_jumlah)
        / std_jumlah
    )

    df["anomali_jumlah"] = (
        df["z_jumlah"]
        .abs()
        > Z_THRESH
    )

    # --------------------------------------------------------
    # Z-SCORE FRP
    # --------------------------------------------------------

    mean_frp = (
        df["frp_mean"]
        .shift(1)
        .rolling(WINDOW_SIZE)
        .mean()
    )

    std_frp = (
        df["frp_mean"]
        .shift(1)
        .rolling(WINDOW_SIZE)
        .std()
    )

    std_frp = std_frp.replace(
        0,
        np.nan
    )

    df["z_frp"] = (
        (df["frp_mean"] - mean_frp)
        / std_frp
    )

    df["anomali_frp"] = (
        df["z_frp"]
        .abs()
        > Z_THRESH
    )

    # --------------------------------------------------------
    # POLA 1
    # THRESHOLD + DURASI
    # --------------------------------------------------------

    df["streak_atas"] = (
        consecutive_true_count(
            df["jumlah"] > df["ucl"]
        )
    )

    df["streak_bawah"] = (
        consecutive_true_count(
            df["jumlah"] < df["lcl"]
        )
    )

    df["pola_threshold"] = (
        (
            df["streak_atas"]
            >= MIN_CONSECUTIVE
        )
        |
        (
            df["streak_bawah"]
            >= MIN_CONSECUTIVE
        )
    )

    df["sisi"] = np.select(
        [
            df["streak_atas"]
            >= MIN_CONSECUTIVE,

            df["streak_bawah"]
            >= MIN_CONSECUTIVE
        ],
        [
            "Atas (UCL)",
            "Bawah (LCL)"
        ],
        default="-"
    )

    # --------------------------------------------------------
    # POLA 2
    # TREND
    # --------------------------------------------------------

    df["rolling_mean"] = (
        df["jumlah"]
        .rolling(WINDOW_SIZE)
        .mean()
    )

    selisih = (
        df["rolling_mean"]
        .diff()
    )

    df["streak_trend"] = (
        consecutive_same_sign(
            selisih
        )
    )

    df["pola_trend"] = (
        df["streak_trend"]
        >= TREND_LEN
    )

    df["arah_trend"] = np.select(
        [
            selisih > 0,
            selisih < 0
        ],
        [
            "Naik",
            "Turun"
        ],
        default="-"
    )

    # --------------------------------------------------------
    # POLA 3
    # SEQUENCE + CORRELATION
    # --------------------------------------------------------

    anomali_sebelumnya = (
        df["anomali_jumlah"]
        .shift(1)
        .rolling(
            3,
            min_periods=1
        )
        .max()
        .fillna(False)
        .astype(bool)
    )

    df["pola_sequence"] = (
        df["anomali_frp"]
        .fillna(False)
        &
        anomali_sebelumnya
    )

    return df


# ============================================================
# 9. MEMBUAT EVENT LOG
# ============================================================

def buat_log(df):

    kejadian = []

    for _, row in df.iterrows():

        # Threshold
        if row["pola_threshold"]:

            durasi = int(
                max(
                    row["streak_atas"],
                    row["streak_bawah"]
                )
            )

            kejadian.append(
                {
                    "Tanggal": row["tanggal"],
                    "Pola": "Threshold + Durasi",
                    "Deskripsi": (
                        f"Jumlah hotspot berada di "
                        f"{row['sisi']} selama "
                        f"{durasi} hari"
                    ),
                    "Jumlah Hotspot": int(
                        row["jumlah"]
                    )
                }
            )

        # Trend
        if row["pola_trend"]:

            kejadian.append(
                {
                    "Tanggal": row["tanggal"],
                    "Pola": "Trend",
                    "Deskripsi": (
                        f"Trend {row['arah_trend']} "
                        f"selama "
                        f"{int(row['streak_trend'])} hari"
                    ),
                    "Jumlah Hotspot": int(
                        row["jumlah"]
                    )
                }
            )

        # Sequence
        if row["pola_sequence"]:

            kejadian.append(
                {
                    "Tanggal": row["tanggal"],
                    "Pola": (
                        "Sequence + Correlation"
                    ),
                    "Deskripsi": (
                        "Anomali FRP menyusul "
                        "anomali jumlah hotspot"
                    ),
                    "Jumlah Hotspot": int(
                        row["jumlah"]
                    )
                }
            )

    if not kejadian:

        return pd.DataFrame(
            columns=[
                "Tanggal",
                "Pola",
                "Deskripsi",
                "Jumlah Hotspot"
            ]
        )

    return (
        pd.DataFrame(kejadian)
        .sort_values(
            "Tanggal",
            ascending=False
        )
        .reset_index(drop=True)
    )


# ============================================================
# 10. MEMBUAT GRAFIK
# ============================================================

def buat_grafik(df):

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[
            0.7,
            0.3
        ],
        vertical_spacing=0.08,
        subplot_titles=[
            "Bagan Kendali c' Laney - Hotspot Harian",
            "Rata-rata FRP Harian"
        ]
    )

    x = df["tanggal"]

    # --------------------------------------------------------
    # JUMLAH HOTSPOT
    # --------------------------------------------------------

    fig.add_trace(
        go.Scatter(
            x=x,
            y=df["jumlah"],
            mode="lines+markers",
            name="Jumlah Hotspot"
        ),
        row=1,
        col=1
    )

    # --------------------------------------------------------
    # CL
    # --------------------------------------------------------

    fig.add_trace(
        go.Scatter(
            x=x,
            y=df["cl"],
            mode="lines",
            name="CL"
        ),
        row=1,
        col=1
    )

    # --------------------------------------------------------
    # UCL
    # --------------------------------------------------------

    fig.add_trace(
        go.Scatter(
            x=x,
            y=df["ucl"],
            mode="lines",
            name="UCL",
            line=dict(
                dash="dash"
            )
        ),
        row=1,
        col=1
    )

    # --------------------------------------------------------
    # LCL
    # --------------------------------------------------------

    fig.add_trace(
        go.Scatter(
            x=x,
            y=df["lcl"],
            mode="lines",
            name="LCL",
            line=dict(
                dash="dash"
            )
        ),
        row=1,
        col=1
    )

    # --------------------------------------------------------
    # ROLLING MEAN
    # --------------------------------------------------------

    fig.add_trace(
        go.Scatter(
            x=x,
            y=df["rolling_mean"],
            mode="lines",
            name="Rolling Mean"
        ),
        row=1,
        col=1
    )

    # --------------------------------------------------------
    # ANOMALI
    # --------------------------------------------------------

    anomali = df[
        df["anomali_jumlah"]
        .fillna(False)
    ]

    if not anomali.empty:

        fig.add_trace(
            go.Scatter(
                x=anomali["tanggal"],
                y=anomali["jumlah"],
                mode="markers",
                name="Anomali Z-Score",
                marker=dict(
                    symbol="x",
                    size=10
                )
            ),
            row=1,
            col=1
        )

    # --------------------------------------------------------
    # THRESHOLD
    # --------------------------------------------------------

    threshold = df[
        df["pola_threshold"]
        .fillna(False)
    ]

    if not threshold.empty:

        fig.add_trace(
            go.Scatter(
                x=threshold["tanggal"],
                y=threshold["jumlah"],
                mode="markers",
                name="Threshold + Durasi",
                marker=dict(
                    symbol="diamond",
                    size=12
                )
            ),
            row=1,
            col=1
        )

    # --------------------------------------------------------
    # TREND
    # --------------------------------------------------------

    trend = df[
        df["pola_trend"]
        .fillna(False)
    ]

    if not trend.empty:

        fig.add_trace(
            go.Scatter(
                x=trend["tanggal"],
                y=trend["jumlah"],
                mode="markers",
                name="Trend",
                marker=dict(
                    symbol="triangle-up",
                    size=9
                )
            ),
            row=1,
            col=1
        )

    # --------------------------------------------------------
    # FRP
    # --------------------------------------------------------

    fig.add_trace(
        go.Bar(
            x=x,
            y=df["frp_mean"],
            name="FRP Rata-rata"
        ),
        row=2,
        col=1
    )

    fig.update_layout(
        height=700,
        template="plotly_white",
        hovermode="x unified"
    )

    fig.update_yaxes(
        title_text="Jumlah Hotspot",
        row=1,
        col=1
    )

    fig.update_yaxes(
        title_text="FRP (MW)",
        row=2,
        col=1
    )

    return fig


# ============================================================
# 11. PROGRAM UTAMA
# ============================================================

if not MAP_KEY:

    st.info(
        "👈 Silakan masukkan MAP_KEY NASA FIRMS "
        "di sidebar terlebih dahulu."
    )

else:

    with st.spinner(
        "🔥 Mengambil data hotspot kebakaran..."
    ):

        titik = tarik_firms(
            MAP_KEY,
            SENSOR,
            AREA,
            HARI_MUNDUR
        )

    if titik.empty:

        st.error(
            "❌ Data hotspot tidak ditemukan. "
            "Periksa kembali MAP_KEY NASA FIRMS."
        )

    else:

        # ----------------------------------------------------
        # AGREGASI
        # ----------------------------------------------------

        harian = agregasi_harian(
            titik
        )

        # ----------------------------------------------------
        # ANALISIS
        # ----------------------------------------------------

        hasil = analisis(
            harian
        )

        # ----------------------------------------------------
        # KPI
        # ----------------------------------------------------

        total_hotspot = len(titik)

        rata_harian = (
            hasil["jumlah"]
            .mean()
        )

        hotspot_terakhir = int(
            hasil.iloc[-1]["jumlah"]
        )

        total_anomali = int(
            hasil["anomali_jumlah"]
            .fillna(False)
            .sum()
        )

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "🔥 Total Hotspot",
            f"{total_hotspot:,}"
        )

        col2.metric(
            "📊 Rata-rata Harian",
            f"{rata_harian:.1f}"
        )

        col3.metric(
            "📍 Hotspot Terakhir",
            hotspot_terakhir
        )

        col4.metric(
            "⚠️ Total Anomali",
            total_anomali
        )

        # ----------------------------------------------------
        # GRAFIK
        # ----------------------------------------------------

        st.subheader(
            "📈 Analisis Hotspot Kebakaran"
        )

        fig = buat_grafik(
            hasil
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # ----------------------------------------------------
        # STATUS TERAKHIR
        # ----------------------------------------------------

        st.subheader(
            "🔎 Status Analisis Terakhir"
        )

        terakhir = hasil.iloc[-1]

        col1, col2, col3 = st.columns(3)

        with col1:

            if terakhir["pola_threshold"]:

                st.error(
                    f"⚠️ Threshold terdeteksi: "
                    f"{terakhir['sisi']}"
                )

            else:

                st.success(
                    "✅ Tidak ada threshold breach"
                )

        with col2:

            if terakhir["pola_trend"]:

                st.warning(
                    f"📈 Trend "
                    f"{terakhir['arah_trend']}"
                )

            else:

                st.success(
                    "✅ Tidak ada trend kuat"
                )

        with col3:

            if terakhir["pola_sequence"]:

                st.error(
                    "⚠️ Sequence terdeteksi"
                )

            else:

                st.success(
                    "✅ Tidak ada sequence"
                )

        # ----------------------------------------------------
        # EVENT LOG
        # ----------------------------------------------------

        st.subheader(
            "📋 Event Pattern Log"
        )

        log = buat_log(
            hasil
        )

        if log.empty:

            st.info(
                "Belum ada pola CEP yang terdeteksi."
            )

        else:

            st.dataframe(
                log.head(20),
                use_container_width=True
            )

        # ----------------------------------------------------
        # DATA MENTAH
        # ----------------------------------------------------

        with st.expander(
            "📌 Lihat Data Hotspot"
        ):

            st.dataframe(
                titik,
                use_container_width=True
            )