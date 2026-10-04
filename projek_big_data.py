import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import date, timedelta


# ============================================================
# KONFIGURASI DASHBOARD
# ============================================================

st.set_page_config(
    page_title="Dashboard Hotspot Indonesia",
    page_icon="🔥",
    layout="wide"
)

st.title("🔥 Dashboard Analisis Hotspot Kebakaran Indonesia")

st.markdown("""
Dashboard ini menganalisis data hotspot kebakaran Indonesia dari
**NASA FIRMS** menggunakan **Bagan Kendali c' Laney** sebagai
metode utama, serta **Z-Score, Moving Average, dan deteksi pola**
sebagai analisis pendukung.
""")

st.divider()


# ============================================================
# NASA FIRMS
# ============================================================

SENSOR = "VIIRS_NOAA21_NRT"

# Batas wilayah Indonesia
AREA = "95,-11,141,6"


# ============================================================
# AMBIL MAP KEY DARI SECRETS
# ============================================================

try:
    MAP_KEY = st.secrets["FIRMS_MAP_KEY"]
except Exception:
    MAP_KEY = ""


if not MAP_KEY:
    st.error("❌ MAP_KEY NASA FIRMS belum dikonfigurasi.")

    st.info("""
Tambahkan MAP_KEY NASA FIRMS pada:

`.streamlit/secrets.toml`

dengan format:

FIRMS_MAP_KEY = "MAP_KEY_KAMU"
""")

    st.stop()


# ============================================================
# SIDEBAR PARAMETER
# ============================================================

st.sidebar.header("⚙️ Parameter Analisis")

HARI_MUNDUR = st.sidebar.slider(
    "Periode Data (hari)",
    min_value=10,
    max_value=60,
    value=30,
    step=5
)

BASELINE_DAYS = st.sidebar.slider(
    "Baseline c' Laney (hari)",
    min_value=7,
    max_value=30,
    value=30,
    step=1
)

SIGMA_K = st.sidebar.slider(
    "Konstanta Batas Kendali (k)",
    min_value=1.0,
    max_value=4.0,
    value=3.0,
    step=0.5
)

WINDOW_SIZE = st.sidebar.slider(
    "Moving Average",
    min_value=3,
    max_value=14,
    value=7,
    step=1
)

Z_THRESH = st.sidebar.slider(
    "Z-Score Threshold",
    min_value=1.0,
    max_value=3.0,
    value=2.0,
    step=0.1
)

MIN_CONSECUTIVE = st.sidebar.slider(
    "Minimum Hari Berurutan",
    min_value=2,
    max_value=7,
    value=3,
    step=1
)

TREND_LEN = st.sidebar.slider(
    "Panjang Deteksi Tren",
    min_value=2,
    max_value=7,
    value=3,
    step=1
)


# ============================================================
# FUNGSI AMBIL DATA NASA FIRMS
# ============================================================

@st.cache_data(ttl=600)
def tarik_firms(map_key, sensor, area, hari_mundur):

    tanggal_akhir = date.today()
    tanggal_awal = tanggal_akhir - timedelta(days=hari_mundur - 1)

    seluruh_data = []

    # NASA FIRMS Area API menggunakan maksimal 5 hari per request
    tanggal_mulai = tanggal_awal

    while tanggal_mulai <= tanggal_akhir:

        tanggal_selesai = min(
            tanggal_mulai + timedelta(days=4),
            tanggal_akhir
        )

        url = (
            f"https://firms.modaps.eosdis.nasa.gov/api/area/"
            f"csv/{map_key}/{sensor}/{area}/"
            f"{(tanggal_selesai - tanggal_mulai).days + 1}/"
            f"{tanggal_mulai.strftime('%Y-%m-%d')}"
        )

        try:

            response = requests.get(
                url,
                timeout=60
            )

            if response.status_code != 200:

                st.warning(
                    f"""
⚠️ NASA FIRMS gagal mengambil data
{tanggal_mulai} sampai {tanggal_selesai}

HTTP Status: {response.status_code}

Response:
{response.text[:300]}
"""
                )

            else:

                from io import StringIO

                df = pd.read_csv(
                    StringIO(response.text)
                )

                if not df.empty:
                    seluruh_data.append(df)

        except Exception as e:

            st.warning(
                f"""
⚠️ Terjadi kesalahan saat mengambil data
{tanggal_mulai} sampai {tanggal_selesai}

Error:
{e}
"""
            )

        tanggal_mulai = tanggal_selesai + timedelta(days=1)

    if not seluruh_data:
        return pd.DataFrame()

    df = pd.concat(
        seluruh_data,
        ignore_index=True
    )

    return df


# ============================================================
# AGREGASI HARIAN
# ============================================================

def agregasi_harian(df):

    df = df.copy()

    if "acq_date" not in df.columns:
        return pd.DataFrame()

    df["tanggal"] = pd.to_datetime(
        df["acq_date"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["tanggal"]
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
    # HITUNG JUMLAH HOTSPOT PER HARI
    # ========================================================

    harian = (
        df.groupby("tanggal")
        .size()
        .reset_index(
            name="hotspot"
        )
    )

    return harian.sort_values(
        "tanggal"
    ).reset_index(
        drop=True
    )


# ============================================================
# c' LANEY
# ============================================================

def batas_kendali_laney(
    df,
    baseline_days=30,
    k=3
):

    df = df.copy()

    if len(df) < 2:
        return df

    # --------------------------------------------------------
    # Baseline
    # --------------------------------------------------------

    baseline = df[
        "hotspot"
    ].tail(
        min(
            baseline_days,
            len(df)
        )
    )

    cbar = baseline.mean()

    if cbar <= 0:
        cbar = 1

    # --------------------------------------------------------
    # Transformasi Z Laney
    # --------------------------------------------------------

    df["z_laney"] = (
        df["hotspot"] - cbar
    ) / np.sqrt(cbar)

    # --------------------------------------------------------
    # Moving Range
    # --------------------------------------------------------

    df["MR"] = (
        df["z_laney"]
        .diff()
        .abs()
    )

    mr_bar = df["MR"].mean()

    if pd.isna(mr_bar) or mr_bar == 0:
        sigma_z = 1
    else:
        sigma_z = mr_bar / 1.128

    # --------------------------------------------------------
    # Batas kendali pada skala Z
    # --------------------------------------------------------

    ucl_z = k * sigma_z
    lcl_z = -k * sigma_z

    # --------------------------------------------------------
    # Kembali ke skala hotspot
    # --------------------------------------------------------

    simpangan = (
        sigma_z *
        np.sqrt(cbar)
    )

    ucl = cbar + k * simpangan
    lcl = cbar - k * simpangan

    lcl = max(
        0,
        lcl
    )

    # --------------------------------------------------------
    # Simpan batas kendali
    # --------------------------------------------------------

    df["CL"] = cbar
    df["UCL"] = ucl
    df["LCL"] = lcl

    # --------------------------------------------------------
    # ANOMALI UTAMA
    # --------------------------------------------------------

    df["anomali_laney"] = (
        (df["hotspot"] > df["UCL"]) |
        (df["hotspot"] < df["LCL"])
    )

    return df


# ============================================================
# DETEKSI CONSECUTIVE
# ============================================================

def consecutive_true_count(series):

    hasil = []
    count = 0

    for value in series:

        if value:
            count += 1
        else:
            count = 0

        hasil.append(count)

    return hasil


# ============================================================
# DETEKSI POLA ARAH
# ============================================================

def consecutive_same_sign(series):

    hasil = []

    previous = None
    count = 0

    for value in series:

        if value > 0:

            current = "naik"

        elif value < 0:

            current = "turun"

        else:

            current = "netral"

        if current == previous:

            count += 1

        else:

            count = 1

        hasil.append(
            f"{current} ({count})"
        )

        previous = current

    return hasil


# ============================================================
# ANALISIS UTAMA
# ============================================================

def analisis(
    df,
    baseline_days,
    k,
    window_size,
    z_thresh,
    min_consecutive,
    trend_len
):

    df = df.copy()

    # --------------------------------------------------------
    # c' LANEY
    # --------------------------------------------------------

    df = batas_kendali_laney(
        df,
        baseline_days,
        k
    )

    # --------------------------------------------------------
    # Z-SCORE
    # --------------------------------------------------------

    mean_hotspot = df[
        "hotspot"
    ].mean()

    std_hotspot = df[
        "hotspot"
    ].std()

    if pd.isna(std_hotspot) or std_hotspot == 0:

        df["z_score"] = 0

    else:

        df["z_score"] = (
            df["hotspot"] -
            mean_hotspot
        ) / std_hotspot

    df["anomali_z"] = (
        df["z_score"].abs() >
        z_thresh
    )

    # --------------------------------------------------------
    # MOVING AVERAGE
    # --------------------------------------------------------

    df["moving_average"] = (
        df["hotspot"]
        .rolling(
            window=window_size,
            min_periods=1
        )
        .mean()
    )

    # --------------------------------------------------------
    # SELISIH DARI MOVING AVERAGE
    # --------------------------------------------------------

    df["selisih_ma"] = (
        df["hotspot"] -
        df["moving_average"]
    )

    # --------------------------------------------------------
    # CONSECUTIVE ANOMALI LANEY
    # --------------------------------------------------------

    df["consecutive_laney"] = (
        consecutive_true_count(
            df["anomali_laney"]
        )
    )

    df["pola_laney"] = (
        df["consecutive_laney"] >=
        min_consecutive
    )

    # --------------------------------------------------------
    # DETEKSI ARAH
    # --------------------------------------------------------

    df["perubahan"] = (
        df["hotspot"]
        .diff()
    )

    df["arah"] = (
        consecutive_same_sign(
            df["perubahan"]
            .fillna(0)
        )
    )

    # --------------------------------------------------------
    # TREND
    # --------------------------------------------------------

    df["trend"] = (
        df["hotspot"]
        .rolling(
            window=trend_len,
            min_periods=trend_len
        )
        .apply(
            lambda x:
            np.polyfit(
                np.arange(len(x)),
                x,
                1
            )[0]
            if len(x) >= 2
            else 0
        )
    )

    # --------------------------------------------------------
    # KATEGORI TREND
    # --------------------------------------------------------

    df["kategori_trend"] = np.select(
        [
            df["trend"] > 0,
            df["trend"] < 0
        ],
        [
            "Meningkat",
            "Menurun"
        ],
        default="Stabil"
    )

    # --------------------------------------------------------
    # ANOMALI GABUNGAN
    # --------------------------------------------------------

    df["anomali_gabungan"] = (
        df["anomali_laney"] |
        df["anomali_z"]
    )

    return df


# ============================================================
# AMBIL DATA
# ============================================================

with st.spinner(
    "🔥 Mengambil data hotspot dari NASA FIRMS..."
):

    data_raw = tarik_firms(
        MAP_KEY,
        SENSOR,
        AREA,
        HARI_MUNDUR
    )


# ============================================================
# CEK DATA
# ============================================================

if data_raw.empty:

    st.error(
        "❌ Data hotspot tidak berhasil diperoleh."
    )

    st.info("""
Kemungkinan penyebab:

• MAP_KEY tidak valid
• NASA FIRMS sedang mengalami gangguan
• Data belum tersedia
• Koneksi internet bermasalah

Periksa juga pesan HTTP Status di atas.
""")

    st.stop()


# ============================================================
# AGREGASI
# ============================================================

data_harian = agregasi_harian(
    data_raw
)


if data_harian.empty:

    st.error(
        "❌ Data hotspot harian tidak tersedia."
    )

    st.stop()


# ============================================================
# ANALISIS
# ============================================================

df = analisis(
    data_harian,
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

total_hotspot = int(
    df["hotspot"].sum()
)

rata_rata = (
    df["hotspot"].mean()
)

hotspot_terakhir = int(
    df.iloc[-1]["hotspot"]
)

# ANOMALI UTAMA = c' LANEY
total_anomali = int(
    df["anomali_laney"].sum()
)


# ============================================================
# HEADER KPI
# ============================================================

st.subheader(
    "📊 Ringkasan Data"
)

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


st.caption(
    "Total hotspot merupakan jumlah deteksi hotspot satelit, "
    "bukan jumlah kejadian kebakaran."
)


# ============================================================
# STATUS TERAKHIR
# ============================================================

st.divider()

tanggal_terakhir = (
    df.iloc[-1]["tanggal"]
)

status_terakhir = (
    "⚠️ ANOMALI"
    if df.iloc[-1]["anomali_laney"]
    else "✅ NORMAL"
)


st.subheader(
    "📍 Status Hotspot Terakhir"
)

col1, col2, col3 = st.columns(3)


with col1:

    st.write(
        "**Tanggal:**"
    )

    st.write(
        tanggal_terakhir.strftime(
            "%d %B %Y"
        )
    )


with col2:

    st.write(
        "**Jumlah Hotspot:**"
    )

    st.write(
        f"{hotspot_terakhir:,}"
    )


with col3:

    st.write(
        "**Status c' Laney:**"
    )

    st.write(
        status_terakhir
    )


# ============================================================
# GRAFIK c' LANEY
# ============================================================

st.divider()

st.subheader(
    "📈 Bagan Kendali c' Laney"
)

fig_laney = go.Figure()


fig_laney.add_trace(
    go.Scatter(
        x=df["tanggal"],
        y=df["hotspot"],
        mode="lines+markers",
        name="Hotspot"
    )
)


fig_laney.add_trace(
    go.Scatter(
        x=df["tanggal"],
        y=df["UCL"],
        mode="lines",
        name="UCL"
    )
)


fig_laney.add_trace(
    go.Scatter(
        x=df["tanggal"],
        y=df["CL"],
        mode="lines",
        name="CL"
    )
)


fig_laney.add_trace(
    go.Scatter(
        x=df["tanggal"],
        y=df["LCL"],
        mode="lines",
        name="LCL"
    )
)


# Titik anomali
anomali_df = df[
    df["anomali_laney"]
]


if not anomali_df.empty:

    fig_laney.add_trace(
        go.Scatter(
            x=anomali_df["tanggal"],
            y=anomali_df["hotspot"],
            mode="markers",
            name="Anomali"
        )
    )


fig_laney.update_layout(
    xaxis_title="Tanggal",
    yaxis_title="Jumlah Hotspot",
    hovermode="x unified",
    height=500
)


st.plotly_chart(
    fig_laney,
    use_container_width=True
)


st.info("""
**Interpretasi c' Laney:**

Suatu hari dikategorikan sebagai anomali apabila jumlah hotspot
berada di atas batas kendali atas (UCL) atau di bawah batas
kendali bawah (LCL).

Dalam dashboard ini, c' Laney digunakan sebagai metode utama
untuk menentukan Total Anomali.
""")


# ============================================================
# GRAFIK Z-SCORE
# ============================================================

st.divider()

st.subheader(
    "📊 Analisis Z-Score"
)

fig_z = go.Figure()


fig_z.add_trace(
    go.Scatter(
        x=df["tanggal"],
        y=df["z_score"],
        mode="lines+markers",
        name="Z-Score"
    )
)


fig_z.add_hline(
    y=Z_THRESH,
    line_dash="dash",
    annotation_text=f"+{Z_THRESH}"
)


fig_z.add_hline(
    y=-Z_THRESH,
    line_dash="dash",
    annotation_text=f"-{Z_THRESH}"
)


fig_z.add_hline(
    y=0,
    line_dash="dot"
)


fig_z.update_layout(
    xaxis_title="Tanggal",
    yaxis_title="Z-Score",
    hovermode="x unified",
    height=450
)


st.plotly_chart(
    fig_z,
    use_container_width=True
)


jumlah_anomali_z = int(
    df["anomali_z"].sum()
)


st.write(
    f"📌 Berdasarkan Z-Score, terdapat "
    f"**{jumlah_anomali_z} hari** yang melewati "
    f"threshold ±{Z_THRESH}."
)


# ============================================================
# MOVING AVERAGE
# ============================================================

st.divider()

st.subheader(
    f"📉 Moving Average ({WINDOW_SIZE} Hari)"
)

fig_ma = go.Figure()


fig_ma.add_trace(
    go.Scatter(
        x=df["tanggal"],
        y=df["hotspot"],
        mode="lines",
        name="Hotspot"
    )
)


fig_ma.add_trace(
    go.Scatter(
        x=df["tanggal"],
        y=df["moving_average"],
        mode="lines",
        name=f"MA {WINDOW_SIZE} Hari"
    )
)


fig_ma.update_layout(
    xaxis_title="Tanggal",
    yaxis_title="Jumlah Hotspot",
    hovermode="x unified",
    height=450
)


st.plotly_chart(
    fig_ma,
    use_container_width=True
)


# ============================================================
# DETEKSI TREND
# ============================================================

st.divider()

st.subheader(
    "📈 Deteksi Tren Hotspot"
)

trend_terakhir = df.iloc[-1][
    "kategori_trend"
]

nilai_trend = df.iloc[-1][
    "trend"
]


col1, col2 = st.columns(2)


with col1:

    st.metric(
        "Trend Terakhir",
        trend_terakhir
    )


with col2:

    st.metric(
        "Kemiringan Trend",
        f"{nilai_trend:,.2f}"
        if pd.notna(nilai_trend)
        else "N/A"
    )


# ============================================================
# DETEKSI POLA
# ============================================================

st.divider()

st.subheader(
    "🔎 Deteksi Pola Anomali Berurutan"
)

pola_df = df[
    df["pola_laney"]
]


if pola_df.empty:

    st.success(
        "✅ Tidak ditemukan anomali c' Laney "
        f"yang terjadi selama minimal "
        f"{MIN_CONSECUTIVE} hari berturut-turut."
    )

else:

    st.warning(
        f"⚠️ Terdapat {len(pola_df)} titik "
        "yang termasuk rangkaian anomali."
    )


# ============================================================
# TABEL ANALISIS
# ============================================================

st.divider()

st.subheader(
    "📋 Tabel Hasil Analisis"
)

tabel = df[
    [
        "tanggal",
        "hotspot",
        "CL",
        "UCL",
        "LCL",
        "anomali_laney",
        "z_score",
        "anomali_z",
        "moving_average",
        "kategori_trend"
    ]
].copy()


tabel["tanggal"] = (
    tabel["tanggal"]
    .dt.strftime("%Y-%m-%d")
)


tabel["anomali_laney"] = (
    tabel["anomali_laney"]
    .map(
        {
            True: "⚠️ Ya",
            False: "Normal"
        }
    )
)


tabel["anomali_z"] = (
    tabel["anomali_z"]
    .map(
        {
            True: "⚠️ Ya",
            False: "Normal"
        }
    )
)


tabel = tabel.rename(
    columns={
        "tanggal": "Tanggal",
        "hotspot": "Hotspot",
        "CL": "CL",
        "UCL": "UCL",
        "LCL": "LCL",
        "anomali_laney": "Anomali c' Laney",
        "z_score": "Z-Score",
        "anomali_z": "Anomali Z-Score",
        "moving_average": "Moving Average",
        "kategori_trend": "Trend"
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

st.divider()

with st.expander(
    "📄 Lihat Data Hotspot Mentah"
):

    st.dataframe(
        data_raw,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# INFORMASI METODE
# ============================================================

st.divider()

st.subheader(
    "📚 Metode yang Digunakan"
)

st.markdown("""
### 1. Bagan Kendali c' Laney
Digunakan sebagai **metode utama** untuk mengetahui apakah
jumlah hotspot harian masih berada dalam batas kendali statistik.

- **CL** = Center Line
- **UCL** = Upper Control Limit
- **LCL** = Lower Control Limit
- Titik di luar UCL/LCL dikategorikan sebagai **anomali**.

### 2. Z-Score
Digunakan untuk mengetahui seberapa jauh jumlah hotspot
menyimpang dari rata-rata dalam satuan standar deviasi.

### 3. Moving Average
Digunakan untuk melihat kecenderungan atau pola pergerakan
jumlah hotspot dari waktu ke waktu.

### 4. Deteksi Pola
Digunakan untuk mengetahui apakah kondisi anomali terjadi
secara berurutan selama beberapa hari.
""")


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Sumber data: NASA FIRMS | "
    "Analisis statistik: c' Laney, Z-Score, Moving Average"
)