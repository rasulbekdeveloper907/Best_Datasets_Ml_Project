# import_csv_to_sql_windows_auth_with_db_creation.py

import pandas as pd
import pyodbc
from sqlalchemy import create_engine

# -----------------------------
# 1️⃣ CSV faylni o'qish
# -----------------------------
csv_file = r"C:\Users\Rasulbek907\Desktop\Project_MP\Data\Preprosessed\Missing_Value.csv"
df = pd.read_csv(csv_file)
print(f"✅ CSV fayl o'qildi: {len(df)} yozuv")

# -----------------------------
# 2️⃣ SQL Server ulanishi (Windows Authentication)
# -----------------------------
server = 'RASULBEK\\SQLEXPRESS'  # SQL Server instance
database = 'Keggle_Database'     # Maqsadli database
driver = 'ODBC Driver 17 for SQL Server'  # ODBC driver

# Avval master DB orqali ulanish
conn_str_master = f'DRIVER={{{driver}}};SERVER={server};DATABASE=master;Trusted_Connection=yes'
conn = pyodbc.connect(conn_str_master)
cursor = conn.cursor()

# Agar database mavjud bo'lmasa, yaratish
cursor.execute(f"IF DB_ID(N'{database}') IS NULL CREATE DATABASE [{database}]")
conn.commit()
cursor.close()
conn.close()
print(f"✅ Database '{database}' tekshirildi / yaratildi")

# SQLAlchemy engine yaratish
engine = create_engine(
    f"mssql+pyodbc://{server}/{database}?driver={driver}&trusted_connection=yes",
    echo=True  # SQL so'rovlarini ko'rish uchun
)

# -----------------------------
# 3️⃣ CSV ni SQL ga yozish
# -----------------------------
table_name = 'Keggle_Dataset'  # Jadval nomi

try:
    with engine.begin() as conn:
        df.to_sql(table_name, con=conn, if_exists='replace', index=False, schema='dbo')
    print(f"✅ CSV muvaffaqiyatli '{table_name}' jadvaliga yuklandi!")
except Exception as e:
    print(f"❌ Xato yuz berdi: {e}")
