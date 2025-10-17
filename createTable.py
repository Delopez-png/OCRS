import os
import pandas as pd
import psycopg2
from psycopg2 import errors
from dotenv import load_dotenv

# Memuat variabel lingkungan dari file .env
load_dotenv()

# Mengambil variabel lingkungan
DB_NAME = os.getenv("DB_DB")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")

def create_database_if_not_exists():
    """
    Membuat database PostgreSQL jika belum ada.
    Terhubung ke database default 'postgres' untuk menjalankan perintah ini.
    """
    try:
        with psycopg2.connect(
            dbname="postgres",
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        ) as conn:
            conn.autocommit = True
            with conn.cursor() as cursor:
                cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", (DB_NAME,))
                exists = cursor.fetchone()

                if not exists:
                    cursor.execute(f'CREATE DATABASE "{DB_NAME}"')
                    print(f"Database '{DB_NAME}' berhasil dibuat.")
                else:
                    print(f"Database '{DB_NAME}' sudah ada.")
    except Exception as e:
        print(f"Error saat membuat database: {e}")

def setup_tables_and_triggers():
    """
    Menyiapkan tabel dan triggers yang diperlukan di database target.
    """
    try:
        with psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        ) as conn:
            with conn.cursor() as cursor:
                # --- Membuat Tabel data_pusat ---
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS data_pusat (
                    id SERIAL PRIMARY KEY,
                    brand TEXT,
                    sub_brand_as TEXT,
                    brand_group_as TEXT,
                    description TEXT,
                    barcode_pieces NUMERIC UNIQUE,
                    harga_ptt NUMERIC,
                    qty_per_carton BIGINT,
                    harga_ppn NUMERIC,
                    harga_jual_saran NUMERIC
                );
                """)
                print("Tabel data_pusat siap digunakan.")

                # --- Membuat Tabel keywords ---
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS keywords (
                    keyword TEXT PRIMARY KEY,
                    normalized_value TEXT NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                );
                """)
                print("Tabel keywords siap digunakan.")

                # --- Membuat Tabel ocr_items ---
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS ocr_items (
                    id SERIAL PRIMARY KEY,
                    invoice_number TEXT,
                    tanggal DATE,
                    nama_toko TEXT,
                    alamat TEXT,
                    phone TEXT,
                    email TEXT,
                    total_belanja NUMERIC,
                    is_confirmed BOOLEAN DEFAULT FALSE,
                    feedback TEXT,
                    created_at TIMESTAMP DEFAULT NOW(),
                    confirmed_at TIMESTAMP
                );
                """)
                print("Tabel ocr_items siap digunakan.")

                # --- Membuat Tabel ocr_results ---
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS ocr_results (
                    id SERIAL PRIMARY KEY,
                    receipt_id INTEGER REFERENCES ocr_items(id) ON DELETE CASCADE,
                    ocr_name TEXT,
                    ocr_quantity NUMERIC,
                    ocr_price NUMERIC,
                    ocr_total NUMERIC,
                    text_accuracy NUMERIC,
                    price_accuracy NUMERIC,
                    final_score NUMERIC,
                    matched_item_id INTEGER REFERENCES data_pusat(id),
                    matched_price NUMERIC,
                    matched_name TEXT,
                    keywords TEXT[],
                    is_correct BOOLEAN,
                    feedback TEXT,
                    confirmed_at TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """)
                print("Tabel ocr_results siap digunakan.")

                # --- Fungsi dan Triggers (Diperiksa keberadaannya sebelum dibuat) ---
                cursor.execute("""
                CREATE OR REPLACE FUNCTION update_timestamp()
                RETURNS TRIGGER AS $$
                BEGIN
                    NEW.updated_at = NOW();
                    RETURN NEW;
                END;
                $$ LANGUAGE plpgsql;
                
                DO $$
                BEGIN
                    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'update_keywords_timestamp') THEN
                        CREATE TRIGGER update_keywords_timestamp
                        BEFORE UPDATE ON keywords
                        FOR EACH ROW
                        EXECUTE FUNCTION update_timestamp();
                    END IF;
                END
                $$;
                
                DO $$
                BEGIN
                    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'update_ocr_results_timestamp') THEN
                        CREATE TRIGGER update_ocr_results_timestamp
                        BEFORE UPDATE ON ocr_results
                        FOR EACH ROW
                        EXECUTE FUNCTION update_timestamp();
                    END IF;
                END
                $$;
                
                CREATE OR REPLACE FUNCTION update_ocr_item_confirmation()
                RETURNS TRIGGER AS $$
                DECLARE
                    all_confirmed BOOLEAN;
                BEGIN
                    SELECT BOOL_AND(is_correct IS NOT NULL) INTO all_confirmed
                    FROM ocr_results
                    WHERE receipt_id = NEW.receipt_id;

                    IF all_confirmed THEN
                        UPDATE ocr_items
                        SET is_confirmed = TRUE, confirmed_at = NOW()
                        WHERE id = NEW.receipt_id;
                    END IF;
                    RETURN NEW;
                END;
                $$ LANGUAGE plpgsql;

                DO $$
                BEGIN
                    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'check_all_items_confirmed') THEN
                        CREATE TRIGGER check_all_items_confirmed
                        AFTER INSERT OR UPDATE ON ocr_results
                        FOR EACH ROW
                        EXECUTE FUNCTION update_ocr_item_confirmation();
                    END IF;
                END
                $$;
                """)
                conn.commit()
                print("Semua triggers berhasil disiapkan.")

    except psycopg2.OperationalError as e:
        print(f"Error koneksi ke database: {e}. Pastikan database '{DB_NAME}' sudah dibuat.")
    except Exception as e:
        print(f"Error saat menyiapkan tabel dan triggers: {e}")

def load_data_from_files():
    """
    Memuat data dari file Excel dan teks ke dalam database.
    """
    try:
        with psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        ) as conn:
            with conn.cursor() as cursor:
                # --- Muat data dari file Excel ke data_pusat ---
                excel_files = [
                    ('SKU.xlsx', 'All NIVEA PL Mei 2024'),
                    ('SKUT.xlsx', 'All NIVEA PL Mei 2024')
                ]
                for excel_file, sheet_name in excel_files:
                    try:
                        if not os.path.exists(excel_file):
                            print(f"Peringatan: File Excel '{excel_file}' tidak ditemukan. Melewatkan.")
                            continue
                        df = pd.read_excel(excel_file, usecols=range(9), skiprows=2, sheet_name=sheet_name)
                        df.rename(columns=lambda x: x.strip(), inplace=True)
                        df = df.dropna()

                        cursor.execute("SELECT COUNT(*) FROM data_pusat")
                        count = cursor.fetchone()[0]

                        # Andalkan ON CONFLICT untuk menangani data yang ada, membuat skrip ini idempoten.
                        for _, row in df.iterrows():
                            cursor.execute("""
                                INSERT INTO data_pusat (
                                    brand, sub_brand_as, brand_group_as, description, barcode_pieces,
                                    harga_ptt, qty_per_carton, harga_ppn, harga_jual_saran
                                )
                                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                                ON CONFLICT (barcode_pieces) DO NOTHING;
                            """, (
                                row.get('Brand'),
                                row.get('Sub Brand AS'),
                                row.get('Brand Group AS'),
                                row.get('Description'),
                                str(row.get('Barcode (Pieces)')),
                                row.get('Harga PTT/Sebelum PPN'),
                                row.get('Kuantiti per Karton/Dus'),
                                row.get('Harga PPN'),
                                row.get('Harga Jual ke Konsumen yg Disarankan')
                            ))
                        conn.commit()
                        print(f"Data dari '{excel_file}' telah diproses.")
                    except Exception as e:
                        print(f"Error saat membaca/menyisipkan data Excel: {e}")
                        conn.rollback()

                # --- Muat data normalisasi.txt ke keywords ---
                normalisasi_file = 'normalisasi.txt'
                if os.path.exists(normalisasi_file):
                    inserted_count = 0
                    with open(normalisasi_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            if '=' in line:
                                keyword, normalized = line.split('=', 1)
                            elif '->' in line:
                                keyword, normalized = line.split('->', 1)
                            else:
                                continue
                            cursor.execute("""
                                INSERT INTO keywords (keyword, normalized_value)
                                VALUES (%s, %s)
                                ON CONFLICT (keyword) DO NOTHING;
                            """, (keyword.strip().lower(), normalized.strip().lower()))
                            if cursor.rowcount > 0:
                                inserted_count += 1
                    conn.commit()
                    print(f"{inserted_count} keywords dari '{normalisasi_file}' berhasil dimasukkan.")
                else:
                    print(f"File '{normalisasi_file}' tidak ditemukan.")

    except psycopg2.OperationalError as e:
        print(f"Error koneksi ke database: {e}. Pastikan database '{DB_NAME}' sudah dibuat.")
    except Exception as e:
        print(f"Error saat memuat data: {e}")        

if __name__ == "__main__":
    create_database_if_not_exists()
    setup_tables_and_triggers()
    load_data_from_files()
    print("Skrip inisialisasi database selesai.")